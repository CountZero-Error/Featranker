import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from featranker import FeatureRanker, Report
from featranker.ablation import _default_ks

N_FEATURES = 6
N_TRAIN, N_EVAL = 160, 80
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N_TRAIN + N_EVAL, N_FEATURES))
Y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * RNG.normal(size=len(X))
NAMES = [f"x{i}" for i in range(N_FEATURES)]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, None)
W = RNG.uniform(0.5, 2.0, size=len(X))
FITS: list = []


class Recorder(RegressorMixin, BaseEstimator):
    """Linear model that logs what its fit() was given."""

    def fit(self, X, y, sample_weight=None):
        FITS.append(
            {
                "shape": np.shape(X),
                "columns": list(getattr(X, "columns", [])),
                "has_nan": bool(np.isnan(np.asarray(X, dtype=float)).any()),
                "weights": None if sample_weight is None else np.array(sample_weight),
            }
        )
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class FailsOnSubset(RegressorMixin, BaseEstimator):
    """Fits on all columns only."""

    def fit(self, X, y):
        if np.shape(X)[1] < N_FEATURES:
            raise ValueError("needs every column")
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class Worse(RegressorMixin, BaseEstimator):
    """Never beats the dummy; counts fits."""

    fits = 0

    def fit(self, X, y):
        type(self).fits += 1
        return self

    def predict(self, X):
        return np.full(len(X), 1e3)


def make_ranker(models=None, **kwargs):
    models = models or {"lr": LinearRegression(), "tree": DecisionTreeRegressor(random_state=0)}
    return FeatureRanker("reg", models=models, verbose=False, **kwargs)


def ranked(models=None, ranker_kwargs=None, feature_groups=None, **fit_kwargs):
    ranker = make_ranker(models, **(ranker_kwargs or {}))
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, **fit_kwargs)
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=2, random_state=0,
        feature_groups=feature_groups,
    )
    return ranker, report


def ablate(ranker, report, **kwargs):
    return ranker.ablate(X[TRAIN], Y[TRAIN], X[EVAL], Y[EVAL], report, **kwargs)


def median_at(block, k):
    return next(row["median_score"] for row in block["curve"] if row["k"] == k)


def test_signal_groups_lead_the_curve():
    ranker, report = ranked()

    block = ablate(ranker, report)["ablation"]

    assert block["curve"][1]["feature_groups"] == [r["feature_group"] for r in report["consensus"][:2]]
    assert median_at(block, 2) > median_at(block, 1) + 0.1
    assert abs(median_at(block, 2) - median_at(block, N_FEATURES)) < 0.05


def test_block_shape_and_report_unchanged():
    ranker, report = ranked()
    before = copy.deepcopy(report)

    result = ablate(ranker, report)

    assert isinstance(result, Report)
    assert result is not report
    assert report == before and "ablation" not in report
    assert {k: v for k, v in result.items() if k != "ablation"} == before
    block = result["ablation"]
    assert set(block) == {
        "scoring", "models", "baseline_score", "n_groups", "curve", "failures",
    }
    assert block["scoring"] == "r2"
    assert block["models"] == ["lr", "tree"]
    assert block["baseline_score"] == report["baseline_score"]
    assert block["n_groups"] == N_FEATURES
    assert block["failures"] == []
    assert set(block["curve"][0]) == {
        "k", "feature_groups", "n_features", "scores", "median_score",
    }
    assert set(block["curve"][0]["scores"]) == {"lr", "tree"}


def test_result_is_independent_of_the_input_report():
    ranker, report = ranked()

    result = ablate(ranker, report)
    result["consensus"][0]["median_rank"] = 99.0

    assert report["consensus"][0]["median_rank"] != 99.0


def test_result_is_strict_json():
    ranker, report = ranked()

    text = json.dumps(ablate(ranker, report), allow_nan=False)

    assert json.loads(text)["ablation"]["curve"]


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, [1]), (3, [1, 2, 3]), (7, [1, 2, 3, 5, 7]), (10, [1, 2, 3, 5, 10]),
     (120, [1, 2, 3, 5, 10, 20, 50, 100, 120]), (2000, [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])],
)
def test_default_ks(n, expected):
    assert _default_ks(n) == expected


def test_default_curve_ks_follow_the_group_count():
    ranker, report = ranked()

    block = ablate(ranker, report)["ablation"]

    assert [row["k"] for row in block["curve"]] == [1, 2, 3, 5, 6]


def test_given_ks_are_deduplicated_and_sorted():
    ranker, report = ranked()

    block = ablate(ranker, report, ks=[4, 1, 4, 2])["ablation"]

    assert [row["k"] for row in block["curve"]] == [1, 2, 4]
    assert [row["n_features"] for row in block["curve"]] == [1, 2, 4]


@pytest.mark.parametrize(
    "ks", [[0], [-1], [True], [1.5], ["2"], [N_FEATURES + 1], [], 3],
)
def test_invalid_ks_raise(ks):
    ranker, report = ranked()

    with pytest.raises(ValueError, match="ks"):
        ablate(ranker, report, ks=ks)


def test_unfitted_ranker_raises():
    _, report = ranked()

    with pytest.raises(ValueError, match="fit"):
        ablate(make_ranker(), report)


def test_resampled_report_raises():
    ranker, _ = ranked()
    report = make_ranker().rank_resampled(X[TRAIN], Y[TRAIN], scoring="r2", cv=3, feature_names=NAMES,
                                   n_repeats=1, random_state=0)

    with pytest.raises(ValueError, match="resampled"):
        ablate(ranker, report)


def test_empty_consensus_raises():
    ranker, report = ranked()

    with pytest.raises(ValueError, match="consensus"):
        ablate(ranker, Report({**report, "consensus": []}))


def test_other_training_data_raises():
    ranker, report = ranked()
    other = X[TRAIN].copy()
    other[0, 0] += 1.0

    with pytest.raises(ValueError, match="training data"):
        ranker.ablate(other, Y[TRAIN], X[EVAL], Y[EVAL], report)


def test_report_without_a_fingerprint_is_not_checked():
    ranker, report = ranked()
    run = {**report["run"], "train_data_sha256": None}

    block = ablate(ranker, Report({**report, "run": run}), ks=[1])["ablation"]

    assert len(block["curve"]) == 1


def test_unresolvable_scorer_name_raises():
    ranker, report = ranked()

    with pytest.raises(ValueError):
        ablate(ranker, Report({**report, "scoring": "my_callable"}))


def test_scoring_override_is_used_with_its_own_baseline():
    ranker, report = ranked()

    block = ablate(ranker, report, scoring="neg_mean_squared_error", ks=[2])["ablation"]

    assert block["scoring"] == "neg_mean_squared_error"
    assert block["baseline_score"] < 0
    assert block["curve"][0]["median_score"] < 0


def test_eval_columns_must_match_training_columns():
    ranker, report = ranked()

    with pytest.raises(ValueError):
        ranker.ablate(X[TRAIN], Y[TRAIN], X[EVAL][:, :3], Y[EVAL], report)


def test_only_voting_models_are_refit():
    Worse.fits = 0
    ranker, report = ranked({"lr": LinearRegression(), "worse": Worse()})
    assert report["models"]["worse"]["in_consensus"] is False
    fits_after_ranking = Worse.fits

    block = ablate(ranker, report, ks=[1, 2])["ablation"]

    assert block["models"] == ["lr"]
    assert set(block["curve"][0]["scores"]) == {"lr"}
    assert Worse.fits == fits_after_ranking


def test_each_k_refits_on_the_union_of_the_top_groups_columns():
    groups = {"pair": ["x0", "x1"], "x2": ["x2"], "x3": ["x3"], "x4": ["x4"], "x5": ["x5"]}
    ranker, report = ranked({"rec": Recorder()}, feature_groups=groups)
    FITS.clear()

    block = ablate(ranker, report, ks=[1, 2, 5])["ablation"]

    assert report["consensus"][0]["feature_group"] == "pair"
    assert [row["n_features"] for row in block["curve"]] == [2, 3, 6]
    assert [fit["shape"] for fit in FITS] == [(N_TRAIN, 2), (N_TRAIN, 3), (N_TRAIN, 6)]


def test_columns_keep_training_order():
    ranker, report = ranked(
        {"rec": Recorder()}, ranker_kwargs={"model_input": "dataframe"},
    )
    top = [row["feature_group"] for row in report["consensus"][:3]]
    FITS.clear()

    ablate(ranker, report, ks=[3])

    assert FITS[0]["columns"] == [name for name in NAMES if name in top]


def test_dataframe_input_is_passed_to_models_with_names():
    frame = pd.DataFrame(X, columns=NAMES)
    ranker = make_ranker({"rec": Recorder()}, model_input="dataframe")
    ranker.fit(frame[:N_TRAIN], Y[TRAIN])
    report = ranker.rank_features(frame[N_TRAIN:], Y[EVAL], scoring="r2", n_repeats=1)
    FITS.clear()

    block = ranker.ablate(
        frame[:N_TRAIN], Y[TRAIN], frame[N_TRAIN:], Y[EVAL], report, ks=[2]
    )["ablation"]

    assert len(FITS[0]["columns"]) == 2
    assert block["curve"][0]["scores"]["rec"] is not None


def test_categorical_columns_pass_through():
    rng = np.random.default_rng(1)
    city = rng.choice(["a", "b", "c"], size=len(X))
    effect = np.select([city == "a", city == "b"], [4.0, -4.0], 0.0)
    frame = pd.DataFrame(X[:, :3], columns=NAMES[:3]).assign(city=city)
    y = effect + 0.1 * rng.normal(size=len(X))
    ranker = make_ranker(
        {"tree": DecisionTreeRegressor(random_state=0)}, categorical=["city"]
    )
    ranker.fit(frame[:N_TRAIN], y[:N_TRAIN])
    report = ranker.rank_features(frame[N_TRAIN:], y[N_TRAIN:], scoring="r2", n_repeats=2)
    assert report["consensus"][0]["feature_group"] == "city"

    block = ranker.ablate(
        frame[:N_TRAIN], y[:N_TRAIN], frame[N_TRAIN:], y[N_TRAIN:], report, ks=[1, 4]
    )["ablation"]

    assert block["failures"] == []
    assert block["curve"][0]["feature_groups"] == ["city"]
    assert block["curve"][0]["median_score"] > 0.8


def test_missing_values_pass_through():
    with_gaps = X.copy()
    with_gaps[::7, 0] = np.nan
    ranker = make_ranker({"hgb": HistGradientBoostingRegressor(max_iter=20)}, missing="impute")
    ranker.fit(with_gaps[TRAIN], Y[TRAIN], feature_names=NAMES)
    report = ranker.rank_features(with_gaps[EVAL], Y[EVAL], scoring="r2", n_repeats=1)

    block = ranker.ablate(
        with_gaps[TRAIN], Y[TRAIN], with_gaps[EVAL], Y[EVAL], report, ks=[N_FEATURES]
    )["ablation"]

    assert block["failures"] == []
    assert block["curve"][0]["scores"]["hgb"] is not None


def test_sample_weights_reach_the_refit_and_the_score():
    ranker, report = ranked({"rec": Recorder()}, sample_weight=W[TRAIN])
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, sample_weight=W[EVAL]
    )
    FITS.clear()

    block = ranker.ablate(
        X[TRAIN], Y[TRAIN], X[EVAL], Y[EVAL], report, ks=[1],
        sample_weight=W[TRAIN], eval_sample_weight=W[EVAL],
    )["ablation"]

    assert np.array_equal(FITS[0]["weights"], W[TRAIN])
    top = NAMES.index(report["consensus"][0]["feature_group"])
    model = LinearRegression().fit(X[TRAIN][:, [top]], Y[TRAIN], sample_weight=W[TRAIN])
    expected = r2_score(Y[EVAL], model.predict(X[EVAL][:, [top]]), sample_weight=W[EVAL])
    assert block["curve"][0]["scores"]["rec"] == pytest.approx(expected)


def test_a_failing_model_gets_null_scores_and_failure_entries():
    ranker, report = ranked({"lr": LinearRegression(), "boom": FailsOnSubset()})

    block = ablate(ranker, report, ks=[1, 3, N_FEATURES])["ablation"]

    assert [row["scores"]["boom"] for row in block["curve"][:2]] == [None, None]
    assert block["curve"][2]["scores"]["boom"] is not None
    assert block["curve"][0]["median_score"] == block["curve"][0]["scores"]["lr"]
    assert [(f["model"], f["stage"], f["k"]) for f in block["failures"]] == [
        ("boom", "fit", 1), ("boom", "fit", 3),
    ]
    assert set(block["failures"][0]) == {"model", "stage", "message", "k"}
    assert "needs every column" in block["failures"][0]["message"]
    json.dumps(block, allow_nan=False)


def test_every_model_failing_at_a_k_gives_a_null_median():
    ranker, report = ranked({"boom": FailsOnSubset(), "boom2": FailsOnSubset()})

    block = ablate(ranker, report, ks=[2, N_FEATURES])["ablation"]

    assert block["curve"][0]["scores"] == {"boom": None, "boom2": None}
    assert block["curve"][0]["median_score"] is None
    assert block["curve"][1]["median_score"] is not None
    assert {f["model"] for f in block["failures"]} == {"boom", "boom2"}
    assert {f["k"] for f in block["failures"]} == {2}


def test_built_in_models_are_refit_with_their_preprocessing():
    ranker = FeatureRanker("reg", group="fast", verbose=False)
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    report = ranker.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0)
    voters = [n for n, m in report["models"].items() if m["in_consensus"]]

    block = ablate(ranker, report, ks=[2])["ablation"]

    assert block["models"] == voters
    assert block["failures"] == []
    assert block["curve"][0]["median_score"] > 0.5


class PredictFailsOnSubset(RegressorMixin, BaseEstimator):
    """Fits on any columns, predicts on all of them only."""

    def fit(self, X, y):
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        if np.shape(X)[1] < N_FEATURES:
            raise ValueError("predicts on every column only")
        return self.model_.predict(X)


def test_baseline_is_the_sub_rankers_own_on_all_eval_rows():
    ranker = make_ranker()
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0, max_samples=20
    )
    dummy = np.full(len(Y[EVAL]), Y[TRAIN].mean())

    block = ablate(ranker, report, ks=[1, 2])["ablation"]
    overridden = ablate(ranker, report, ks=[1], scoring="neg_mean_squared_error")["ablation"]

    assert block["baseline_score"] == pytest.approx(r2_score(Y[EVAL], dummy))
    assert block["baseline_score"] != report["baseline_score"]
    assert overridden["baseline_score"] == pytest.approx(-np.mean((Y[EVAL] - dummy) ** 2))


def test_baseline_survives_a_failing_first_k():
    ranker, report = ranked({"lr": LinearRegression(), "boom": FailsOnSubset()})

    block = ablate(ranker, report, ks=[1, 3])["ablation"]

    assert block["baseline_score"] == pytest.approx(report["baseline_score"])


def test_training_weights_must_match_the_report():
    ranker, report = ranked(sample_weight=W[TRAIN])

    with pytest.raises(ValueError, match="sample_weight"):
        ablate(ranker, report)
    with pytest.raises(ValueError, match="sample_weight"):
        ablate(ranker, report, sample_weight=W[TRAIN] * 2)
    assert ablate(ranker, report, ks=[1], sample_weight=W[TRAIN])["ablation"]["curve"]


def test_training_weights_given_to_an_unweighted_report_raise():
    ranker, report = ranked()

    with pytest.raises(ValueError, match="sample_weight"):
        ablate(ranker, report, sample_weight=W[TRAIN])


def test_eval_weights_are_checked_when_the_eval_data_is_the_reports():
    ranker, _ = ranked(sample_weight=W[TRAIN])
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, sample_weight=W[EVAL]
    )
    options = {"ks": [1], "sample_weight": W[TRAIN]}

    with pytest.raises(ValueError, match="eval_sample_weight"):
        ablate(ranker, report, **options)
    with pytest.raises(ValueError, match="eval_sample_weight"):
        ablate(ranker, report, eval_sample_weight=W[EVAL] * 2, **options)
    assert ablate(ranker, report, eval_sample_weight=W[EVAL], **options)["ablation"]["curve"]


def test_eval_weights_are_free_for_another_eval_set():
    ranker, report = ranked()
    other = X[EVAL] + 1.0

    block = ranker.ablate(
        X[TRAIN], Y[TRAIN], other, Y[EVAL], report, ks=[1], eval_sample_weight=W[EVAL]
    )["ablation"]

    assert block["curve"][0]["median_score"] is not None


def test_a_report_without_a_run_block_is_not_checked():
    ranker, report = ranked()
    bare = Report({k: v for k, v in report.items() if k != "run"})

    block = ablate(ranker, bare, ks=[1], sample_weight=W[TRAIN])["ablation"]

    assert len(block["curve"]) == 1


def test_a_report_column_the_ranker_lacks_raises():
    ranker, report = ranked()
    groups = {**report["feature_groups"], "x0": ["ghost"]}

    with pytest.raises(ValueError, match="ghost"):
        ablate(ranker, Report({**report, "feature_groups": groups}), ks=[N_FEATURES])


def test_a_ranking_stage_failure_is_reported_per_model():
    ranker, report = ranked(
        {"lr": LinearRegression(), "boom": FailsOnSubset(), "pf": PredictFailsOnSubset()}
    )

    block = ablate(ranker, report, ks=[1])["ablation"]

    assert block["curve"][0]["scores"]["lr"] is not None
    assert block["curve"][0]["scores"]["boom"] is None
    assert [(f["model"], f["stage"], f["k"]) for f in block["failures"]] == [
        ("boom", "fit", 1), ("pf", "ranking", 1),
    ]


def test_every_model_failing_at_the_ranking_stage_is_reported():
    ranker, report = ranked({"boom": FailsOnSubset(), "pf": PredictFailsOnSubset()})

    block = ablate(ranker, report, ks=[1])["ablation"]

    assert block["curve"][0]["scores"] == {"boom": None, "pf": None}
    assert block["curve"][0]["median_score"] is None
    assert [(f["model"], f["stage"]) for f in block["failures"]] == [
        ("boom", "fit"), ("pf", "ranking"),
    ]


def test_a_voter_the_ranker_lacks_raises():
    _, report = ranked()
    narrow = make_ranker({"lr": LinearRegression()})
    narrow.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    built_in = FeatureRanker("reg", group="fast", verbose=False)
    built_in.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)

    with pytest.raises(ValueError, match="tree"):
        ablate(narrow, report)
    with pytest.raises(ValueError, match="lr"):
        ablate(built_in, report)


def test_all_models_losing_to_the_dummy_is_an_empty_consensus():
    ranker, report = ranked({"worse": Worse()})

    assert report["consensus"] == []
    with pytest.raises(ValueError, match="consensus"):
        ablate(ranker, report)


def test_auto_categorical_with_built_in_linear_models():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(X[:, :3], columns=NAMES[:3]).assign(
        city=rng.choice(["a", "b", "c"], size=len(X))
    )
    y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=len(X))
    ranker = FeatureRanker("reg", group="linear", categorical="auto", verbose=False)
    ranker.fit(frame[:N_TRAIN], y[:N_TRAIN])
    report = ranker.rank_features(frame[N_TRAIN:], y[N_TRAIN:], scoring="r2", n_repeats=1)
    assert report["consensus"][0]["feature_group"] == "x0"

    block = ranker.ablate(
        frame[:N_TRAIN], y[:N_TRAIN], frame[N_TRAIN:], y[N_TRAIN:], report, ks=[1, 4]
    )["ablation"]

    # PLS cannot fit one column; every other refit works.
    assert [(f["model"], f["k"]) for f in block["failures"]] == [("pls_regression", 1)]
    assert block["curve"][0]["feature_groups"] == ["x0"]
    assert block["curve"][0]["median_score"] > 0.9
    assert block["curve"][1]["n_features"] == 4
