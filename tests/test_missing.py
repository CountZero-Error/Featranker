import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker, auto_groups
from featranker.evidence import MIN_PAIR_ROWS, _spearman_matrix, correlation_evidence

NAMES = ["a", "b", "c"]
N_TRAIN = 80
RNG = np.random.default_rng(0)
X_CLEAN = RNG.normal(size=(120, 3))
Y_REG = 2.0 * X_CLEAN[:, 0] + X_CLEAN[:, 1] + 0.3 * RNG.normal(size=120)
Y_CLF = (Y_REG > np.median(Y_REG)).astype(int)
X_NAN = X_CLEAN.copy()
X_NAN[::9, 1] = np.nan
X_NAN[4::11, 2] = np.nan

FINITE_ERROR = (
    "[!] X must contain only finite values; impute or drop missing "
    "values before ranking."
)


def ranker(task="reg", group="fast", **kwargs):
    return FeatureRanker(task, group, verbose=False, **{"missing": "impute", **kwargs})


def run(r, X=X_NAN, y=Y_REG, **kwargs):
    r.fit(X[:N_TRAIN], y[:N_TRAIN], NAMES)
    kwargs = {"scoring": "r2", "n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_features(X[N_TRAIN:], y[N_TRAIN:], **kwargs)


class RecordingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.seen_ = np.array(X)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# -- policy and validation ------------------------------------------------------


def test_missing_defaults_to_error():
    assert FeatureRanker("reg", verbose=False).missing == "error"


@pytest.mark.parametrize("value", ["median", "IMPUTE", "", None, 1, True])
def test_invalid_missing_raises_at_construction(value):
    with pytest.raises(ValueError, match="missing"):
        FeatureRanker("reg", verbose=False, missing=value)


def test_default_rejects_nan_with_the_unchanged_message():
    r = FeatureRanker("reg", "fast", verbose=False)

    with pytest.raises(ValueError, match="finite values") as caught:
        r.fit(X_NAN[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)
    assert str(caught.value) == FINITE_ERROR

    r.fit(X_CLEAN[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)
    with pytest.raises(ValueError, match="finite values"):
        r.rank_features(X_NAN[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2")
    with pytest.raises(ValueError, match="finite values"):
        r.rank_resampled(X_NAN, Y_REG, scoring="r2", cv=2, feature_names=NAMES)


@pytest.mark.parametrize(("task", "y"), [("reg", Y_REG), ("clf", Y_CLF)])
def test_impute_accepts_nan_in_fit_and_rank_features(task, y):
    scoring = "r2" if task == "reg" else "accuracy"

    report = run(ranker(task), y=y, scoring=scoring)

    assert report["consensus"]
    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


def test_impute_accepts_nan_in_rank_resampled():
    r = ranker()

    report = r.rank_resampled(
        X_NAN, Y_REG, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1, random_state=0
    )

    assert report["n_splits"] == 3 and report["stability"]
    assert not report["failures"] or all(
        f["stage"] == "initialization" for f in report["failures"]
    )


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_impute_still_rejects_infinite_values(bad):
    X_bad = X_NAN.copy()
    X_bad[1, 0] = X_bad[N_TRAIN + 1, 0] = bad  # one training row, one evaluation row
    r = ranker()

    with pytest.raises(ValueError, match="infinite"):
        r.fit(X_bad[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)
    r.fit(X_NAN[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)
    with pytest.raises(ValueError, match="infinite"):
        r.rank_features(X_bad[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2")
    with pytest.raises(ValueError, match="infinite"):
        r.rank_resampled(X_bad, Y_REG, scoring="r2", cv=2, feature_names=NAMES)


def test_explicit_error_matches_the_default_report():
    models = {"ridge": Ridge()}

    default = run(FeatureRanker("reg", models=models, verbose=False), X=X_CLEAN)
    explicit = run(
        FeatureRanker("reg", models=models, verbose=False, missing="error"), X=X_CLEAN
    )

    assert "missing" not in default
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit, sort_keys=True)


def test_ranker_unpickled_without_missing_behaves_as_error():
    r = FeatureRanker("reg", "fast", verbose=False)
    r.fit(X_CLEAN[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)
    state = pickle.loads(pickle.dumps(r)).__dict__.copy()
    del state["missing"]  # what a 1.6 pickle holds
    old = FeatureRanker.__new__(FeatureRanker)

    old.__setstate__(state)

    assert old.missing == "error"
    with pytest.raises(ValueError, match="finite values"):
        old.rank_features(X_NAN[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2")
    assert "missing" not in old.rank_features(
        X_CLEAN[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2", n_repeats=1
    )


# -- imputation and the record --------------------------------------------------


@pytest.mark.parametrize("task", ["clf", "reg"])
@pytest.mark.parametrize("group", ["linear", "tree", "all", "fast"])
def test_builtin_models_start_with_a_median_imputer(task, group):
    registry = FeatureRanker(task, group, verbose=False, missing="impute").models[task]

    models = {**registry["linear"], **registry["tree"]}
    assert models
    for name, model in models.items():
        assert isinstance(model, Pipeline), name
        imputer = model.steps[0][1]
        assert isinstance(imputer, SimpleImputer), name
        assert imputer.strategy == "median" and imputer.add_indicator is False
    for model in registry["linear"].values():
        assert isinstance(model.steps[1][1], StandardScaler)
    for model in registry["tree"].values():
        assert len(model.steps) == 2


def test_error_policy_registry_has_no_imputer():
    registry = FeatureRanker("reg", verbose=False).models["reg"]

    for model in registry["tree"].values():
        assert not isinstance(model, Pipeline)
    for model in registry["linear"].values():
        assert isinstance(model.steps[0][1], StandardScaler)


def _median_scenario():
    """Training median of column a is 1.0, evaluation median is 100.0."""
    X_train = np.array(
        [[0.0, 1], [1.0, 2], [2.0, 3], [np.nan, 4], [1.0, 5], [1.0, 6], [0.5, 7], [1.5, 8]]
    )
    X_eval = np.array([[100.0, 1], [100.0, 2], [np.nan, 3], [99.0, 4], [101.0, 5]])
    return X_train, np.arange(8.0), X_eval, np.arange(5.0)


@pytest.mark.parametrize("shadows", [False, True])
def test_imputer_learns_from_training_rows_only(shadows):
    X_train, y_train, X_eval, y_eval = _median_scenario()
    r = ranker(shadows=shadows, shadow_seed=0)
    r.fit(X_train, y_train, ["a", "b"])
    assert r.trained_models

    r.rank_features(X_eval, y_eval, scoring="r2", n_repeats=1, random_state=0)

    for name, model in r.trained_models.items():
        statistics = model.steps[0][1].statistics_
        assert statistics[0] == 1.0, name
        assert statistics[1] == 4.5, name
        if shadows:  # shadow columns are permutations, so their medians match
            np.testing.assert_array_equal(statistics[2:], statistics[:2])


def test_all_missing_training_column_keeps_its_position():
    X = X_NAN.copy()
    X[:, 1] = np.nan
    r = ranker()

    report = run(r, X=X)

    for model in r.trained_models.values():
        assert model.steps[0][1].transform(X[:5]).shape[1] == 3
    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


def test_custom_models_receive_nan_unchanged():
    r = FeatureRanker(
        "reg", models={"rec": RecordingRegressor()}, verbose=False, missing="impute"
    )

    r.fit(X_NAN[:N_TRAIN], Y_REG[:N_TRAIN], NAMES)

    seen = r.trained_models["rec"].seen_
    np.testing.assert_array_equal(seen, X_NAN[:N_TRAIN])
    assert np.isnan(seen).any()


def test_shadows_work_with_nan():
    report = run(ranker(shadows=True, shadow_seed=3))

    assert report["shadows"] is True
    assert any("shadow_max" in m for m in report["models"].values())
    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


def test_train_counts_list_every_feature():
    X = np.arange(40.0).reshape(20, 2)
    X[[1, 5, 9], 0] = np.nan
    y = np.arange(20.0)
    r = FeatureRanker(
        "reg", models={"ridge": make_pipeline(SimpleImputer(), Ridge())},
        verbose=False, missing="impute", shadows=True, shadow_seed=0,
    )
    r.fit(X[:15], y[:15], ["a", "b"])

    report = r.rank_features(X[15:], y[15:], scoring="r2", n_repeats=1)

    assert report["missing"] == {
        "policy": "impute", "strategy": "median", "train_counts": {"a": 3, "b": 0}
    }


@pytest.mark.filterwarnings("ignore:No model beat the dummy baseline")
def test_train_counts_ignore_evaluation_rows():
    X = np.arange(40.0).reshape(20, 2)
    X[[16, 17], 0] = np.nan  # evaluation rows only
    r = FeatureRanker(
        "reg", models={"rec": RecordingRegressor()}, verbose=False, missing="impute"
    )
    r.fit(X[:15], np.arange(15.0), ["a", "b"])

    report = r.rank_features(X[15:], np.arange(5.0), scoring="r2", n_repeats=1)

    assert report["missing"]["train_counts"] == {"a": 0, "b": 0}


def test_resampled_split_reports_carry_the_record():
    r = FeatureRanker(
        "reg", models={"ridge": make_pipeline(SimpleImputer(), Ridge())},
        verbose=False, missing="impute",
    )

    report = r.rank_resampled(
        X_NAN, Y_REG, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1, random_state=0
    )

    assert "missing" not in report
    for entry in report["splits"]:
        record = entry["report"]["missing"]
        assert record["policy"] == "impute" and record["strategy"] == "median"
        assert set(record["train_counts"]) == set(NAMES)
        assert sum(record["train_counts"].values()) > 0
    first = report["splits"][0]
    # The first split's training rows are everything outside its evaluation rows.
    assert sum(first["report"]["missing"]["train_counts"].values()) <= int(
        np.isnan(X_NAN).sum()
    )


def test_error_policy_reports_have_no_missing_key():
    r = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False)

    single = run(r, X=X_CLEAN)
    resampled = r.rank_resampled(
        X_CLEAN, Y_REG, scoring="r2", cv=2, feature_names=NAMES, n_repeats=1
    )

    assert "missing" not in single
    assert all("missing" not in entry["report"] for entry in resampled["splits"])


def test_impute_report_is_strict_json():
    report = run(ranker(shadows=True, shadow_seed=0, evidence=True))

    json.dumps(report, allow_nan=False)
    assert report["missing"]["train_counts"] == {
        name: int(np.isnan(X_NAN[:N_TRAIN, i]).sum()) for i, name in enumerate(NAMES)
    }


# -- evidence and auto groups ---------------------------------------------------


def _reference_spearman(X):
    centered = rankdata(X, axis=0) - (X.shape[0] + 1) / 2
    norm = np.sqrt((centered**2).sum(axis=0))
    return np.clip(centered.T @ centered / np.outer(norm, norm), -1.0, 1.0)


def test_complete_data_correlations_are_unchanged():
    X = np.column_stack([X_CLEAN, X_CLEAN[:, 0] + 0.1 * RNG.normal(size=120)])

    np.testing.assert_array_equal(_spearman_matrix(X), _reference_spearman(X))


def test_spearman_is_pairwise_complete_with_nan():
    rho = _spearman_matrix(X_NAN)

    for i in range(3):
        for j in range(3):
            keep = ~(np.isnan(X_NAN[:, i]) | np.isnan(X_NAN[:, j]))
            expected = spearmanr(X_NAN[keep, i], X_NAN[keep, j]).statistic
            assert rho[i, j] == pytest.approx(expected, abs=1e-12)


def test_pair_needs_at_least_thirty_shared_rows():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(100, 2))
    base[:, 1] = base[:, 0] + 0.01 * rng.normal(size=100)  # near copies
    for shared, expect_nan in ((MIN_PAIR_ROWS - 1, True), (MIN_PAIR_ROWS, False)):
        X = base.copy()
        X[shared:, 1] = np.nan  # column 1 is present in `shared` rows only
        assert np.isnan(_spearman_matrix(X)[0, 1]) == expect_nan


def test_independent_columns_sharing_few_rows_are_not_paired_or_grouped():
    assert MIN_PAIR_ROWS == 30
    for seed in range(60):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 2))
        X[10:, 1] = np.nan  # the columns are present together in 10 rows
        X[:, 0] = np.where(np.arange(200) < 150, X[:, 0], np.nan)  # still 10 shared

        pairs = correlation_evidence(X, {"a": [0], "b": [1]})["pairs"]

        assert pairs == []
        assert auto_groups(X, feature_names=["a", "b"], threshold=0.1) == {}


def _copy_pair():
    rng = np.random.default_rng(1)
    e = rng.normal(size=200)
    f = e + 0.05 * rng.normal(size=200)
    g = rng.normal(size=200)
    e_missing = e.copy()
    e_missing[rng.choice(200, size=20, replace=False)] = np.nan
    return np.column_stack([e_missing, f, g])


def test_correlation_evidence_and_auto_groups_use_pairwise_complete_rows():
    X = _copy_pair()

    pairs = correlation_evidence(X, {"e": [0], "f": [1], "g": [2]})["pairs"]
    groups = auto_groups(X, feature_names=["e", "f", "g"])

    assert pairs[0]["groups"] == ["e", "f"] and pairs[0]["rho"] > 0.9
    assert groups == {"e+f": ["e", "f"]}


def test_ranker_evidence_lists_correlated_pair_with_nan():
    X = _copy_pair()
    y = X[:, 1] + np.random.default_rng(2).normal(size=200)
    r = FeatureRanker(
        "reg", models={"ridge": make_pipeline(SimpleImputer(), Ridge())},
        verbose=False, missing="impute", evidence=True,
    )
    r.fit(X[:150], y[:150], ["e", "f", "g"])

    report = r.rank_features(X[150:], y[150:], scoring="r2", n_repeats=1)

    pairs = report["evidence"]["correlation"]["pairs"]
    assert pairs[0]["groups"] == ["e", "f"] and pairs[0]["rho"] > 0.9


def test_auto_groups_accepts_dataframe_with_nan():
    frame = pd.DataFrame(_copy_pair(), columns=["e", "f", "g"])

    assert auto_groups(frame) == {"e+f": ["e", "f"]}


def test_auto_groups_still_rejects_infinite_values():
    X = _copy_pair()
    X[0, 2] = np.inf

    with pytest.raises(ValueError, match="finite"):
        auto_groups(X)


def _shap_ranker(models):
    pytest.importorskip("shap", exc_type=ImportError)
    return FeatureRanker(
        "reg", models=models, verbose=False, missing="impute", evidence=True
    )


def test_shap_explains_imputer_pipelines_as_linear_and_tree():
    r = _shap_ranker(
        {
            "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
            "forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=10, random_state=0),
            ),
        }
    )

    report = run(r)

    assert report["models"]["ridge"]["shap"]["explainer"] == "linear"
    assert report["models"]["forest"]["shap"]["explainer"] == "tree"
    assert report["evidence"]["shap"]["unsupported"] == []
    assert set(report["models"]["ridge"]["shap"]["groups"]) == set(NAMES)


def test_shap_builtin_pipelines_are_not_unsupported_because_of_the_imputer():
    r = ranker(evidence=True)
    pytest.importorskip("shap", exc_type=ImportError)

    report = run(r)

    reasons = [u["reason"] for u in report["evidence"]["shap"]["unsupported"]]
    assert not [reason for reason in reasons if "SimpleImputer" in reason]


def test_shap_does_not_unwrap_an_indicator_imputer():
    r = _shap_ranker({"ind": make_pipeline(SimpleImputer(add_indicator=True), Ridge())})

    report = run(r)

    unsupported = report["evidence"]["shap"]["unsupported"]
    assert [u["model"] for u in unsupported] == ["ind"]
    assert "SimpleImputer" in unsupported[0]["reason"]


# -- pass-through ---------------------------------------------------------------


def test_selector_missing_defaults_round_trips_and_clones():
    default = ConsensusSelector("reg", "r2", k=1)
    chosen = ConsensusSelector("reg", "r2", k=1, missing="impute")

    assert default.missing == "error" and chosen.missing == "impute"
    assert clone(chosen).get_params() == chosen.get_params()
    assert chosen.set_params(missing="error").missing == "error"


def test_selector_pickled_without_missing_loads_clones_and_refits():
    models = {"ridge": Ridge()}
    fitted = ConsensusSelector("reg", "r2", k=1, models=models, cv=2, n_repeats=1, random_state=0)
    fitted.fit(X_CLEAN, Y_REG)
    del fitted.__dict__["missing"]  # what a 1.6 pickle holds
    old = pickle.loads(pickle.dumps(fitted))

    assert old.get_params()["missing"] == "error"
    assert clone(old).missing == "error"
    old.fit(X_CLEAN, Y_REG)
    with pytest.raises(ValueError, match="finite values"):
        old.fit(X_NAN, Y_REG)


def test_selector_invalid_missing_raises_in_fit():
    with pytest.raises(ValueError, match="missing"):
        ConsensusSelector("reg", "r2", k=1, missing="median").fit(X_CLEAN, Y_REG)


def test_selector_passes_missing_to_its_ranker(monkeypatch):
    seen = []
    original = FeatureRanker.rank_resampled

    def spy(self, *args, **kwargs):
        seen.append(self.missing)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FeatureRanker, "rank_resampled", spy)

    for policy in ("error", "impute"):
        ConsensusSelector(
            "reg", "r2", k=1, models={"ridge": Ridge()}, cv=2, n_repeats=1,
            random_state=0, missing=policy,
        ).fit(X_CLEAN, Y_REG)

    assert seen == ["error", "impute"]


def test_selector_in_a_pipeline_cross_validates_data_with_nan():
    pipeline = make_pipeline(
        ConsensusSelector("reg", "r2", k=2, missing="impute", group="fast", cv=2,
                          n_repeats=1, random_state=0),
        SimpleImputer(),
        Ridge(),
    )

    scores = cross_val_score(pipeline, X_NAN, Y_REG, cv=3, error_score="raise")

    assert np.isfinite(scores).all()


def test_selector_transform_rejects_nan_under_the_default_policy():
    fitted = ConsensusSelector(
        "reg", "r2", k=1, models={"ridge": Ridge()}, cv=2, n_repeats=1, random_state=0
    ).fit(X_CLEAN, Y_REG)

    with pytest.raises(ValueError, match="NaN"):
        fitted.transform(X_NAN)


# -- CLI ----------------------------------------------------------------------


def _write_csv(path, rows):
    path.write_text("a,b,target\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_csv_reads_empty_cells_as_nan_only_when_allowed(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(path, ["1,,3", "4,5,6", "nan,8,9"])

    X, _, _ = cli_module._load_csv(path, "target", allow_nan=True)

    assert np.isnan(X[0, 1]) and np.isnan(X[2, 0]) and X[1, 0] == 4.0
    with pytest.raises(ValueError, match="numeric"):
        cli_module._load_csv(path, "target")


def test_load_csv_with_allow_nan_still_rejects_infinite(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(path, ["1,2,3", "inf,5,6"])

    with pytest.raises(ValueError, match="finite"):
        cli_module._load_csv(path, "target", allow_nan=True)


def _cli_files(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 2 * X[:, 0] + 0.1 * rng.normal(size=40)

    def rows(lo, hi):
        lines = []
        for index in range(lo, hi):
            a = "" if index % 7 == 0 else f"{X[index, 0]:.6f}"
            lines.append(f"{a},{X[index, 1]:.6f},{y[index]:.6f}")
        return lines

    _write_csv(tmp_path / "train.csv", rows(0, 28))
    _write_csv(tmp_path / "eval.csv", rows(28, 40))
    return [
        "featranker", "--task", "reg", "--group", "linear", "--scoring", "r2",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--n-repeats", "2", "--random-state", "0", "--quiet",
    ]


def test_cli_missing_impute_ranks_csv_with_empty_cells(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [*_cli_files(tmp_path), "--missing", "impute"])

    assert cli_module.main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["missing"]["policy"] == "impute"
    assert report["missing"]["train_counts"] == {"a": 4, "b": 0}


def test_cli_default_still_rejects_empty_cells(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", _cli_files(tmp_path))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "numeric" in capsys.readouterr().err


def test_cli_rejects_unknown_missing_choice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [*_cli_files(tmp_path), "--missing", "drop"])

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "--missing" in capsys.readouterr().err


# -- review fixes -------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_pandas_nullable_dtypes_follow_the_policy(dtype):
    frame = pd.DataFrame(np.round(X_CLEAN * 10), columns=NAMES).astype(dtype)
    frame.iloc[3, 1] = pd.NA
    frame.iloc[70, 2] = pd.NA

    with pytest.raises(ValueError, match="finite values"):
        FeatureRanker("reg", "fast", verbose=False).fit(frame[:N_TRAIN], Y_REG[:N_TRAIN])

    r = ranker()
    r.fit(frame[:N_TRAIN], Y_REG[:N_TRAIN])
    report = r.rank_features(frame[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2", n_repeats=1)
    assert report["missing"]["train_counts"] == {"a": 0, "b": 1, "c": 1}
    resampled = ranker().rank_resampled(
        frame, Y_REG, scoring="r2", cv=2, n_repeats=1, random_state=0
    )
    assert resampled["n_splits"] == 2
    assert auto_groups(frame) == {}


def test_pandas_frames_are_copied():
    frame = pd.DataFrame(X_CLEAN, columns=NAMES)
    before = frame.to_numpy().copy()
    r = FeatureRanker(
        "reg", models={"rec": RecordingRegressor()}, verbose=False, missing="impute"
    )

    r.fit(frame, Y_REG)
    frame.iloc[0, 0] = 99.0

    np.testing.assert_array_equal(r.trained_models["rec"].seen_, before)


def test_resampled_imputer_medians_and_counts_are_per_split(monkeypatch):
    fitted = []
    original = FeatureRanker.fit

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        fitted.append(self)
        return result

    monkeypatch.setattr(FeatureRanker, "fit", spy)
    splits = [
        (np.arange(0, 70), np.arange(70, 120)),
        (np.arange(50, 120), np.arange(0, 50)),
    ]

    report = ranker(shadows=True, shadow_seed=0).rank_resampled(
        X_NAN, Y_REG, scoring="r2", cv=splits, feature_names=NAMES, n_repeats=1,
        random_state=0,
    )

    assert len(fitted) == 2
    for entry, (train, _), split_ranker in zip(report["splits"], splits, fitted):
        expected = np.nanmedian(X_NAN[train], axis=0)
        for model in split_ranker.trained_models.values():
            statistics = model.steps[0][1].statistics_
            np.testing.assert_allclose(statistics[:3], expected)
            np.testing.assert_allclose(statistics[3:], expected)
        assert entry["report"]["missing"]["train_counts"] == {
            name: int(np.isnan(X_NAN[train, i]).sum()) for i, name in enumerate(NAMES)
        }


def test_cli_default_error_hints_missing_impute(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(path, ["1,,3", "4,5,6"])
    with pytest.raises(ValueError, match=r"--missing impute"):
        cli_module._load_csv(path, "target")
    _write_csv(path, ["1,nan,3", "4,5,6"])
    with pytest.raises(ValueError, match=r"--missing impute"):
        cli_module._load_csv(path, "target")


def test_cli_only_empty_and_nan_cells_are_missing(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(path, ["NA,2,3", "4,5,6"])

    with pytest.raises(ValueError, match="numeric"):
        cli_module._load_csv(path, "target", allow_nan=True)


# -- second review round --------------------------------------------------------


def _temporal_columns():
    days = pd.date_range("2020-01-01", periods=N_TRAIN + 40)
    return {
        "datetime": days,
        "tz-aware": days.tz_localize("UTC"),
        "timedelta": days - days[0],
    }


@pytest.mark.parametrize("kind", ["datetime", "tz-aware", "timedelta"])
@pytest.mark.parametrize("policy", ["error", "impute"])
def test_temporal_columns_are_rejected_under_both_policies(kind, policy):
    frame = pd.DataFrame(X_CLEAN, columns=NAMES)
    frame["when"] = _temporal_columns()[kind]
    r = FeatureRanker("reg", "fast", verbose=False, missing=policy)

    with pytest.raises(ValueError, match="numeric"):
        r.fit(frame[:N_TRAIN], Y_REG[:N_TRAIN])
    with pytest.raises(ValueError, match="numeric"):
        auto_groups(frame)
    only = frame[["when"]]
    with pytest.raises(ValueError, match="numeric"):
        r.fit(only[:N_TRAIN], Y_REG[:N_TRAIN])


def test_spearman_complete_pairs_ignore_nan_elsewhere_in_small_data():
    rng = np.random.default_rng(3)
    a = rng.normal(size=25)
    X = np.column_stack([a, a + 0.01 * rng.normal(size=25), rng.normal(size=25)])
    X[4, 2] = np.nan  # one NaN in an unrelated column

    rho = _spearman_matrix(X)

    np.testing.assert_allclose(rho[:2, :2], _spearman_matrix(X[:, :2]), rtol=0, atol=1e-15)
    assert rho[0, 1] > 0.9
    assert np.isnan(rho[0, 2]) and np.isnan(rho[2, 2])  # 24 shared rows < 30
    pairs = correlation_evidence(X, {"a": [0], "b": [1], "c": [2]})["pairs"]
    assert [pair["groups"] for pair in pairs] == [["a", "b"]]
    assert auto_groups(X, feature_names=["a", "b", "c"]) == {"a+b": ["a", "b"]}
