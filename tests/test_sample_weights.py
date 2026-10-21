import json
import sys

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker
from featranker.importance import _resolve_splits

N = 120
N_TRAIN = 80
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, 3))
Y = 2 * X[:, 0] + X[:, 1] + 0.1 * RNG.normal(size=N)
NAMES = ["a", "b", "c"]
W = RNG.uniform(0.5, 2.0, size=N)
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, N)
FITS: list = []


class Recorder(RegressorMixin, BaseEstimator):
    """Keeps the sample_weight of its fit(); also logs it in FITS."""

    def fit(self, X, y, sample_weight=None):
        self.sample_weight_ = None if sample_weight is None else np.array(sample_weight)
        FITS.append(self.sample_weight_)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class NoWeights(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def ranker(models, **kwargs):
    return FeatureRanker("reg", models=models, verbose=False, **kwargs)


def fit_rank(r, train_weight=None, eval_weight=None, **kwargs):
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=train_weight)
    kwargs = {"n_repeats": 2, "random_state": 0, "scoring": "r2", **kwargs}
    return r.rank_features(
        X[EVAL], Y[EVAL], sample_weight=eval_weight, **kwargs
    )


# -- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "weights",
    [
        np.ones(N_TRAIN - 1),
        np.ones((N_TRAIN, 1)),
        np.r_[-1.0, np.ones(N_TRAIN - 1)],
        np.r_[np.nan, np.ones(N_TRAIN - 1)],
        np.r_[np.inf, np.ones(N_TRAIN - 1)],
        np.zeros(N_TRAIN),
        ["x"] * N_TRAIN,
        np.full(N_TRAIN, 1e308),
    ],
)
def test_invalid_weights_raise_before_any_fit(weights):
    FITS.clear()
    r = ranker({"rec": Recorder()})
    with pytest.raises(ValueError, match="sample_weight"):
        r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=weights)
    assert FITS == []
    assert not r.is_fitted_


def test_invalid_eval_and_resampled_weights_raise():
    r = ranker({"rec": Recorder()})
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    with pytest.raises(ValueError, match="sample_weight"):
        r.rank_features(X[EVAL], Y[EVAL], scoring="r2", sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="sample_weight"):
        r.rank_resampled(X, Y, scoring="r2", feature_names=NAMES, sample_weight=np.ones(3))


def test_weights_are_copied():
    weights = W[TRAIN].copy()
    r = ranker({"rec": Recorder()}, evidence=True)
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=weights)
    weights[:] = 99.0
    assert np.array_equal(r.train_weight_, W[TRAIN])
    assert not np.shares_memory(r.train_weight_, weights)


def test_unweighted_report_has_no_sample_weight_key_and_is_deterministic():
    first = fit_rank(ranker({"ridge": Ridge()}, evidence=True))
    second = fit_rank(ranker({"ridge": Ridge()}, evidence=True), None, None)
    assert "sample_weight" not in first
    first.pop("provenance"), second.pop("provenance")
    assert json.dumps(first) == json.dumps(second)


# -- fitting ---------------------------------------------------------------


def test_bare_estimator_and_pipeline_final_step_receive_weights():
    r = ranker({
        "bare": Recorder(),
        "piped": make_pipeline(StandardScaler(), Recorder()),
    })
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    assert np.array_equal(r.trained_models["bare"].sample_weight_, W[TRAIN])
    assert np.array_equal(r.trained_models["piped"].steps[-1][1].sample_weight_, W[TRAIN])


def test_model_without_sample_weight_is_a_fit_failure_not_unweighted_fit():
    r = ranker({"ok": Recorder(), "plain": NoWeights(), "piped": make_pipeline(NoWeights())})
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    assert sorted(r.trained_models) == ["ok"]
    assert {f["model"]: f["error_type"] for f in r.fit_failures_} == {
        "plain": "SampleWeightNotSupported",
        "piped": "SampleWeightNotSupported",
    }
    assert all(f["stage"] == "fit" for f in r.fit_failures_)


def test_unweighted_fit_still_accepts_models_without_sample_weight():
    r = ranker({"plain": NoWeights()})
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    assert r.fit_failures_ == []


def test_builtin_classification_lda_and_qda_are_unsupported():
    labels = (Y > 0).astype(int)
    r = FeatureRanker("clf", "all", verbose=False)
    r.fit(X[TRAIN], labels[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    failed = {f["model"]: f["error_type"] for f in r.fit_failures_}
    assert failed == {
        "lda": "SampleWeightNotSupported",
        "qda": "SampleWeightNotSupported",
    }
    assert "lda" not in r.trained_models and r.trained_models


def test_builtin_regression_pls_is_unsupported():
    r = FeatureRanker("reg", "linear", verbose=False)
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    assert [f["model"] for f in r.fit_failures_] == ["pls_regression"]


def test_weighted_dummy_predicts_the_weighted_majority():
    y = np.array([0, 0, 0, 1, 1])
    weights = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    features = np.arange(10.0).reshape(5, 2)
    r = FeatureRanker("clf", "tree", verbose=False)
    r.fit(features, y, feature_names=["p", "q"], sample_weight=weights)
    assert r.baseline_model_.predict(features).tolist() == [1] * 5
    unweighted = FeatureRanker("clf", "tree", verbose=False)
    unweighted.fit(features, y, feature_names=["p", "q"])
    assert unweighted.baseline_model_.predict(features).tolist() == [0] * 5


# -- scoring and record ----------------------------------------------------


def test_scores_are_weighted_like_manual_weighted_scores():
    r = ranker({"ridge": Ridge()}, evidence=True)
    report = fit_rank(r, W[TRAIN], W[EVAL], n_repeats=1)
    model = r.trained_models["ridge"]
    assert np.allclose(
        model.coef_, Ridge().fit(X[TRAIN], Y[TRAIN], sample_weight=W[TRAIN]).coef_
    )
    entry = report["models"]["ridge"]
    baseline = r2_score(Y[EVAL], model.predict(X[EVAL]), sample_weight=W[EVAL])
    assert entry["evaluation_score"] == pytest.approx(baseline)
    assert entry["train_score"] == pytest.approx(
        r2_score(Y[TRAIN], model.predict(X[TRAIN]), sample_weight=W[TRAIN])
    )
    dummy_mean = np.average(Y[TRAIN], weights=W[TRAIN])
    assert report["baseline_score"] == pytest.approx(
        r2_score(Y[EVAL], np.full(N - N_TRAIN, dummy_mean), sample_weight=W[EVAL])
    )
    order = np.random.default_rng(0).permutation(N - N_TRAIN)
    permuted = X[EVAL].copy()
    permuted[:, 0] = X[EVAL][order, 0]
    expected = baseline - r2_score(Y[EVAL], model.predict(permuted), sample_weight=W[EVAL])
    assert entry["importance"]["a"]["values"][0] == pytest.approx(expected)


def test_weights_change_the_ranking():
    rng = np.random.default_rng(1)
    n = 600
    heavy = np.arange(n) % 2 == 0
    a, b = rng.normal(size=n), rng.normal(size=n)
    features = np.column_stack([a, b, heavy.astype(float)])
    target = np.where(heavy, a, b)
    weights = np.where(heavy, 1.0, 0.001)
    model = {"forest": RandomForestRegressor(n_estimators=40, random_state=0)}

    def importance(w):
        r = ranker(model)
        r.fit(features[:400], target[:400], feature_names=["a", "b", "t"],
              sample_weight=None if w is None else w[:400])
        report = r.rank_features(
            features[400:], target[400:], scoring="r2", n_repeats=5, random_state=0,
            sample_weight=None if w is None else w[400:],
        )
        scores = report["models"]["forest"]["importance"]
        return scores["a"]["mean"], scores["b"]["mean"]

    weighted_a, weighted_b = importance(weights)
    plain_a, plain_b = importance(None)
    assert weighted_a > 5 * weighted_b
    assert 0.5 < plain_a / plain_b < 2.0


@pytest.mark.parametrize(
    ("train_weight", "eval_weight", "record"),
    [
        (W[TRAIN], None, {"train": True, "eval": False}),
        (None, W[EVAL], {"train": False, "eval": True}),
        (W[TRAIN], W[EVAL], {"train": True, "eval": True}),
    ],
)
def test_report_records_which_weights_were_used(train_weight, eval_weight, record):
    report = fit_rank(ranker({"rec": Recorder()}), train_weight, eval_weight)
    assert report["sample_weight"] == record
    json.dumps(report, allow_nan=False)


def test_refit_without_weights_forgets_training_weights():
    r = ranker({"rec": Recorder()})
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    report = r.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=1)
    assert "sample_weight" not in report


def test_evidence_correlation_stays_unweighted():
    weighted = fit_rank(ranker({"ridge": Ridge()}, evidence=True), W[TRAIN], W[EVAL])
    plain = fit_rank(ranker({"ridge": Ridge()}, evidence=True))
    assert weighted["evidence"]["correlation"] == plain["evidence"]["correlation"]
    assert weighted["models"]["ridge"]["train_score"] != plain["models"]["ridge"]["train_score"]


def test_weights_work_with_impute_categorical_dataframe_and_shadows():
    frame = pd.DataFrame(X.copy(), columns=NAMES)
    frame["city"] = np.where(X[:, 2] > 0, "rome", "oslo")
    frame.loc[::7, "a"] = np.nan
    r = FeatureRanker(
        "reg", "linear", verbose=False, missing="impute", categorical="auto",
        model_input="dataframe", shadows=True, shadow_seed=0, evidence=True,
    )
    r.fit(frame.iloc[:N_TRAIN], Y[:N_TRAIN], sample_weight=W[:N_TRAIN])
    report = r.rank_features(
        frame.iloc[N_TRAIN:], Y[N_TRAIN:], scoring="r2", n_repeats=2, random_state=0,
        sample_weight=W[N_TRAIN:],
    )
    assert report["sample_weight"] == {"train": True, "eval": True}
    assert [(f["model"], f["error_type"]) for f in report["failures"]] == [
        ("pls_regression", "SampleWeightNotSupported")
    ]
    assert all(entry["train_score"] is not None for entry in report["models"].values())


# -- resampling ------------------------------------------------------------


def test_rank_resampled_slices_weights_per_split():
    FITS.clear()
    weights = np.arange(1.0, N + 1)
    folds = list(KFold(3).split(X))
    report = ranker({"rec": Recorder()}).rank_resampled(
        X, Y, scoring="r2", cv=folds, feature_names=NAMES, n_repeats=1,
        sample_weight=weights,
    )
    assert len(FITS) == 3
    for fitted, (train, _) in zip(FITS, folds):
        assert np.array_equal(fitted, weights[train])
    assert all(
        split["report"]["sample_weight"] == {"train": True, "eval": True}
        for split in report["splits"]
    )


def test_rank_resampled_without_weights_has_no_record():
    report = ranker({"rec": Recorder()}).rank_resampled(
        X, Y, scoring="r2", cv=2, feature_names=NAMES, n_repeats=1
    )
    assert all("sample_weight" not in split["report"] for split in report["splits"])


# -- weight-support detection ----------------------------------------------


def test_nested_pipeline_and_var_keyword_fit_are_weight_capable():
    from sklearn.compose import TransformedTargetRegressor

    r = ranker({
        "nested": make_pipeline(StandardScaler(), make_pipeline(Recorder())),
        "ttr": TransformedTargetRegressor(regressor=Ridge()),
    })
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    assert sorted(r.trained_models) == ["nested", "ttr"]
    assert r.fit_failures_ == []
    inner = r.trained_models["nested"].steps[-1][1].steps[-1][1]
    assert np.array_equal(inner.sample_weight_, W[TRAIN])


def test_var_keyword_fit_error_is_an_ordinary_fit_failure():
    class Kwargs(RegressorMixin, BaseEstimator):
        def fit(self, X, y, **fit_params):
            raise TypeError("unexpected keyword")

    r = ranker({"ok": Recorder(), "kw": Kwargs()})
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=W[TRAIN])
    assert [f["error_type"] for f in r.fit_failures_] == ["TypeError"]


# -- metadata routing --------------------------------------------------------

needs_routing = pytest.mark.skipif(
    tuple(map(int, sklearn.__version__.split(".")[:2])) < (1, 4),
    reason="metadata routing needs scikit-learn 1.4",
)


def unexpected_failures(failures):
    # PLS has no sample_weight, so it is the one expected failure under weights.
    return [f for f in failures if f["error_type"] != "SampleWeightNotSupported"]


@needs_routing
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_weighted_builtin_pipelines_work_under_metadata_routing(n_jobs):
    def run():
        r = FeatureRanker("reg", "linear", verbose=False, n_jobs=n_jobs)
        report = fit_rank(r, W[TRAIN], W[EVAL], n_repeats=2)
        assert unexpected_failures(r.fit_failures_ + report["failures"]) == []
        report.pop("provenance")
        return json.dumps(report)

    plain = run()
    with sklearn.config_context(enable_metadata_routing=True):
        assert run() == plain


@needs_routing
def test_selector_in_a_pipeline_with_builtin_models_has_no_failures_under_routing():
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            ("select", ConsensusSelector(
                "reg", "r2", k=2, cv=2, n_repeats=1, group="linear", random_state=0,
            ).set_fit_request(sample_weight=True)),
            ("model", Ridge().set_fit_request(sample_weight=True)),
        ])
        pipe.fit(X, Y, sample_weight=W)
    assert unexpected_failures(pipe.named_steps["select"].report_["failures"]) == []


# -- selector --------------------------------------------------------------


def selector(**kwargs):
    return ConsensusSelector(
        "reg", "r2", k=2, cv=2, n_repeats=1, models={"rec": Recorder()}, random_state=0,
        **kwargs,
    )


def test_selector_fit_passes_weights_to_its_ranking():
    FITS.clear()
    fitted = selector().fit(X, Y, sample_weight=W)
    folds = _resolve_splits(2, X, Y, None, "reg", 0)
    assert len(FITS) == 2
    for recorded, (train, _) in zip(FITS, folds):
        assert np.array_equal(recorded, W[train])
    assert fitted.report_["splits"][0]["report"]["sample_weight"] == {
        "train": True, "eval": True,
    }


def test_selector_params_do_not_include_sample_weight():
    sel = selector()
    assert "sample_weight" not in sel.get_params()
    assert clone(sel).get_params().keys() == sel.get_params().keys()


@pytest.mark.skipif(
    tuple(map(int, sklearn.__version__.split(".")[:2])) < (1, 4),
    reason="metadata routing needs scikit-learn 1.4",
)
def test_selector_in_a_routed_pipeline_receives_weights():
    FITS.clear()
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([
            ("select", selector().set_fit_request(sample_weight=True)),
            ("model", Ridge().set_fit_request(sample_weight=True)),
        ])
        pipe.fit(X, Y, sample_weight=W)
    assert len(FITS) == 2
    assert pipe.named_steps["select"].report_["splits"][0]["report"]["sample_weight"]


# -- CLI -------------------------------------------------------------------


def write_weighted_csv(path, n_rows, rng):
    features = rng.normal(size=(n_rows, 2))
    target = 3.0 * features[:, 0] + 0.1 * rng.normal(size=n_rows)
    lines = ["signal,w,noise,target"]
    for row, y in zip(features, target):
        lines.append(f"{row[0]},{rng.uniform(0.5, 2)},{row[1]},{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli_argv(tmp_path, *extra):
    return [
        "featranker", "--task", "reg", "--group", "linear",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--scoring", "r2", "--n-repeats", "2",
        "--random-state", "0", "--quiet", "--output", str(tmp_path / "out.json"), *extra,
    ]


def test_cli_sample_weight_column_is_weights_not_a_feature(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    write_weighted_csv(tmp_path / "train.csv", 60, rng)
    write_weighted_csv(tmp_path / "eval.csv", 30, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--sample-weight", "w"))

    assert cli_module.main() == 0

    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert sorted(report["feature_groups"]) == ["noise", "signal"]
    assert report["sample_weight"] == {"train": True, "eval": True}


def test_cli_without_sample_weight_treats_the_column_as_a_feature(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    write_weighted_csv(tmp_path / "train.csv", 60, rng)
    write_weighted_csv(tmp_path / "eval.csv", 30, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path))

    assert cli_module.main() == 0

    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "w" in report["feature_groups"] and "sample_weight" not in report


@pytest.mark.parametrize("missing_in", ["train", "eval"])
def test_cli_missing_weight_column_errors(monkeypatch, tmp_path, capsys, missing_in):
    rng = np.random.default_rng(0)
    write_weighted_csv(tmp_path / "train.csv", 60, rng)
    write_weighted_csv(tmp_path / "eval.csv", 30, rng)
    path = tmp_path / f"{missing_in}.csv"
    path.write_text(path.read_text().replace(",w,", ",v,"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--sample-weight", "w"))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "Weight column 'w'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--sample-weight", "target"], "target column"),
        (["--sample-weight", "w", "--categorical", "signal,w"], "both --sample-weight and --categorical"),
    ],
)
def test_cli_rejects_bad_weight_column_up_front(monkeypatch, tmp_path, capsys, extra, message):
    rng = np.random.default_rng(0)
    write_weighted_csv(tmp_path / "train.csv", 60, rng)
    write_weighted_csv(tmp_path / "eval.csv", 30, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, *extra))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert message in capsys.readouterr().err
