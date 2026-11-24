"""Acceptance check for codex-review-7 (lead-written; builder must not edit)."""
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import ConsensusSelector, FeatureRanker
from featranker import cli
from test_codex_review_5 import ranked, shap_installed, supported  # noqa: F401

ROOT = Path(__file__).parents[1]
RNG = np.random.default_rng(42)
X = np.c_[RNG.normal(size=400), RNG.integers(0, 3, 400)]
Y = (X[:, 0] > 0).astype(float)
WRAPS = [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)]


def run(model, task="reg", y=Y, X=X, scoring=None):
    return ranked(model, X[:300], y[:300], X[300:], y[300:], task, scoring=scoring)


@pytest.mark.parametrize("spec", [[1], [0], [False, True]])
def test_native_categorical_boosting_is_unsupported(shap_installed, spec):
    for cls, task in ((HistGradientBoostingRegressor, "reg"), (HistGradientBoostingClassifier, "clf")):
        data = X if spec != [0] else X[:, ::-1]
        model = cls(categorical_features=spec, max_iter=10, random_state=0)
        y = Y if task == "reg" else Y.astype(int)
        assert not supported(run(model, task, y, data))


def test_native_categories_from_a_dataframe_are_unsupported(shap_installed):
    pd = pytest.importorskip("pandas")
    model = HistGradientBoostingRegressor(categorical_features=["b"], max_iter=10, random_state=0)
    ranker = FeatureRanker("reg", models={"m": model}, evidence=True, verbose=False,
                           model_input="dataframe")
    frame = pd.DataFrame(X, columns=["a", "b"])
    ranker.fit(frame[:300], Y[:300])
    assert not supported(ranker.rank_features(frame[300:], Y[300:], scoring="r2", n_repeats=2,
                                              random_state=0))


def test_plain_histogram_boosting_stays_supported(shap_installed):
    for kwargs in ({}, {"categorical_features": None}, {"categorical_features": []}):
        model = HistGradientBoostingRegressor(max_iter=10, random_state=0, **kwargs)
        try:
            model.fit(X, Y)
        except ValueError:
            continue  # this scikit-learn refuses the option
        for wrap in WRAPS:
            assert supported(run(wrap(model)))


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("missing", [np.nan, None])
def test_gblinear_explains_nan(shap_installed, wrap, missing):
    xgboost = pytest.importorskip("xgboost")
    data = np.random.default_rng(42).normal(size=(160, 2)) + 3
    data[::4, 1] = np.nan
    y = 2 * data[:, 0] + np.nan_to_num(data[:, 1])
    model = wrap(xgboost.XGBRegressor(booster="gblinear", n_estimators=20, n_jobs=1, missing=missing))
    ranker = FeatureRanker("reg", models={"m": model}, evidence=True, verbose=False, missing="impute")
    ranker.fit(data[:120], y[:120], feature_names=["a", "b"])
    report = ranker.rank_features(data[120:], y[120:], scoring="r2", n_repeats=2, random_state=0)
    json.dumps(report, allow_nan=False)
    assert supported(report)
    fitted = ranker.trained_models["m"]
    inner = fitted[-1] if hasattr(fitted, "steps") else fitted
    seen = fitted[:-1].transform(data) if hasattr(fitted, "steps") else data
    zeroed = np.nan_to_num(seen)
    expected = np.abs((zeroed[120:] - zeroed[:120].mean(axis=0)) * np.ravel(inner.coef_)).mean(axis=0)
    got = [report["models"]["m"]["shap"]["groups"][n]["mean_abs"] for n in "ab"]
    assert got == pytest.approx(expected, rel=1e-4)
    assert inner.predict(seen) == pytest.approx(inner.predict(zeroed))


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("model_input", ["array", "dataframe"])
@pytest.mark.parametrize("wrap", WRAPS + [
    lambda m: make_pipeline(StandardScaler().set_output(transform="pandas"), m)])
def test_gblinear_on_dataframes(shap_installed, wrap, model_input, with_nan):
    xgboost, pd = pytest.importorskip("xgboost"), pytest.importorskip("pandas")
    data = np.random.default_rng(0).normal(size=(160, 2))
    if with_nan:
        data[::4, 1] = np.nan
    y = 2 * data[:, 0] + np.nan_to_num(data[:, 1])
    frame = pd.DataFrame(data, columns=["a", "b"])
    ranker = FeatureRanker("reg", models={"m": wrap(xgboost.XGBRegressor(booster="gblinear"))},
                           evidence=True, verbose=False, missing="impute", model_input=model_input)
    ranker.fit(frame[:120], y[:120])
    assert supported(ranker.rank_features(frame[120:], y[120:], scoring="r2", n_repeats=2,
                                          random_state=0))


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("policy,expected", [
    (None, True), ("SymmetricTree", True), ("Lossguide", False), ("Depthwise", False)])
def test_catboost_grow_policy(shap_installed, wrap, policy, expected):
    catboost = pytest.importorskip("catboost")
    kwargs = {"iterations": 20, "random_seed": 0, "verbose": 0, "allow_writing_files": False,
              **({"grow_policy": policy} if policy else {})}
    for cls, task, y in ((catboost.CatBoostRegressor, "reg", Y),
                         (catboost.CatBoostClassifier, "clf", Y.astype(int))):
        assert supported(run(wrap(cls(**kwargs)), task, y)) is expected


# seeds


BAD_TYPE = [np.random.default_rng(0), np.random.RandomState(0), np.random.SeedSequence(1), 1.5, 1.0,
            True, np.bool_(True), "x", [1, 2], (1,), np.array([1]), np.float64(1)]
BAD_VALUE = [-1, np.int64(-5), 2 ** 64, 1 << 15000]
GOOD = [None, 0, 7, np.int64(7), np.uint8(7), np.uint64(2 ** 64 - 1), 2 ** 32, 2 ** 64 - 1]


def ranker():
    return FeatureRanker("reg", models={"m": Ridge()}, verbose=False)


def seed_calls(seed):
    fitted = ranker().fit(X[:300], Y[:300], feature_names=["a", "b"])
    return {
        "fit": lambda: ranker().fit(X, Y, feature_names=["a", "b"], random_state=seed),
        "rank_features": lambda: fitted.rank_features(
            X[300:], Y[300:], scoring="r2", n_repeats=2, random_state=seed),
        "rank_resampled": lambda: ranker().rank_resampled(
            X, Y, scoring="r2", n_repeats=2, cv=2, feature_names=["a", "b"], random_state=seed),
    }


@pytest.mark.parametrize("call", ["fit", "rank_features", "rank_resampled"])
@pytest.mark.parametrize("error,seed", [(TypeError, s) for s in BAD_TYPE] + [(ValueError, s) for s in BAD_VALUE],
                         ids=lambda v: type(v).__name__)
def test_bad_random_state_is_refused(call, error, seed):
    with pytest.raises(error, match="random_state"):
        seed_calls(seed)[call]()


@pytest.mark.parametrize("seed", BAD_TYPE, ids=lambda v: type(v).__name__)
def test_bad_shadow_seed_type(seed):
    with pytest.raises(TypeError, match="shadow_seed"):
        FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadow_seed=seed)


@pytest.mark.parametrize("seed", BAD_VALUE, ids=lambda v: str(v)[:8] if v < 10 ** 100 else "huge")
def test_bad_shadow_seed_value(seed):
    with pytest.raises(ValueError, match="shadow_seed"):
        FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadow_seed=seed)


@pytest.mark.parametrize("seed", GOOD, ids=repr)
def test_good_seeds(seed):
    ranker().fit(X, Y, feature_names=["a", "b"], random_state=seed)
    report = seed_calls(seed)["rank_features"]()
    json.dumps(report, allow_nan=False)
    assert report["random_state"] is None or type(report["random_state"]) is int
    assert report["random_state"] == seed
    FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadow_seed=seed)
    resampled = seed_calls(seed)["rank_resampled"]() if seed is None or int(seed) < 2 ** 32 - 10 else None
    if resampled is not None:
        json.dumps(resampled, allow_nan=False)


def test_bad_seed_is_refused_before_scoring_and_by_the_selector():
    calls = []

    def scorer(model, X, y):
        calls.append(1)
        return 0.0

    fitted = ranker().fit(X[:300], Y[:300], feature_names=["a", "b"])
    with pytest.raises(TypeError, match="random_state"):
        fitted.rank_features(X[300:], Y[300:], scoring=scorer, random_state=np.random.default_rng(0))
    assert not calls
    with pytest.raises(TypeError, match="random_state"):
        ConsensusSelector("reg", "r2", group="fast", k=1,
                          random_state=np.random.RandomState(0)).fit(X, Y)


# ablate(), CLI, benchmark, spike


def test_old_report_warning_comes_before_the_model_check():
    first = ranker().fit(X[:300], Y[:300], feature_names=["a", "b"])
    report = json.loads(json.dumps(
        first.rank_features(X[300:], Y[300:], scoring="r2", n_repeats=2, random_state=0)))
    del report["schema_version"]
    report["models"]["m"]["params"]["alpha"] = 2.0
    with pytest.warns(UserWarning, match="report.schema.json") as caught:
        with pytest.raises(ValueError, match="'m'"):
            first.ablate(X[:300], Y[:300], X[300:], Y[300:], report)
    assert [w for w in caught if "report.schema.json" in str(w.message)][0].filename == __file__
    report["schema_version"] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="'m'"):
            first.ablate(X[:300], Y[:300], X[300:], Y[300:], report)


@pytest.fixture
def csv(tmp_path, monkeypatch):
    rows = np.c_[X[:60], Y[:60]]
    np.savetxt(tmp_path / "d.csv", rows, delimiter=",", header="a,b,y", comments="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", sys.argv)
    return ["--task", "reg", "--group", "linear", "--train-data", "d.csv", "--eval-data", "d.csv",
            "--target", "y", "--scoring", "r2", "--n-repeats", "2", "--quiet"]


def run_cli(arguments):
    try:
        sys.argv = ["featranker", *arguments]  # restored by the csv fixture
        return cli.main() or 0
    except SystemExit as error:
        return error.code or 0


@pytest.mark.parametrize("output,code", [("-", 2), ("./-", 0), ("sub/-", 0)])
def test_cli_output_dash(csv, tmp_path, capsys, output, code):
    (tmp_path / "sub").mkdir()
    assert run_cli(csv + ["--output", output]) == code
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("-*"))
    assert written == {"-": [], "./-": ["-.json"], "sub/-": ["sub/-.json"]}[output]
    if code:
        assert "not supported" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--max-train-samples", "40"]])
@pytest.mark.parametrize("seed", ["-1", str(2 ** 64)])
def test_cli_refuses_a_bad_seed_before_fitting(csv, capsys, monkeypatch, seed, extra):
    def fail(*args, **kwargs):
        raise AssertionError("fit was called")

    monkeypatch.setattr(FeatureRanker, "fit", fail)
    assert run_cli(csv + ["--random-state", seed, "--output", "o.json"] + extra) == 2
    assert "random_state" in capsys.readouterr().err.replace("random-state", "random_state")


def load_run():
    path = ROOT / "benchmarks" / "realdata" / "run.py"
    spec = importlib.util.spec_from_file_location("realdata_run_7", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["realdata_run_7"] = module
    spec.loader.exec_module(module)
    return module


def test_failing_split_is_recorded_and_commit_defaults_to_null(tmp_path, monkeypatch):
    run_module = load_run()
    data = np.random.default_rng(0).normal(size=(569, 30))
    y = np.r_[np.zeros(568, dtype=int), 1]
    monkeypatch.setattr(run_module, "load", lambda *a, **k: (data, y, [f"f{i}" for i in range(30)]))
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", "1510", "--seeds", "0,1", "--out", str(out)])
    run_module.main()
    result = json.loads(out.read_text())
    assert len(result["rows"]) == 2 * len(run_module.ARMS) * len(run_module.MODELS)
    assert all(row["error"] and row["dataset"] for row in result["rows"])
    assert {row["seed"] for row in result["rows"]} == {0, 1}
    assert '"commit": null' in out.read_text() and "482b62d" not in out.read_text()


def test_frozen_spike_digest():
    folder = ROOT / "spikes" / "investigator"
    digest = hashlib.sha256((folder / "spike.py").read_text(encoding="utf-8").encode()).hexdigest()  # LF
    findings = (folder / "FINDINGS.md").read_text(encoding="utf-8")
    assert digest == "b90a37cd4c12e7a1a6485c5056475fd3129ccf9903caf811d48e47c1d0cc54e4"
    assert digest in findings and "951088f" in findings and "featranker<2.0" in findings
