"""Acceptance check for codex-review-8 (lead-written; builder must not edit)."""
import copy
import importlib.util
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import sklearn
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from featranker import ConsensusSelector, FeatureRanker, selection
from test_codex_review_7 import load_run

pytest.importorskip("pandas")
ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location("make_report_1_21", FIXTURES / "make_report_1_21.py")
made = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(made)
REPORTS = json.loads((FIXTURES / "report_1_21_categorical.json").read_text(encoding="utf-8"))
SAME_SKLEARN = sklearn.__version__ == "1.5.1"
LINEAR_CASES = [c for c in made.CASES if c[1] != "tree"]


def ablated(case, report=None, ranker=None, **kwargs):
    fitted, X, y = made.fitted(*case)
    report = copy.deepcopy(REPORTS[made.key(*case)]) if report is None else report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (ranker or fitted).ablate(X[:100], y[:100], X[100:], y[100:], report, **{"ks": [3], **kwargs})


# 1.21 reports


@pytest.mark.parametrize("case", made.CASES, ids=str)
def test_report_made_by_1_21_ablates(case):
    report = REPORTS[made.key(*case)]
    with pytest.warns(UserWarning, match="report.schema.json"):
        fitted, X, y = made.fitted(*case)
        result = fitted.ablate(X[:100], y[:100], X[100:], y[100:], copy.deepcopy(report), ks=[1, 3])
    json.dumps(result, allow_nan=False)
    assert len(result["ablation"]["curve"]) == 2
    if SAME_SKLEARN and sys.platform == "darwin":
        for name, score in result["ablation"]["curve"][1]["scores"].items():
            assert score == pytest.approx(report["models"][name]["evaluation_score"], abs=1e-9), name


@pytest.mark.parametrize("case", LINEAR_CASES, ids=str)
def test_refits_are_dense_for_1_21_and_sparse_for_2(case, monkeypatch):
    seen = []
    real = FeatureRanker.fit

    def spy(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        seen.append({n: m.get_params().get("columntransformer__sparse_threshold")
                     for n, m in self.trained_models.items()})
        return out

    ranker, X, y = made.fitted(*case)
    before = {n: m.get_params()["columntransformer__sparse_threshold"]
              for n, m in ranker.trained_models.items() if "columntransformer" in m.get_params()}
    assert 1.0 in before.values()
    own = ranker.rank_features(X[100:], y[100:], scoring="r2" if case[0] == "reg" else "accuracy",
                               n_repeats=1, random_state=1)
    state = pickle.dumps(ranker.get_params() if hasattr(ranker, "get_params") else sorted(vars(ranker)))
    monkeypatch.setattr(FeatureRanker, "fit", spy)
    ablated(case, ranker=ranker)
    old, seen[:] = list(seen), []
    ablated(case, report=json.loads(json.dumps(own)), ranker=ranker)
    values = {v for fit in old[-1:] for v in fit.values() if v is not None}
    assert values == {0}
    assert 1.0 in seen[-1].values()
    after = {n: m.get_params()["columntransformer__sparse_threshold"]
             for n, m in ranker.trained_models.items() if "columntransformer" in m.get_params()}
    assert after == before
    assert state == pickle.dumps(ranker.get_params() if hasattr(ranker, "get_params") else sorted(vars(ranker)))


def test_other_differences_still_raise_for_1_21():
    case = ("reg", "fast", None)
    report = copy.deepcopy(REPORTS[made.key(*case)])
    report["models"]["ridge_regression"]["params"]["ridge__alpha"] = 123.0
    if SAME_SKLEARN:
        with pytest.raises(ValueError, match="ridge_regression"):
            ablated(case, report)
        return
    # Another scikit-learn version: a difference is a warning (existing rule).
    ranker, X, y = made.fitted(*case)
    with pytest.warns(UserWarning, match="ridge_regression"):
        ranker.ablate(X[:100], y[:100], X[100:], y[100:], report, ks=[3])


def test_2x_report_with_a_dense_threshold_still_raises():
    case = ("reg", "fast", None)
    ranker, X, y = made.fitted(*case)
    report = json.loads(json.dumps(ranker.rank_features(X[100:], y[100:], scoring="r2", n_repeats=1)))
    report["models"]["ridge_regression"]["params"]["columntransformer__sparse_threshold"] = 0
    with pytest.raises(ValueError, match="ridge_regression"):
        ablated(case, report, ranker)


def test_user_pipeline_threshold_is_still_compared():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    def pipeline(threshold):
        return make_pipeline(ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), [1])], remainder="passthrough",
            sparse_threshold=threshold), Ridge())

    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=120), rng.integers(0, 3, 120)]
    y = X[:, 0] + (X[:, 1] == 1)

    def fit(threshold):
        return FeatureRanker("reg", models={"m": pipeline(threshold)}, verbose=False).fit(
            X[:80], y[:80], feature_names=["a", "b"])

    report = json.loads(json.dumps(fit(0).rank_features(X[80:], y[80:], scoring="r2", n_repeats=1)))
    del report["schema_version"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="'m'"):
            fit(1.0).ablate(X[:80], y[:80], X[80:], y[80:], report)
        assert fit(0).ablate(X[:80], y[:80], X[80:], y[80:], report)["ablation"]["curve"]


def test_old_pickle_without_new_state_still_fits():
    ranker, X, y = made.fitted("reg", "fast", None)
    known = {"dense", "_dense"}
    state = {k: v for k, v in vars(ranker).items() if not any(w in k for w in known)}
    old = FeatureRanker.__new__(FeatureRanker)
    (old.__setstate__ if hasattr(old, "__setstate__") else old.__dict__.update)(state)
    old.fit(X[:100], y[:100])
    assert old.rank_features(X[100:], y[100:], scoring="r2", n_repeats=1)["consensus"]


# shap floor


def shap_report(**kwargs):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    ranker = FeatureRanker("reg", models={"m": Ridge(), "t": DecisionTreeRegressor(max_depth=2)},
                           evidence=True, verbose=False)
    ranker.fit(X[:80], X[:80, 0], feature_names=["a", "b"])
    return ranker.rank_features(X[80:], X[80:, 0], scoring="r2", n_repeats=1)


@pytest.mark.parametrize("version", ["0.46.0", "0.48.0", "0.44.1"])
def test_old_shap_is_not_used(monkeypatch, version):
    shap = pytest.importorskip("shap")
    monkeypatch.setattr(shap, "__version__", version)
    with pytest.warns(UserWarning, match=r"0\.49") as caught:
        report = shap_report()
    assert any(version in str(w.message) and w.filename == __file__ for w in caught)
    assert report["evidence"]["shap"] == {"available": False}
    assert not any("shap" in m for m in report["models"].values())


@pytest.mark.parametrize("version", ["0.49.0", "0.49.1", "0.54.1", "1.0.0", "unknown"])
def test_new_or_unknown_shap_is_used(monkeypatch, version):
    shap = pytest.importorskip("shap")
    monkeypatch.setattr(shap, "__version__", version)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report = shap_report()
    assert report["evidence"]["shap"]["unsupported"] == []


def test_extras_require_the_floor():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert "shap>=0.49" in extras["shap"]
    assert not any(r.strip() == "shap" for group in extras.values() for r in group)


def test_nan_tree_reproduces_the_model():
    pytest.importorskip("shap")
    train = np.r_[np.full(70, -1.0), np.full(20, 1.0), np.full(10, np.nan)][:, None]
    X = np.r_[np.full(10, -1.0), np.full(10, 1.0), np.full(80, np.nan)][:, None]
    y = lambda v: np.where(v[:, 0] < 0, 0.0, 10.0)  # noqa: E731
    ranker = FeatureRanker("reg", models={"t": DecisionTreeRegressor(max_depth=1, random_state=0)},
                           missing="impute", evidence=True, verbose=False)
    ranker.fit(train, y(train), feature_names=["a"])
    report = ranker.rank_features(X, y(X), scoring="r2", n_repeats=1)
    assert report["models"]["t"]["shap"]["groups"]["a"]["mean_abs"] == pytest.approx(6.6)


# resume, order, counts


def test_resume_keeps_the_commit(tmp_path, monkeypatch, capsys):
    run_module = load_run()
    data = np.random.default_rng(0).normal(size=(569, 30))
    y = np.r_[np.zeros(568, dtype=int), 1]  # unsplittable: error rows only, fast
    monkeypatch.setattr(run_module, "load", lambda *a, **k: (data, y, [f"f{i}" for i in range(30)]))
    out = tmp_path / "run.json"

    def main(*arguments):
        monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", "1510", "--out", str(out), *arguments])
        run_module.main()
        return json.loads(out.read_text())

    assert main("--seeds", "0", "--commit", "70e1fcf")["featranker"]["commit"] == "70e1fcf"
    result = main("--seeds", "0,1", "--resume")
    assert result["featranker"]["commit"] == "70e1fcf"
    assert {r["seed"] for r in result["rows"]} == {0, 1}
    assert main("--seeds", "0,1,2", "--resume", "--commit", "70e1fcf")["featranker"]["commit"] == "70e1fcf"
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main("--seeds", "0,1,2,3", "--resume", "--commit", "bfcba9a")
    assert out.read_bytes() == before and "commit" in capsys.readouterr().err
    out.unlink()
    assert main("--seeds", "0")["featranker"]["commit"] is None
    assert main("--seeds", "0,1", "--resume")["featranker"]["commit"] is None
    with pytest.raises(SystemExit):
        main("--seeds", "0,1,2", "--resume", "--commit", "70e1fcf")


def test_bad_seed_is_refused_before_grouping(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("work was done")

    X = np.random.default_rng(0).normal(size=(60, 3))
    monkeypatch.setattr(selection, "auto_groups", fail)
    for seed, error in ((42.0, TypeError), (-1, ValueError)):
        selector = ConsensusSelector("reg", "r2", models={"r": Ridge()}, k=1, feature_groups="auto",
                                     random_state=seed)
        with pytest.raises(error, match="random_state"):
            selector.fit(X, X[:, 0])
    from featranker import importance
    monkeypatch.setattr(importance, "_validate_X", fail)
    with pytest.raises(TypeError, match="random_state"):
        FeatureRanker("reg", models={"r": Ridge()}, verbose=False).rank_resampled(
            X, X[:, 0], scoring="r2", random_state=42.0)


@pytest.mark.parametrize("huge", [2 ** 63, 10 ** 5000], ids=["2**63", "10**5000"])
@pytest.mark.parametrize("option", ["n_repeats", "max_samples"])
def test_huge_counts_are_refused(option, huge):
    X = np.random.default_rng(0).normal(size=(60, 3))
    ranker = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).fit(
        X, X[:, 0], feature_names=list("abc"))
    with pytest.raises(ValueError, match=option):
        ranker.rank_features(X, X[:, 0], scoring="r2", **{option: huge})
    for name in (option, "top_k", "max_train_samples"):
        with pytest.raises(ValueError, match="max_samples" if "samples" in name else name):
            FeatureRanker("reg", models={"r": Ridge()}, verbose=False).rank_resampled(
                X, X[:, 0], scoring="r2", feature_names=list("abc"), **{name: huge})


def test_large_but_valid_max_samples_still_caps():
    X = np.random.default_rng(0).normal(size=(60, 3))
    ranker = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).fit(
        X, X[:, 0], feature_names=list("abc"))
    report = ranker.rank_features(X, X[:, 0], scoring="r2", n_repeats=1, max_samples=2 ** 63 - 1)
    json.dumps(report, allow_nan=False)
    assert report["n_eval_rows"] == 60


# review round 1: everything is checked before grouping or fitting


@pytest.mark.parametrize("options, error", [
    ({"shadow_seed": 42.0}, TypeError),
    ({"shadow_seed": -1}, ValueError),
    ({"n_repeats": 2 ** 63}, ValueError),
    ({"n_repeats": 0}, ValueError),
    ({"k": 2 ** 63}, ValueError),
    ({"max_samples": 2 ** 63}, ValueError),
    ({"max_train_samples": 2 ** 63}, ValueError),
])
def test_selector_checks_options_before_grouping(monkeypatch, options, error):
    def fail(*args, **kwargs):
        raise AssertionError("work was done")

    monkeypatch.setattr(selection, "auto_groups", fail)
    X = np.random.default_rng(0).normal(size=(60, 3))
    selector = ConsensusSelector("reg", "r2", models={"r": Ridge()}, feature_groups="auto",
                                 **{"k": 1, **options})
    name = next(iter(options)).replace("max_train_samples", "max_samples")
    with pytest.raises(error, match=name):
        selector.fit(X, X[:, 0])


def test_selector_still_takes_numpy_integers():
    X = np.random.default_rng(0).normal(size=(60, 3))
    selector = ConsensusSelector("reg", "r2", models={"r": Ridge()}, feature_groups="auto", k=np.int64(1),
                                 n_repeats=np.int64(2), max_samples=np.int64(10), shadow_seed=np.int64(3),
                                 max_train_samples=np.float32(0.5), random_state=np.int64(0))
    assert selector.fit(X, X[:, 0]).get_support().sum() == 1


@pytest.mark.parametrize("option, value", [
    ("--n-repeats", 2 ** 63), ("--n-repeats", 0), ("--max-samples", 2 ** 63),
    ("--max-train-samples", 2 ** 63), ("--random-state", 2 ** 64),
])
def test_cli_checks_options_before_grouping_and_fitting(tmp_path, monkeypatch, capsys, option, value):
    from featranker import cli

    def fail(*args, **kwargs):
        raise AssertionError("work was done")

    monkeypatch.setattr(cli, "auto_groups", fail)
    monkeypatch.setattr(FeatureRanker, "fit", fail)
    rng = np.random.default_rng(0)
    for name in ("train", "eval"):
        rows = np.c_[rng.normal(size=(40, 2)), rng.normal(size=40)]
        (tmp_path / f"{name}.csv").write_text(
            "a,b,y\n" + "\n".join(",".join(map(str, row)) for row in rows) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--train-data", str(tmp_path / "train.csv"), "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "y", "--task", "reg", "--group", "fast", "--scoring", "r2",
        "--feature-groups", "auto", f"{option}={value}"])
    try:
        code = cli.main()
    except SystemExit as exit_:
        code = exit_.code
    assert code
    error = capsys.readouterr().err.split("error: ")[-1]
    assert "[!]" in error and option.strip("-").replace("-", "_").replace("max_train_samples", "max_samples") in error.replace("-", "_")


def test_huge_fraction_is_a_value_error():
    from fractions import Fraction

    X = np.random.default_rng(0).normal(size=(60, 3))
    ranker = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).fit(
        X, X[:, 0], feature_names=list("abc"))
    for value in (Fraction(10 ** 5000), float("inf"), float("nan"), -0.5):
        with pytest.raises(ValueError, match="max_samples"):
            ranker.rank_features(X, X[:, 0], scoring="r2", max_samples=value)
    assert ranker.rank_features(X, X[:, 0], scoring="r2", n_repeats=1, max_samples=Fraction(1, 2))["n_eval_rows"]
