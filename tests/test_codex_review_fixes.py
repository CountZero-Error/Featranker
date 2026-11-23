"""Acceptance check for codex-review-fixes (lead-written; builder must not edit)."""
import copy
import importlib.util
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import featranker.importance as importance
from featranker import FeatureRanker

ROOT = Path(__file__).resolve().parents[1]
SAGA = ("logistic_regression_l1", "logistic_regression_elasticnet")


# 1. saga gets dense one-hot; scores match


def small_clf(seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 20, 200)
    X = pd.DataFrame({"c": codes.astype(str), "x": rng.normal(size=200)})
    signal = (codes % 3) + X["x"].to_numpy() + rng.normal(size=200)
    return X, (signal > np.median(signal)).astype(int)


def rank_clf(seed):
    X, y = small_clf(seed)
    ranker = FeatureRanker("clf", "linear", verbose=False, categorical=["c"])
    ranker.fit(X.iloc[:120], y[:120])
    report = ranker.rank_features(X.iloc[120:], y[120:], scoring="accuracy", n_repeats=2, random_state=0)
    return ranker, report, X


@pytest.mark.parametrize("seed", [5, 7])
def test_saga_scores_match_dense(monkeypatch, seed):
    ranker, sparse_report, X = rank_clf(seed)
    for name in SAGA:
        out = ranker.trained_models[name].steps[0][1].transform(X.iloc[120:])
        assert not sp.issparse(out), name
    monkeypatch.setattr(importance, "_FORCE_DENSE_ONE_HOT", True)
    _, dense_report, _ = rank_clf(seed)
    for name in SAGA:
        assert sparse_report["models"][name]["evaluation_score"] == pytest.approx(
            dense_report["models"][name]["evaluation_score"], abs=1e-3), name


# 2. SHAP folds only FeatRanker-built pipelines

shap = pytest.importorskip("shap")
from featranker.evidence import _shap_values  # noqa: E402


def codes_data(seed=1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, 4, 300), rng.normal(size=(300, 2))]).astype(float)
    return X, X[:, 1] + (X[:, 0] == 2) + 0.1 * rng.normal(size=300)


@pytest.mark.parametrize("cols", [[0, 0], [0]])
def test_user_mimic_pipeline_not_folded(cols):
    X, y = codes_data()
    model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cols)],
                          remainder=StandardScaler()),
        Ridge(),
    ).fit(X, y)
    with pytest.raises(ValueError, match="ColumnTransformer"):
        _shap_values(model, X[:20], X[20:60], shap)


def test_user_mimic_pipeline_listed_unsupported():
    X, y = codes_data()
    model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [0])],
                          remainder=StandardScaler()),
        Ridge(),
    )
    ranker = FeatureRanker("reg", models={"m": model, "r": Ridge()}, verbose=False, evidence=True)
    ranker.fit(X[:200], y[:200], feature_names=["c", "x", "z"])
    report = ranker.rank_features(X[200:], y[200:], scoring="r2", n_repeats=2, random_state=0)
    reasons = {e["model"]: e["reason"] for e in report["evidence"]["shap"]["unsupported"]}
    assert "ColumnTransformer" in reasons.get("m", "")
    assert report["models"]["r"]["shap"]["explainer"] == "linear"


# 3. Schema types for scoring and importance

jsonschema = pytest.importorskip("jsonschema")
SCHEMA = json.loads((files("featranker") / "report.schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def plain_reports():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=150)
    names = ["a", "b", "c"]
    held = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).fit(X[:100], y[:100], feature_names=names)
    held = held.rank_features(X[100:], y[100:], scoring="r2", n_repeats=2, random_state=0)
    resampled = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).rank_resampled(
        X, y, scoring="r2", cv=3, n_repeats=2, random_state=0, feature_names=names)
    return [json.loads(json.dumps(r, allow_nan=False)) for r in (held, resampled)]


def test_scoring_must_be_string():
    for report in plain_reports():
        assert VALIDATOR.is_valid(report)
        bad = copy.deepcopy(report)
        bad["scoring"] = []
        assert not VALIDATOR.is_valid(bad), report["evaluation_mode"]


def test_model_importance_must_be_object():
    held = plain_reports()[0]
    bad = copy.deepcopy(held)
    bad["models"]["r"]["importance"] = False
    assert not VALIDATOR.is_valid(bad)


# 4. Benchmark RF arm keeps column positions


def load_run():
    spec = importlib.util.spec_from_file_location("realdata_run_codex_fix", ROOT / "benchmarks" / "realdata" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rf_arm_with_all_missing_column():
    run = load_run()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[:, 0] = np.nan
    y = X[:, 3].copy()
    kept = run._select("rf_median", X, y, "reg", 0, True, 20)
    assert 3 in kept and 0 not in kept and max(kept) <= 3


def test_deviation_recorded():
    text = (ROOT / "benchmarks" / "realdata" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert "keep_empty_features" in text
