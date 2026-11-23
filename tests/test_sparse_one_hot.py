"""Acceptance check for sparse-one-hot (lead-written; builder must not edit).

Hooks the builder provides: featranker.importance._FORCE_DENSE_ONE_HOT (bool, default False)
and featranker.importance.DENSE_ONLY_ESTIMATORS (frozenset of class names).
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import featranker.importance as importance
from featranker import FeatureRanker

DENSE_ONLY = {
    "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis", "GaussianNB", "PLSRegression",
    "SGDClassifier", "Perceptron",
}
DIVERGED = -10.0  # unregularised least squares on a full one-hot block plus intercept
LEAST_SQUARES = {"linear_regression", "ransac_regression"}


def frame(levels, n=600, task="reg", seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, levels, size=n)
    X = pd.DataFrame({"x": rng.normal(size=n), "c": codes.astype(str), "z": rng.normal(size=n)})
    signal = (codes % 3).astype(float) + X["x"].to_numpy()
    y = (signal > np.median(signal)).astype(int) if task == "clf" else signal + 0.1 * rng.normal(size=n)
    return X, y


def linear_fitted(ranker, task):
    names = set(ranker.models[task]["linear"])
    return {name: model for name, model in ranker.trained_models.items() if name in names}


def final_class(model):
    return type(model.steps[-1][1]).__name__


def fit_rank(task, **kwargs):
    X, y = frame(50, task=task)
    ranker = FeatureRanker(task, "linear", verbose=False, categorical=["c"], **kwargs)
    ranker.fit(X.iloc[:400], y[:400])
    scoring = "balanced_accuracy" if task == "clf" else "r2"
    report = ranker.rank_features(X.iloc[400:], y[400:], scoring=scoring, n_repeats=2, random_state=0)
    return ranker, report, X


def test_dense_only_set_is_exact():
    assert set(importance.DENSE_ONLY_ESTIMATORS) == DENSE_ONLY


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_sparse_capable_models_get_sparse_input(task):
    ranker, _, X = fit_rank(task)
    models = linear_fitted(ranker, task)
    assert models
    seen_dense_only = False
    for name, model in models.items():
        out = model.steps[0][1].transform(X.iloc[400:])
        if final_class(model) in DENSE_ONLY or getattr(model.steps[-1][1], "solver", None) in ("sag", "saga"):
            seen_dense_only = True
            assert isinstance(out, np.ndarray) and not sp.issparse(out), name
        else:
            assert sp.issparse(out), name
    assert seen_dense_only


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_scores_match_dense_within_tolerance(task, monkeypatch):
    _, sparse_report, _ = fit_rank(task)
    monkeypatch.setattr(importance, "_FORCE_DENSE_ONE_HOT", True)
    _, dense_report, _ = fit_rank(task)

    def failed(report):
        return {f["model"] for f in report["failures"]}

    assert failed(sparse_report) == failed(dense_report)
    for name, model in dense_report["models"].items():
        s = sparse_report["models"][name]["evaluation_score"]
        d = model["evaluation_score"]
        if d < DIVERGED:  # the dense path diverged; sparse must not
            assert np.isfinite(s) and s >= d and s > DIVERGED, (name, s, d)
        elif name in LEAST_SQUARES:  # no unique solution; owner ruling 2026-11-23
            assert np.isfinite(s), (name, s, d)
        else:
            assert abs(s - d) <= 1e-3, (name, s, d)


def test_forced_dense_gives_dense_everywhere(monkeypatch):
    monkeypatch.setattr(importance, "_FORCE_DENSE_ONE_HOT", True)
    ranker, _, X = fit_rank("reg")
    for name, model in linear_fitted(ranker, "reg").items():
        assert not sp.issparse(model.steps[0][1].transform(X.iloc[400:])), name


def test_max_categories_caps_sparse_output():
    X, y = frame(1000, n=3000)
    ranker = FeatureRanker("reg", "linear", verbose=False, categorical=["c"], max_categories=20).fit(X, y)
    for name, model in linear_fitted(ranker, "reg").items():
        encoder = model.steps[0][1].named_transformers_["cat"]
        assert len(encoder.get_feature_names_out()) == 20, name
