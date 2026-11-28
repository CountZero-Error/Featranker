"""Regression tests for index checks with mixed-type labels and ablate weights."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from featranker import FeatureRanker


def mixed(n=60):
    index = [i if i % 2 else f"r{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)
    return X, pd.Series(2 * X["a"].to_numpy(), index=index)


def ranker():
    return FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)


def test_mixed_type_labels_warn_instead_of_crashing():
    X, y = mixed()
    with pytest.warns(UserWarning, match="index"):
        ranker().fit(X, y.reset_index(drop=True))


def test_mixed_type_labels_in_another_order_are_refused():
    X, y = mixed()
    with pytest.raises(ValueError, match="index"):
        ranker().fit(X, y.iloc[::-1])


def test_reordered_eval_weights_are_refused_in_ablate():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series(3 * X["a"].to_numpy() + rng.normal(scale=0.1, size=200))
    r = ranker().fit(X[:100], y[:100])
    report = r.rank_features(X[100:], y[100:], scoring="r2", random_state=0)
    weights = pd.Series(np.linspace(1, 2, 100), index=X.index[100:]).sort_values(ascending=False)
    with pytest.raises(ValueError, match="index"):
        r.ablate(X[:100], y[:100], X[100:], y[100:], report, ks=[1], eval_sample_weight=weights)
