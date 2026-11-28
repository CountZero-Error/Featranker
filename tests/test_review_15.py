"""Acceptance check for review-15 fixes (lead-written; builders must not edit)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from featranker import FeatureRanker
from featranker.importance import _check_index


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3 * X["a"].to_numpy() + rng.normal(scale=0.1, size=n), index=X.index)
    return X, y


# 1. Scorers do not share pandas axis objects

def renames_axes(estimator, X, y):
    X.columns.name = "changed"
    X.index.name = "changed"
    return 0.0


def reads_axes(estimator, X, y):
    return float(X.columns.name == "changed" or X.index.name == "changed")


def test_axis_name_change_does_not_reach_the_next_scorer():
    X, y = data()
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, model_input="dataframe", verbose=False)
    ranker.fit(X[:100], y[:100])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ranker.rank_features(X[100:], y[100:], scoring=[renames_axes, reads_axes], n_repeats=2, random_state=0)
    assert report["secondary_baseline_scores"]["reads_axes"] == 0.0
    assert report["models"]["lin"]["secondary"]["reads_axes"]["evaluation_score"] == 0.0
    assert all(v["mean"] == 0.0 for v in report["models"]["lin"]["secondary"]["reads_axes"]["importance"].values())


def test_scorer_still_gets_named_columns_and_read_only_values():
    X, y = data()
    seen = []

    def inspect(estimator, X_, y_):
        seen.append((list(X_.columns), X_.to_numpy().flags.writeable))
        return 0.0

    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, model_input="dataframe", verbose=False)
    ranker.fit(X[:100], y[:100])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.rank_features(X[100:], y[100:], scoring=inspect, n_repeats=2, random_state=0)
    assert seen and all(columns == ["a", "b"] for columns, _ in seen)


# 2. Missing labels inside nested tuples

def nested(values):
    return pd.Index([(1, (np.nan, "x")), (2, (float("nan"), "y")), (3, (3.0, "z"))], dtype=object, tupleize_cols=False)


def test_reordered_nested_tuple_index_with_nan_raises():
    left = nested(None)
    right = pd.Index(list(reversed([(1, (float("nan"), "x")), (2, (np.nan, "y")), (3, (3.0, "z"))])),
                     dtype=object, tupleize_cols=False)
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=left)
    with pytest.raises(ValueError, match="index"):
        _check_index(X, pd.Series([3.0, 2.0, 1.0], index=right), "y")


def test_nested_tuple_index_with_other_labels_warns():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=nested(None))
    other = pd.Index([(1, (np.nan, "x")), (2, (np.nan, "q")), (3, (3.0, "z"))], dtype=object, tupleize_cols=False)
    with pytest.warns(UserWarning, match="index"):
        _check_index(X, pd.Series([1.0, 2.0, 3.0], index=other), "y")
