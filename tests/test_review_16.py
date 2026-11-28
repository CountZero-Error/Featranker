"""Acceptance check for review-16 fixes (lead-written; builders must not edit)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from featranker import FeatureRanker
from featranker.importance import _check_index


def nested(nan):
    return pd.Index([(1, (nan(), "x")), (2, (nan(), "y")), (3, (3.0, "z"))], dtype=object, tupleize_cols=False)


def test_aligned_nested_index_with_separate_nans_is_accepted_silently():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=nested(lambda: float("nan")))
    y = pd.Series([1.0, 2.0, 3.0], index=nested(lambda: float("nan")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_index(X, y, "y")


def test_aligned_nested_index_fits():
    rows = [(i, (float("nan") if i % 2 else float(i), "k")) for i in range(40)]
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.random.default_rng(0).normal(size=40)},
                     index=pd.Index(rows, dtype=object, tupleize_cols=False))
    y = pd.Series(2 * np.arange(40.0), index=pd.Index([(i, (float("nan") if i % 2 else float(i), "k"))
                                                        for i in range(40)], dtype=object, tupleize_cols=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(X, y)
    assert ranker.trained_models["lin"].coef_[0] == pytest.approx(2.0)


def test_reordered_nested_index_still_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=nested(lambda: float("nan")))
    y = pd.Series([3.0, 2.0, 1.0], index=nested(lambda: float("nan"))[::-1])
    with pytest.raises(ValueError, match="index"):
        _check_index(X, y, "y")


def test_scorer_values_are_read_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series(3 * X["a"].to_numpy(), index=X.index)
    flags = []

    def inspect(estimator, X_, y_):
        flags.append(X_.to_numpy().flags.writeable)
        return 0.0

    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, model_input="dataframe", verbose=False)
    ranker.fit(X[:100], y[:100])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.rank_features(X[100:], y[100:], scoring=inspect, n_repeats=2, random_state=0)
    assert flags and not any(flags)
