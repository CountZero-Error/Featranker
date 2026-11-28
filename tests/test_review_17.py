"""Acceptance check for review-17 fixes (lead-written; builders must not edit)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from featranker import FeatureRanker
from featranker.importance import _check_index


def obj(labels):
    return pd.Index(labels, dtype=object, tupleize_cols=False)


@pytest.mark.parametrize(
    "first, second",
    [(np.nan, pd.NA), (np.nan, None), (pd.NA, None), (pd.NaT, np.nan)],
    ids=["nan-NA", "nan-None", "NA-None", "NaT-nan"],
)
def test_swapping_distinct_missing_labels_raises(first, second):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=obj([1, first, second, 4]))
    y = pd.Series([1.0, 3.0, 2.0, 4.0], index=obj([1, second, first, 4]))
    with pytest.raises(ValueError, match="index"):
        _check_index(X, y, "y")


def test_swapping_nested_distinct_missing_labels_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=obj([(1, (np.nan, "k")), (2, (pd.NA, "k")), (3, (3, "k"))]))
    y = pd.Series([2.0, 1.0, 3.0], index=obj([(2, (pd.NA, "k")), (1, (np.nan, "k")), (3, (3, "k"))]))
    with pytest.raises(ValueError, match="index"):
        _check_index(X, y, "y")


def test_swapped_missing_labels_cannot_change_the_fit():
    labels = [np.nan, pd.NA] + list(range(38))
    swapped = [pd.NA, np.nan] + list(range(38))
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.random.default_rng(0).normal(size=40)}, index=obj(labels))
    y = pd.Series(2 * np.arange(40.0), index=obj(labels))
    y_swapped = pd.Series(y.to_numpy()[[1, 0] + list(range(2, 40))], index=obj(swapped))
    with pytest.raises(ValueError, match="index"):
        FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(X, y_swapped)


@pytest.mark.parametrize(
    "make",
    [lambda: float("nan"), lambda: np.float64("nan"), lambda: np.nan],
    ids=["float", "float64", "np.nan"],
)
def test_equivalent_nans_in_the_same_order_stay_silent(make):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=obj([(1, (np.nan, "x")), (2, (np.nan, "y")), (3, (3.0, "z"))]))
    y = pd.Series([1.0, 2.0, 3.0], index=obj([(1, (make(), "x")), (2, (make(), "y")), (3, (3.0, "z"))]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_index(X, y, "y")
