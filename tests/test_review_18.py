"""Acceptance check for review-18 fixes (lead-written; builders must not edit)."""
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
    [
        (np.datetime64("NaT", "D"), np.datetime64("NaT", "ns")),
        (np.timedelta64("NaT", "s"), np.timedelta64("NaT", "ms")),
        (np.datetime64("NaT", "D"), np.timedelta64("NaT", "D")),
    ],
    ids=["datetime-units", "timedelta-units", "datetime-timedelta"],
)
def test_swapping_numpy_nat_of_other_units_raises(first, second):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=obj([first, second, 3]))
    with pytest.raises(ValueError, match="index"):
        _check_index(X, pd.Series([2.0, 1.0, 3.0], index=obj([second, first, 3])), "y")
    nested = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=obj([(1, first), (1, second), (3, 3)]))
    with pytest.raises(ValueError, match="index"):
        _check_index(nested, pd.Series([2.0, 1.0, 3.0], index=obj([(1, second), (1, first), (3, 3)])), "y")


def test_swapped_nat_units_cannot_change_the_fit():
    labels = [np.datetime64("NaT", "D"), np.datetime64("NaT", "ns")] + list(range(38))
    swapped = [labels[1], labels[0]] + labels[2:]
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.random.default_rng(0).normal(size=40)}, index=obj(labels))
    y = pd.Series(2 * np.arange(40.0)[[1, 0] + list(range(2, 40))], index=obj(swapped))
    with pytest.raises(ValueError, match="index"):
        FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(X, y)


def test_same_unit_nat_built_separately_stays_silent():
    X = pd.DataFrame({"a": [1.0, 2.0]}, index=obj([(1, np.datetime64("NaT", "D")), (2, np.timedelta64("NaT", "s"))]))
    y = pd.Series([1.0, 2.0], index=obj([(1, np.datetime64("NaT", "D")), (2, np.timedelta64("NaT", "s"))]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_index(X, y, "y")
