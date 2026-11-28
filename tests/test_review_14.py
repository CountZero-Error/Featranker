"""Acceptance check for review-14 fixes (lead-written; builders must not edit)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from featranker import ConsensusSelector, FeatureRanker
from featranker.importance import _check_index


class Bad(LinearRegression):
    def predict(self, X):
        return -5 * super().predict(X)


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3 * X["a"].to_numpy() + rng.normal(scale=0.1, size=n), index=X.index)
    return X, y


# 1. The dummy sees the shadow columns too

def shadow_r2(estimator, X, y):
    if "shadow__a" not in X.columns:
        raise KeyError("shadow__a")
    return r2_score(y, estimator.predict(X))


def test_scorer_reading_a_shadow_column_keeps_the_gate():
    X, y = data()
    ranker = FeatureRanker(
        task="reg", models={"bad": Bad()}, model_input="dataframe", shadows=True, shadow_seed=0, verbose=False
    )
    ranker.fit(X[:100], y[:100])
    report = ranker.rank_features(X[100:], y[100:], scoring=shadow_r2, random_state=0)
    assert report["baseline_score"] is not None
    assert report["models"]["bad"]["in_consensus"] is False
    assert report["consensus"] == []


# 1b. The selector refuses to select without the gate (owner ruling: raise)

def fails_on_dummy(estimator, X, y):
    if isinstance(estimator, DummyRegressor):
        raise RuntimeError("dummy scorer exploded")
    return r2_score(y, estimator.predict(X))


def test_selector_raises_when_the_dummy_cannot_be_scored():
    X, y = data(seed=1)
    X = X.assign(noise=np.random.default_rng(2).normal(size=len(X)))
    selector = ConsensusSelector(
        task="reg", scoring=fails_on_dummy, k=1, models={"bad": Bad()}, cv=3, n_repeats=2, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="(?i)dummy"):
            selector.fit(X, y)


def test_rank_features_keeps_gate_off_when_the_dummy_cannot_be_scored():
    X, y = data()
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(X[:100], y[:100])
    report = ranker.rank_features(X[100:], y[100:], scoring=fails_on_dummy, random_state=0)
    assert report["baseline_score"] is None
    assert report["models"]["lin"]["in_consensus"] is True
    assert [f for f in report["failures"] if f["model"] == "dummy_baseline"]


def test_selector_with_a_working_dummy_still_selects():
    X, y = data(seed=1)
    selector = ConsensusSelector(
        task="reg", scoring="r2", k=1, models={"lin": LinearRegression()}, cv=3, n_repeats=2, random_state=0
    )
    selector.fit(X, y)
    assert selector.support_.tolist() == [True, False]


# 2. Missing labels in the index check

@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1.0, np.nan, 3.0, 4.0]),
        pd.Index(["a", None, "c", "d"], dtype=object),
        pd.MultiIndex.from_tuples([(1, "a"), (np.nan, "b"), (3, "c"), (4, "d")]),
    ],
    ids=["float-nan", "object-none", "multiindex-nan"],
)
def test_reordered_index_with_missing_labels_raises(index):
    X = pd.DataFrame({"a": range(4)}, index=index)
    with pytest.raises(ValueError, match="index"):
        _check_index(X, pd.Series(range(4), index=index)[::-1], "y")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_index(X, pd.Series(range(4), index=index), "y")


def test_other_labels_with_missing_values_still_warn():
    X = pd.DataFrame({"a": range(4)}, index=pd.Index([1.0, np.nan, 3.0, 4.0]))
    with pytest.warns(UserWarning, match="index"):
        _check_index(X, pd.Series(range(4), index=pd.Index([1.0, np.nan, 3.0, 5.0])), "y")
    with pytest.warns(UserWarning, match="index"):
        _check_index(X, pd.Series(range(4), index=pd.Index([1.0, np.nan, np.nan, 4.0])), "y")


# 3. One scorer cannot change the DataFrame the next scorer gets

def replaces_a(estimator, X, y):
    X["a"] = 1e6
    return 0.0


def mean_a(estimator, X, y):
    return float(X["a"].mean())


def test_scorer_column_replacement_does_not_reach_the_next_scorer():
    X, y = data()
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, model_input="dataframe", verbose=False)
    ranker.fit(X[:100], y[:100])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ranker.rank_features(X[100:], y[100:], scoring=[replaces_a, mean_a], n_repeats=2, random_state=0)
    expected = float(X["a"][100:].mean())
    assert report["secondary_baseline_scores"]["mean_a"] == pytest.approx(expected)
    assert report["models"]["lin"]["secondary"]["mean_a"]["evaluation_score"] == pytest.approx(expected)
