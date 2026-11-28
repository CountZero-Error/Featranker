"""Acceptance check for review-13 fixes (lead-written; builders must not edit)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from featranker import ConsensusSelector, FeatureRanker


class Bad(LinearRegression):
    """A model far worse than the dummy."""

    def predict(self, X):
        return -5 * super().predict(X)


def df_r2(estimator, X, y):
    """A scorer that needs named columns."""
    return r2_score(y, estimator.predict(X[list(X.columns)]))


def data(n=200, seed=0, index=None):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)
    y = pd.Series(3 * X["a"].to_numpy() + rng.normal(scale=0.1, size=n), index=X.index, name="y")
    return X, y


def index_warnings(record):
    return [w for w in record if issubclass(w.category, UserWarning) and "index" in str(w.message).lower()]


# 1. Dummy gate under model_input="dataframe"

def test_dataframe_scorer_keeps_the_dummy_gate():
    X, y = data()
    ranker = FeatureRanker(task="reg", models={"bad": Bad()}, model_input="dataframe", verbose=False)
    ranker.fit(X[:100], y[:100])
    report = ranker.rank_features(X[100:], y[100:], scoring=df_r2, random_state=0)
    assert report["baseline_score"] is not None
    assert report["models"]["bad"]["in_consensus"] is False
    assert report["consensus"] == []
    assert not [f for f in report["failures"] if f["model"] == "dummy_baseline"]


def test_selector_with_only_a_bad_model_keeps_nothing():
    X, y = data(seed=1)
    X = X.assign(noise=np.random.default_rng(2).normal(size=len(X)))
    selector = ConsensusSelector(
        task="reg", scoring=df_r2, k=1, models={"bad": Bad()}, model_input="dataframe", cv=3, n_repeats=2,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X, y)
    assert not selector.support_.any()


# 4. The dummy cannot change the evaluation data

def test_scorer_cannot_mutate_y_through_the_dummy():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 2))
    t = 2 * x[:, 0]

    def mutating(estimator, X, y):
        if isinstance(estimator, DummyRegressor):
            y[:] = 0
        return r2_score(y, estimator.predict(X))

    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)
    ranker.fit(x[:100], t[:100], feature_names=["a", "b"])
    report = ranker.rank_features(x[100:], t[100:], scoring=mutating, random_state=0)
    assert report["models"]["lin"]["evaluation_score"] == pytest.approx(1.0)
    assert [f for f in report["failures"] if f["model"] == "dummy_baseline"]


# 2. Pandas index alignment

def fitted(X, y):
    return FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(X[:100], y[:100])


def test_sorted_target_is_refused_in_rank_features():
    X, y = data()
    ranker = fitted(X, y)
    with pytest.raises(ValueError, match="index"):
        ranker.rank_features(X[100:], y[100:].sort_values(), scoring="r2", random_state=0)


def test_reordered_target_is_refused_in_fit():
    X, y = data()
    with pytest.raises(ValueError, match="index"):
        FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False).fit(
            X[:100], y[:100].sample(frac=1, random_state=1)
        )


def test_reordered_weights_are_refused():
    X, y = data()
    ranker = fitted(X, y)
    weights = pd.Series(np.linspace(1, 2, 100), index=X.index[100:]).sort_values(ascending=False)
    with pytest.raises(ValueError, match="index"):
        ranker.rank_features(X[100:], y[100:], scoring="r2", sample_weight=weights, random_state=0)


def test_reordered_groups_are_refused_in_rank_resampled():
    X, y = data()
    groups = pd.Series(np.repeat(np.arange(40), 5), index=X.index).sample(frac=1, random_state=0)
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)
    with pytest.raises(ValueError, match="index"):
        ranker.rank_resampled(X, y, scoring="r2", cv=GroupKFold(3), groups=groups, n_repeats=2, random_state=0)


def test_sorted_eval_target_is_refused_in_ablate():
    X, y = data()
    ranker = fitted(X, y)
    report = ranker.rank_features(X[100:], y[100:], scoring="r2", random_state=0)
    with pytest.raises(ValueError, match="index"):
        ranker.ablate(X[:100], y[:100], X[100:], y[100:].sort_values(), report, ks=[1])


def test_other_labels_warn_and_use_positions():
    X, y = data(index=np.arange(1000, 1200))
    ranker = fitted(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warned = ranker.rank_features(X[100:], y[100:].reset_index(drop=True), scoring="r2", random_state=0)
    assert [w for w in index_warnings(record) if "position" in str(w.message).lower()]
    plain = ranker.rank_features(X[100:], y[100:].to_numpy(), scoring="r2", random_state=0)
    assert warned["models"]["lin"]["evaluation_score"] == plain["models"]["lin"]["evaluation_score"]
    assert warned["consensus"] == plain["consensus"]


def test_aligned_and_numpy_inputs_do_not_warn():
    X, y = data(index=np.random.default_rng(5).permutation(200))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ranker = fitted(X, y)
        ranker.rank_features(X[100:], y[100:], scoring="r2", random_state=0)
        ranker.rank_features(X[100:], y[100:].to_numpy(), scoring="r2", random_state=0)
        numpy_ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)
        numpy_ranker.fit(X.to_numpy()[:100], y[:100].sort_values(), feature_names=["a", "b"])
    assert not index_warnings(record)


# 3. groups with an integer cv

def grouped():
    X, y = data()
    return X, y, np.repeat(np.arange(40), 5)


def groupkfold_warnings(record):
    return [w for w in record if issubclass(w.category, UserWarning) and "GroupKFold" in str(w.message)]


def test_groups_with_integer_cv_warn():
    X, y, groups = grouped()
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ranker.rank_resampled(X, y, scoring="r2", cv=3, groups=groups, n_repeats=2, random_state=0)
    assert groupkfold_warnings(record)


def test_selector_groups_with_integer_cv_warn():
    X, y, groups = grouped()
    selector = ConsensusSelector(
        task="reg", scoring="r2", k=1, models={"lin": LinearRegression()}, cv=3, n_repeats=2, random_state=0,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        selector.fit(X, y, groups=groups)
    assert groupkfold_warnings(record)


def test_group_splitter_does_not_warn():
    X, y, groups = grouped()
    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression()}, verbose=False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ranker.rank_resampled(X, y, scoring="r2", cv=GroupKFold(3), groups=groups, n_repeats=2, random_state=0)
    assert not groupkfold_warnings(record)
