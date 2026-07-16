import numpy as np
import pytest
from sklearn.metrics import get_scorer

from featranker import FeatureRanker


class MeanRegressor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def score(self, X, y):
        return -float(np.mean(np.abs(np.asarray(y) - self.predict(X))))


class FailFitRegressor(MeanRegressor):
    def fit(self, X, y):
        raise ValueError("fit exploded")


def ranker_with_models(monkeypatch, models):
    def init_models(self):
        return {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": models, "tree": {}},
        }

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)
    return FeatureRanker(task="reg", group="linear")


X_TRAIN = np.array(
    [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 1.0], [3.0, 3.0, 1.0]]
)
Y_TRAIN = np.array([0.0, 1.0, 1.0, 2.0])
X_EVAL = np.array([[4.0, 3.0, 0.0], [5.0, 5.0, 1.0], [6.0, 4.0, 1.0]])
Y_EVAL = np.array([2.0, 3.0, 3.0])


def fitted_ranker(monkeypatch, X_train=None, names=None, models=None):
    ranker = ranker_with_models(
        monkeypatch, models if models is not None else {"ok": MeanRegressor()}
    )
    ranker.fit(
        X_TRAIN if X_train is None else X_train,
        Y_TRAIN if X_train is None else np.arange(len(X_train), dtype=float),
        names if names is not None else ["a", "b", "c"],
    )
    return ranker


def test_constructor_is_unfitted_without_loading_default_prep(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    assert ranker.is_fitted_ is False
    assert ranker.trained_models == {}


def test_numpy_fit_requires_feature_names(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    with pytest.raises(ValueError, match="feature_names"):
        ranker.fit(np.ones((4, 2)), np.arange(4))


def test_fit_validates_rows_target_shape_and_name_count(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    with pytest.raises(ValueError, match="row"):
        ranker.fit(np.ones((4, 2)), np.arange(3), ["a", "b"])
    with pytest.raises(ValueError, match="one-dimensional"):
        ranker.fit(np.ones((4, 2)), np.ones((4, 1)), ["a", "b"])
    with pytest.raises(ValueError, match="2 feature names"):
        ranker.fit(np.ones((4, 2)), np.arange(4), ["a"])


def test_dataframe_names_are_retained_and_conflicts_rejected(monkeypatch):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    ranker.fit(frame, [1, 2])

    assert ranker.feature_names_ == ["a", "b"]
    conflicting = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    with pytest.raises(ValueError, match="conflict"):
        conflicting.fit(frame, [1, 2], ["b", "a"])


def test_fit_records_partial_failures(monkeypatch):
    ranker = ranker_with_models(
        monkeypatch, {"ok": MeanRegressor(), "bad": FailFitRegressor()}
    )

    returned = ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])

    assert returned is ranker
    assert ranker.is_fitted_ is True
    assert set(ranker.trained_models) == {"ok"}
    assert ranker.fit_failures_ == [
        {
            "model": "bad",
            "stage": "fit",
            "error_type": "ValueError",
            "message": "fit exploded",
        }
    ]


def test_fit_raises_when_every_model_fails(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"bad": FailFitRegressor()})

    with pytest.raises(RuntimeError, match="No models fit successfully"):
        ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])

    assert ranker.is_fitted_ is False


def test_normal_fit_does_not_retain_training_data(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    X = np.ones((4, 1))
    y = np.arange(4)

    ranker.fit(X, y, ["a"])

    assert not hasattr(ranker, "_legacy_X")
    assert not hasattr(ranker, "_legacy_y")
    assert ranker.n_features_in_ == 1


def test_rank_features_requires_fitted_state(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    with pytest.raises(RuntimeError, match="fit"):
        ranker.rank_features(X_EVAL, Y_EVAL, scoring="r2")


def test_rank_features_scores_evaluation_not_training(monkeypatch):
    seen = []

    def scorer(model, X, y):
        seen.append((X.copy(), np.asarray(y).copy()))
        return model.score(X, y)

    ranker = fitted_ranker(monkeypatch)

    report = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring=scorer, n_repeats=2, random_state=7
    )

    assert report["evaluation_mode"] == "held_out"
    assert all(X.shape == X_EVAL.shape for X, _ in seen)
    assert np.array_equal(seen[0][0], X_EVAL)
    assert np.array_equal(seen[0][1], Y_EVAL)


def test_named_scorer_sets_baseline_score(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

    report = ranker.rank_features(
        X_EVAL,
        Y_EVAL,
        scoring="neg_mean_absolute_error",
        n_repeats=2,
        random_state=1,
    )

    model = ranker.trained_models["ok"]
    expected = get_scorer("neg_mean_absolute_error")(model, X_EVAL, Y_EVAL)
    assert report["models"]["ok"]["evaluation_score"] == expected
    assert report["scoring"] == "neg_mean_absolute_error"


def test_group_columns_share_each_row_permutation(monkeypatch):
    observed = []

    def scorer(model, X, y):
        observed.append(X.copy())
        assert np.array_equal(X[:, 1] - X[:, 0], np.full(len(X), 100.0))
        return 0.0

    X = np.column_stack(
        [np.arange(8.0), np.arange(8.0) + 100.0, np.arange(8.0)]
    )
    ranker = fitted_ranker(monkeypatch, X_train=X, names=["g1", "g2", "other"])

    ranker.rank_features(
        X,
        np.arange(8.0),
        scoring=scorer,
        feature_groups={"gene": ["g1", "g2"]},
        n_repeats=3,
        random_state=2,
    )

    assert len(observed) == 7


def test_ungrouped_features_become_singletons(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

    report = ranker.rank_features(
        X_EVAL,
        Y_EVAL,
        scoring="r2",
        feature_groups={"pair": ["a", "b"]},
        n_repeats=2,
        random_state=2,
    )

    assert report["feature_groups"] == {"pair": ["a", "b"], "c": ["c"]}


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ({"empty": []}, "empty"),
        ({"bad": ["missing"]}, "unknown"),
        ({"duplicate": ["a", "a"]}, "duplicate"),
        ({"first": ["a"], "second": ["a"]}, "overlap"),
        ({"c": ["a", "b"]}, "collides"),
    ],
)
def test_invalid_feature_groups_fail_clearly(monkeypatch, groups, message):
    ranker = fitted_ranker(monkeypatch)

    with pytest.raises(ValueError, match=message):
        ranker.rank_features(
            X_EVAL, Y_EVAL, scoring="r2", feature_groups=groups, random_state=1
        )


def test_evaluation_shape_and_dataframe_names_are_validated(monkeypatch):
    pd = pytest.importorskip("pandas")
    ranker = fitted_ranker(monkeypatch)

    with pytest.raises(ValueError, match="3 feature names"):
        ranker.rank_features(np.ones((3, 2)), Y_EVAL, scoring="r2")
    with pytest.raises(ValueError, match="rows"):
        ranker.rank_features(X_EVAL, Y_EVAL[:2], scoring="r2")
    with pytest.raises(ValueError, match="names and order"):
        ranker.rank_features(
            pd.DataFrame(X_EVAL, columns=["b", "a", "c"]),
            Y_EVAL,
            scoring="r2",
        )


def test_rank_features_validates_repeat_count_and_scorer(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

    with pytest.raises(ValueError, match="n_repeats"):
        ranker.rank_features(X_EVAL, Y_EVAL, scoring="r2", n_repeats=0)
    with pytest.raises(TypeError, match="scoring"):
        ranker.rank_features(X_EVAL, Y_EVAL, scoring=123)
