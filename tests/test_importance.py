import numpy as np
import pytest

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
