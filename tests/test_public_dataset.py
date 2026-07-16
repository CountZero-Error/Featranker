import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker


def test_diabetes_regression_end_to_end(monkeypatch):
    dataset = load_diabetes()
    X_train, X_eval, y_train, y_eval = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
    )

    def init_models(self):
        return {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {"ridge": Ridge()}, "tree": {}},
        }

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)
    ranker = FeatureRanker(task="reg", group="linear")
    ranker.fit(X_train, y_train, feature_names=list(dataset.feature_names))

    report = ranker.rank_features(
        X_eval,
        y_eval,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
    )

    assert report["evaluation_mode"] == "held_out"
    assert set(report["models"]) == {"ridge"}
    assert np.isfinite(report["models"]["ridge"]["evaluation_score"])
    assert set(report["feature_groups"]) == set(dataset.feature_names)
    assert len(report["consensus"]) == dataset.data.shape[1]
    assert all(row["n_models"] == 1 for row in report["consensus"])
    assert report["failures"] == []
