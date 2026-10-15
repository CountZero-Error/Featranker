import json

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
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


REPORT_KEYS = {
    "evaluation_mode",
    "scoring",
    "n_repeats",
    "random_state",
    "feature_groups",
    "baseline_score",
    "models",
    "consensus",
    "failures",
}


def _assert_shipped_registry_report(report, n_features):
    assert REPORT_KEYS <= set(report)
    # Models that failed to initialize (e.g. absent optional xgboost/catboost)
    # only report stage "initialization"; any fit/ranking failure is a bug.
    broken = [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]
    assert broken == []
    assert report["models"]
    assert isinstance(report["baseline_score"], float)
    assert any(m["in_consensus"] for m in report["models"].values())
    assert len(report["consensus"]) == n_features


def test_iris_classification_with_shipped_registry():
    dataset = load_iris()
    X_train, X_eval, y_train, y_eval = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.3,
        random_state=0,
        stratify=dataset.target,
    )
    # Shifted labels (1..3) catch estimators that need 0-based classes.
    ranker = FeatureRanker(task="clf", group="all")
    ranker.fit(X_train, y_train + 1, feature_names=list(dataset.feature_names))

    report = ranker.rank_features(
        X_eval, y_eval + 1, scoring="accuracy", n_repeats=2, random_state=0
    )

    _assert_shipped_registry_report(report, dataset.data.shape[1])
    assert {"logistic_regression", "random_forest"} <= set(report["models"])
    families = {name: m["family"] for name, m in report["models"].items()}
    assert families["logistic_regression_l1"] == families["logistic_regression"]
    assert families["logistic_regression"] != families["random_forest"]


def test_diabetes_regression_with_shipped_registry():
    dataset = load_diabetes()
    X_train, X_eval, y_train, y_eval = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=0
    )
    ranker = FeatureRanker(task="reg", group="all")
    ranker.fit(X_train, y_train, feature_names=list(dataset.feature_names))

    report = ranker.rank_features(
        X_eval, y_eval, scoring="r2", n_repeats=2, random_state=0
    )

    _assert_shipped_registry_report(report, dataset.data.shape[1])
    assert {"ridge_regression", "random_forest_regressor"} <= set(report["models"])


def test_shipped_registry_reports_are_strict_json():
    dataset = load_diabetes()
    ranker = FeatureRanker(task="reg", group="all")
    ranker.fit(dataset.data[:200], dataset.target[:200], list(dataset.feature_names))

    report = ranker.rank_features(
        dataset.data[200:], dataset.target[200:], scoring="r2", n_repeats=2,
        random_state=0,
    )

    json.dumps(report, allow_nan=False)
    assert all(m["params"] for m in report["models"].values())


def test_shipped_linear_classifiers_are_reproducible():
    dataset = load_breast_cancer()
    names = list(dataset.feature_names)

    def run():
        ranker = FeatureRanker(task="clf", group="linear")
        ranker.fit(dataset.data[:400], dataset.target[:400], feature_names=names)
        return ranker.rank_features(
            dataset.data[400:], dataset.target[400:], scoring="accuracy",
            n_repeats=2, random_state=0,
        )

    assert run()["models"] == run()["models"]
