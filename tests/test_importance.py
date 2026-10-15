import importlib.metadata
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

import featranker
from featranker import FeatureRanker
import featranker.importance as importance_module

OPTIONAL_NAMES = {"xgboost", "lightgbm", "catboost"}
from featranker.importance import _assign_ranks, _build_consensus


class MeanRegressor(RegressorMixin, BaseEstimator):
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


class FailingScoreRegressor(MeanRegressor):
    def score(self, X, y):
        raise ValueError("score exploded")


class FixedScoreRegressor(MeanRegressor):
    def __init__(self, score_value):
        self.score_value = score_value

    def score(self, X, y):
        return self.score_value


class ConstantRegressor(MeanRegressor):
    """Predicts the training mean plus an offset; offset 0 ties the dummy."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def predict(self, X):
        return np.full(len(X), self.mean_ + self.offset)


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
        monkeypatch, models if models is not None else {"ok": LinearRegression()}
    )
    ranker.fit(
        X_TRAIN if X_train is None else X_train,
        Y_TRAIN if X_train is None else np.arange(len(X_train), dtype=float),
        names if names is not None else ["a", "b", "c"],
    )
    return ranker


def test_constructor_is_unfitted(monkeypatch):
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


def test_fit_records_feature_count(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    X = np.ones((4, 1))
    y = np.arange(4)

    ranker.fit(X, y, ["a"])

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


@pytest.mark.filterwarnings("ignore:No model beat the dummy baseline")
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

    assert len(observed) == 8  # dummy baseline + model baseline + 2 groups x 3


@pytest.mark.filterwarnings("ignore:No model beat the dummy baseline")
def test_permutation_only_changes_group_columns_and_spares_input(monkeypatch):
    seen = []

    def scorer(model, X, y):
        if not isinstance(model, DummyRegressor):
            seen.append(X.copy())
        return 0.0

    X = np.column_stack([np.arange(8.0), np.arange(8.0) * 2, np.arange(8.0) * 3])
    X_before = X.copy()
    ranker = fitted_ranker(monkeypatch, X_train=X, names=["a", "b", "c"])

    ranker.rank_features(
        X,
        np.arange(8.0),
        scoring=scorer,
        feature_groups={"ab": ["a", "b"]},
        n_repeats=3,
        random_state=3,
    )

    assert np.array_equal(X, X_before)
    assert np.array_equal(seen[0], X)  # unpermuted baseline call
    ab_calls, c_calls = seen[1:4], seen[4:7]
    for call in ab_calls:
        assert np.array_equal(call[:, 2], X[:, 2])
        assert not np.array_equal(call[:, :2], X[:, :2])
    for call in c_calls:  # earlier group's columns restored
        assert np.array_equal(call[:, :2], X[:, :2])
        assert not np.array_equal(call[:, 2], X[:, 2])


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


def test_ties_share_minimum_rank():
    assert _assign_ranks({"b": 2.0, "a": 2.0, "c": 1.0}) == {
        "a": 1,
        "b": 1,
        "c": 3,
    }


def test_consensus_uses_model_ranks_not_raw_importance():
    model_reports = {
        "large_scale": {
            "in_consensus": True,
            "importance": {
                "a": {"mean": 1000.0, "rank": 1},
                "b": {"mean": 999.0, "rank": 2},
            }
        },
        "small_scale": {
            "in_consensus": True,
            "importance": {
                "a": {"mean": 0.0, "rank": 2},
                "b": {"mean": 1.0, "rank": 1},
            }
        },
    }

    consensus = {
        row["feature_group"]: row for row in _build_consensus(model_reports)
    }

    assert consensus["a"] == {
        "feature_group": "a",
        "median_rank": 1.5,
        "mean_rank": 1.5,
        "rank_std": 0.5,
        "n_models": 2,
        "n_families": 2,
        "family_ranks": {"large_scale": 1.0, "small_scale": 2.0},
    }
    assert consensus["b"]["mean_rank"] == 1.5
    assert "mean_importance" not in consensus["a"]


def baseline_report(monkeypatch, models, scoring="r2"):
    ranker = fitted_ranker(monkeypatch, models=models)
    return ranker.rank_features(
        X_EVAL, Y_EVAL, scoring=scoring, n_repeats=2, random_state=1
    )


def test_models_beating_dummy_baseline_join_consensus(monkeypatch):
    report = baseline_report(
        monkeypatch, {"good": LinearRegression(), "bad": ConstantRegressor(100.0)}
    )

    dummy = DummyRegressor().fit(X_TRAIN, Y_TRAIN)
    assert report["baseline_score"] == get_scorer("r2")(dummy, X_EVAL, Y_EVAL)
    assert report["models"]["good"]["in_consensus"] is True
    assert report["models"]["bad"]["in_consensus"] is False
    assert set(report["models"]) == {"good", "bad"}
    assert all(row["n_models"] == 1 for row in report["consensus"])
    assert len(report["consensus"]) == 3


def test_no_model_beating_baseline_warns_and_empties_consensus(monkeypatch):
    with pytest.warns(UserWarning, match="dummy baseline") as caught:
        report = baseline_report(
            monkeypatch,
            {"tie": ConstantRegressor(0.0), "bad": ConstantRegressor(100.0)},
        )

    assert report["consensus"] == []
    assert caught[0].filename == __file__  # points at the rank_features caller
    assert not any(m["in_consensus"] for m in report["models"].values())
    assert set(report["models"]) == {"tie", "bad"}


def test_baseline_fit_failure_disables_gating(monkeypatch):
    class FailFitDummy(DummyRegressor):
        def fit(self, X, y, sample_weight=None):
            raise ValueError("dummy fit exploded")

    monkeypatch.setattr(importance_module, "DummyRegressor", FailFitDummy)

    report = baseline_report(monkeypatch, {"bad": ConstantRegressor(100.0)})

    assert report["baseline_score"] is None
    assert report["models"]["bad"]["in_consensus"] is True
    assert len(report["consensus"]) == 3
    failures = [f for f in report["failures"] if f["model"] == "dummy_baseline"]
    assert len(failures) == 1
    assert failures[0]["message"] == "dummy fit exploded"


def test_baseline_score_failure_disables_gating(monkeypatch):
    def scorer(model, X, y):
        if isinstance(model, DummyRegressor):
            raise ValueError("dummy score exploded")
        return model.score(X, y)

    report = baseline_report(
        monkeypatch, {"bad": ConstantRegressor(100.0)}, scoring=scorer
    )

    assert report["baseline_score"] is None
    assert report["models"]["bad"]["in_consensus"] is True
    assert len(report["consensus"]) == 3
    failures = [f for f in report["failures"] if f["model"] == "dummy_baseline"]
    assert [f["stage"] for f in failures] == ["ranking"]


def test_classification_baseline_is_fit_on_encoded_labels(monkeypatch):
    ranker = clf_ranker_with_models(monkeypatch, {"ok": RecordingClassifier()})

    ranker.fit(X_TRAIN, np.array(["x", "y", "z", "z"]), ["a", "b", "c"])

    assert isinstance(ranker.baseline_model_, DummyClassifier)
    assert ranker.baseline_model_.classes_.tolist() == [0, 1, 2]


def test_model_report_contains_raw_importance_and_rank(monkeypatch):
    ranker = fitted_ranker(monkeypatch, models={"ok": LinearRegression()})

    report = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring="r2", n_repeats=3, random_state=42
    )

    importance = report["models"]["ok"]["importance"]["a"]
    assert len(importance["values"]) == 3
    assert set(importance) == {"values", "mean", "std", "rank"}
    assert report["consensus"][0]["n_models"] == 1


def test_fixed_seed_is_reproducible(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

    first = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring="r2", n_repeats=4, random_state=42
    )
    second = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring="r2", n_repeats=4, random_state=42
    )

    assert first == second


def test_partial_ranking_failure_is_reported_and_excluded(monkeypatch):
    ranker = fitted_ranker(
        monkeypatch,
        models={"ok": LinearRegression(), "bad": FailingScoreRegressor()},
    )

    report = ranker.rank_features(
        X_EVAL,
        Y_EVAL,
        scoring=lambda model, X, y: model.score(X, y),
        n_repeats=2,
        random_state=1,
    )

    assert set(report["models"]) == {"ok"}
    assert all(row["n_models"] == 1 for row in report["consensus"])
    assert any(
        failure["model"] == "bad" and failure["stage"] == "ranking"
        for failure in report["failures"]
    )


def test_all_ranking_failures_raise(monkeypatch):
    ranker = fitted_ranker(
        monkeypatch, models={"bad": FailingScoreRegressor()}
    )

    with pytest.raises(RuntimeError, match="No models ranked successfully"):
        ranker.rank_features(
            X_EVAL,
            Y_EVAL,
            scoring=lambda model, X, y: model.score(X, y),
            n_repeats=2,
        )


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_non_finite_baseline_is_reported_and_excluded(monkeypatch, non_finite):
    ranker = fitted_ranker(
        monkeypatch,
        models={
            "ok": LinearRegression(),
            "bad": FixedScoreRegressor(non_finite),
        },
    )

    report = ranker.rank_features(
        X_EVAL,
        Y_EVAL,
        scoring=lambda model, X, y: model.score(X, y),
        n_repeats=2,
        random_state=1,
    )

    assert set(report["models"]) == {"ok"}
    failure = next(item for item in report["failures"] if item["model"] == "bad")
    assert failure["stage"] == "ranking"
    assert "baseline score" in failure["message"]
    assert "finite" in failure["message"]
    for model_report in report["models"].values():
        assert np.isfinite(model_report["evaluation_score"])
        for importance in model_report["importance"].values():
            assert np.all(np.isfinite(importance["values"]))
            assert np.isfinite(importance["mean"])
            assert np.isfinite(importance["std"])
    assert all(
        np.isfinite(row[key])
        for row in report["consensus"]
        for key in ("median_rank", "mean_rank", "rank_std")
    )


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_non_finite_permuted_score_causes_zero_success_error(
    monkeypatch, non_finite
):
    calls = 0

    def scorer(model, X, y):
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 2 else non_finite  # dummy, then model baseline

    ranker = fitted_ranker(monkeypatch, models={"bad": MeanRegressor()})

    with pytest.raises(
        RuntimeError,
        match="No models ranked successfully.*permuted score.*finite",
    ):
        ranker.rank_features(
            X_EVAL,
            Y_EVAL,
            scoring=scorer,
            n_repeats=2,
            random_state=1,
        )


def test_legacy_prep_api_is_removed():
    assert not hasattr(featranker, "build_ranker")
    assert not hasattr(importance_module, "build_ranker")
    assert not hasattr(FeatureRanker, "rankFeatures")
    with pytest.raises(TypeError):
        FeatureRanker(task="reg", prep_file="prep.py")


class RecordingRegressor(MeanRegressor):
    def fit(self, X, y):
        self.seen_y_ = np.asarray(y)
        return super().fit(X, y)


def test_regression_keeps_object_numeric_target_values(monkeypatch):
    model = RecordingRegressor()
    ranker = ranker_with_models(monkeypatch, {"ok": model})

    ranker.fit(X_TRAIN, np.array([10, 20, 100, 200], dtype=object), ["a", "b", "c"])

    assert ranker.label_encoder is None
    assert ranker.trained_models["ok"].seen_y_.tolist() == [10.0, 20.0, 100.0, 200.0]


def test_regression_rejects_non_numeric_target(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})

    with pytest.raises(ValueError, match="numeric"):
        ranker.fit(X_TRAIN, np.array(["a", "b", "c", "d"]), ["a", "b", "c"])


class RecordingClassifier(BaseEstimator):
    def fit(self, X, y):
        self.seen_y_ = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def clf_ranker_with_models(monkeypatch, models):
    def init_models(self):
        return {"clf": {"linear": models, "tree": {}}, "reg": {}}

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)
    return FeatureRanker(task="clf", group="linear")


def test_classification_encodes_numeric_labels_from_zero(monkeypatch):
    model = RecordingClassifier()
    ranker = clf_ranker_with_models(monkeypatch, {"ok": model})

    ranker.fit(X_TRAIN, np.array([1, 2, 3, 3]), ["a", "b", "c"])

    assert ranker.trained_models["ok"].seen_y_.tolist() == [0, 1, 2, 2]


def test_failed_refit_leaves_ranker_unfitted(monkeypatch):
    ranker = clf_ranker_with_models(monkeypatch, {"ok": RecordingClassifier()})
    ranker.fit(X_TRAIN, np.array([0, 1, 1, 0]), ["a", "b", "c"])

    with pytest.raises(TypeError):
        ranker.fit(X_TRAIN, np.array(["a", 1, "b", 2], dtype=object), ["a", "b", "c"])

    assert ranker.is_fitted_ is False


def test_missing_optional_estimators_skip_only_affected_models(monkeypatch):
    real_import = importance_module.importlib.import_module

    def import_without_optional(module_name):
        if module_name in {"xgboost", "catboost"}:
            raise ModuleNotFoundError(f"No module named '{module_name}'")
        return real_import(module_name)

    monkeypatch.setattr(
        importance_module.importlib, "import_module", import_without_optional
    )

    with pytest.warns(UserWarning, match="optional dependency"):
        ranker = FeatureRanker(task="reg", group="tree")

    tree_models = ranker.models["reg"]["tree"]
    assert "decision_tree_regressor" in tree_models
    assert "xgboost_regressor" not in tree_models
    assert "catboost_regressor" not in tree_models
    assert {failure["model"] for failure in ranker.initialization_failures_} == {
        "xgboost_regressor",
        "catboost_regressor",
    }
    assert all(
        failure["stage"] == "initialization"
        for failure in ranker.initialization_failures_
    )


def test_optional_binary_load_error_skips_model_and_core_model_still_fits(
    monkeypatch,
):
    config = {
        "regression": {
            "tree": [
                {
                    "name": "core",
                    "import": "core_estimators",
                    "class": "MeanRegressor",
                },
                {
                    "name": "optional",
                    "import": "xgboost",
                    "class": "XGBRegressor",
                },
            ]
        }
    }

    monkeypatch.setattr(importance_module.yaml, "safe_load", lambda stream: config)

    def import_models(module_name):
        if module_name == "xgboost":
            raise OSError("binary image not found")
        return SimpleNamespace(MeanRegressor=MeanRegressor)

    monkeypatch.setattr(importance_module.importlib, "import_module", import_models)

    with pytest.warns(UserWarning, match="optional.*binary image not found"):
        ranker = FeatureRanker(task="reg", group="tree")

    ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])
    assert set(ranker.trained_models) == {"core"}
    assert ranker.initialization_failures_ == [
        {
            "model": "optional",
            "stage": "initialization",
            "error_type": "OSError",
            "message": "binary image not found",
        }
    ]


def test_constructor_failure_is_recorded_without_blocking_other_model(monkeypatch):
    class BrokenRegressor:
        def __init__(self):
            raise ValueError("invalid constructor")

    config = {
        "regression": {
            "linear": [
                {
                    "name": "broken",
                    "import": "estimators",
                    "class": "BrokenRegressor",
                },
                {
                    "name": "core",
                    "import": "estimators",
                    "class": "MeanRegressor",
                },
            ]
        }
    }
    module = SimpleNamespace(
        BrokenRegressor=BrokenRegressor,
        MeanRegressor=MeanRegressor,
    )
    monkeypatch.setattr(importance_module.yaml, "safe_load", lambda stream: config)
    monkeypatch.setattr(
        importance_module.importlib, "import_module", lambda module_name: module
    )

    with pytest.warns(UserWarning, match="broken.*invalid constructor"):
        ranker = FeatureRanker(task="reg", group="linear")

    assert set(ranker.models["reg"]["linear"]) == {"core"}
    assert ranker.initialization_failures_[0] == {
        "model": "broken",
        "stage": "initialization",
        "error_type": "ValueError",
        "message": "invalid constructor",
    }


def test_zero_success_fit_error_includes_initialization_failures(monkeypatch):
    class BrokenRegressor:
        def __init__(self):
            raise RuntimeError("constructor unavailable")

    config = {
        "regression": {
            "linear": [
                {
                    "name": "broken",
                    "import": "estimators",
                    "class": "BrokenRegressor",
                }
            ]
        }
    }
    monkeypatch.setattr(importance_module.yaml, "safe_load", lambda stream: config)
    monkeypatch.setattr(
        importance_module.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(BrokenRegressor=BrokenRegressor),
    )

    with pytest.warns(UserWarning, match="broken.*constructor unavailable"):
        ranker = FeatureRanker(task="reg", group="linear")
    with pytest.raises(RuntimeError, match="broken.*constructor unavailable"):
        ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])

    assert ranker.is_fitted_ is False


def test_optional_estimators_are_not_core_dependencies():
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    core, extras = project.split("[project.optional-dependencies]", maxsplit=1)

    assert all(f'  "{name}",' not in core for name in OPTIONAL_NAMES)
    assert 'xgboost = ["xgboost"]' in extras
    assert 'lightgbm = ["lightgbm"]' in extras
    assert 'catboost = ["catboost"]' in extras
    assert 'all = ["xgboost", "lightgbm", "catboost"]' in extras
    assert 'test = ["pytest", "pandas", "build"]' in extras


def test_pre_1_8_sklearn_gets_penalty_from_l1_ratio(monkeypatch):
    monkeypatch.setattr(importance_module, "SKLEARN_NEEDS_PENALTY", True)

    models = FeatureRanker(task="clf", group="linear").models["clf"]["linear"]

    def inner(name):  # the estimator wrapped by make_pipeline(StandardScaler(), ...)
        return models[name].steps[-1][1]

    assert inner("logistic_regression_l1").penalty == "l1"
    assert inner("logistic_regression_l2").penalty == "l2"
    assert inner("logistic_regression_elasticnet").penalty == "elasticnet"
    assert inner("logistic_regression_elasticnet").l1_ratio == 0.5


def test_user_models_replace_registry_and_are_not_mutated(monkeypatch):
    def fail_init(self):
        raise AssertionError("registry must not load when models= is given")

    monkeypatch.setattr(FeatureRanker, "init_models", fail_init)
    ridge = Ridge()
    pipeline = make_pipeline(StandardScaler(), Ridge())
    ranker = FeatureRanker(task="reg", models={"ridge": ridge, "pipe": pipeline})

    ranker.fit(X_TRAIN, Y_TRAIN, ["a", "b", "c"])
    report = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring="r2", n_repeats=2, random_state=0
    )

    assert set(report["models"]) == {"ridge", "pipe"}
    assert ranker.trained_models["ridge"] is not ridge
    for untouched in (ridge, pipeline):
        with pytest.raises(NotFittedError):
            check_is_fitted(untouched)


@pytest.mark.parametrize(
    ("models", "error", "message"),
    [
        ({}, ValueError, "non-empty"),
        ([Ridge()], TypeError, "dict"),
        ({"": Ridge()}, ValueError, "non-empty"),
        ({1: Ridge()}, ValueError, "non-empty"),
        ({"bad": object()}, TypeError, "fit"),
        ({"dummy_baseline": Ridge()}, ValueError, "reserved"),
    ],
)
def test_user_models_are_validated(models, error, message):
    with pytest.raises(error, match=message):
        FeatureRanker(task="reg", models=models)


def test_user_models_reject_explicit_group():
    with pytest.raises(ValueError, match="group"):
        FeatureRanker(task="reg", group="linear", models={"ridge": Ridge()})


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_builtin_linear_models_are_standardized_and_tree_models_are_not(task):
    registry = FeatureRanker(task=task).models[task]

    assert registry["linear"]
    for model in registry["linear"].values():
        assert isinstance(model, Pipeline)
        assert isinstance(model.steps[0][1], StandardScaler)
    for model in registry["tree"].values():
        assert not isinstance(model, Pipeline)


def _score_fails_for_trees(model, X, y):
    if isinstance(model, DecisionTreeRegressor):
        raise ValueError("tree scoring exploded")
    return float(model.score(X, y))


def _real_models():
    return {
        "ridge": Ridge(),
        "lasso": Lasso(alpha=0.5),
        "tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    }


def _diabetes_split():
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    return (
        data.data[:80],
        data.target[:80],
        data.data[80:140],
        data.target[80:140],
        list(data.feature_names),
    )


def test_n_jobs_report_matches_sequential_report():
    X_train, y_train, X_eval, y_eval, names = _diabetes_split()
    reports = []
    for n_jobs in (None, 2):
        ranker = FeatureRanker(task="reg", models=_real_models(), n_jobs=n_jobs)
        ranker.fit(X_train, y_train, names)
        reports.append(
            ranker.rank_features(
                X_eval, y_eval, scoring="r2", n_repeats=3, random_state=0
            )
        )

    assert reports[0] == reports[1]
    assert list(reports[1]["models"]) == ["ridge", "lasso", "tree"]


def test_n_jobs_keeps_per_model_failure_semantics():
    X_train, y_train, X_eval, y_eval, names = _diabetes_split()
    models = {"ok": Ridge(), "bad": Ridge(alpha=-1.0), "also_ok": Lasso(alpha=0.5)}
    ranker = FeatureRanker(task="reg", models=models, n_jobs=2)

    ranker.fit(X_train, y_train, names)

    assert list(ranker.trained_models) == ["ok", "also_ok"]
    assert [(f["model"], f["stage"]) for f in ranker.fit_failures_] == [
        ("bad", "fit")
    ]


def test_n_jobs_keeps_per_model_ranking_failure_semantics():
    X_train, y_train, X_eval, y_eval, names = _diabetes_split()
    ranker = FeatureRanker(
        task="reg", models={"ok": Ridge(), "tree": DecisionTreeRegressor()}, n_jobs=2
    )
    ranker.fit(X_train, y_train, names)

    report = ranker.rank_features(
        X_eval, y_eval, scoring=_score_fails_for_trees, n_repeats=2, random_state=0
    )

    assert list(report["models"]) == ["ok"]
    assert [(f["model"], f["stage"]) for f in report["failures"]] == [
        ("tree", "ranking")
    ]


class ParamsEstimator(BaseEstimator):
    def __init__(
        self, a=1, b=float("nan"), c=float("inf"), d="x", e=None, f=True,
        g=(1, 2), inner=None,
    ):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.e, self.f, self.g, self.inner = e, f, g, inner


def test_json_params_keeps_only_finite_scalars():
    params = importance_module._json_params(ParamsEstimator(inner=Ridge(alpha=2.0)))

    assert params["a"] == 1
    assert params["d"] == "x"
    assert params["e"] is None
    assert params["f"] is True
    assert params["inner__alpha"] == 2.0
    assert not {"b", "c", "g", "inner"} & set(params)
    json.dumps(params, allow_nan=False)


def test_report_has_provenance_and_model_params(monkeypatch):
    ranker = fitted_ranker(
        monkeypatch, models={"pipe": make_pipeline(StandardScaler(), Ridge(alpha=3.0))}
    )

    report = ranker.rank_features(
        X_EVAL, Y_EVAL, scoring="r2", n_repeats=2, random_state=0
    )

    provenance = report["provenance"]
    assert set(provenance) == {"featranker", "python", "numpy", "scikit-learn"}
    assert provenance["python"] == platform.python_version()
    assert provenance["numpy"] == np.__version__
    assert provenance["scikit-learn"] == sklearn.__version__
    assert report["models"]["pipe"]["params"]["ridge__alpha"] == 3.0
    json.dumps(report, allow_nan=False)


def test_provenance_featranker_is_none_when_not_installed(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importance_module.importlib.metadata, "version", missing)

    assert importance_module._provenance()["featranker"] is None


class PlainMeanModel:
    """Duck-typed estimator without get_params (not clonable by sklearn)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_user_models_without_get_params_are_copied_not_rejected():
    plain = PlainMeanModel()
    ranker = FeatureRanker(task="reg", models={"plain": plain})

    ranker.fit(X_TRAIN, Y_TRAIN, ["a", "b", "c"])

    assert ranker.fit_failures_ == []
    assert not hasattr(plain, "mean_")


@pytest.mark.parametrize("eval_label", ["yesss", "maybe"])
def test_unseen_eval_label_is_rejected_with_its_full_name(monkeypatch, eval_label):
    ranker = clf_ranker_with_models(monkeypatch, {"ok": RecordingClassifier()})
    ranker.fit(X_TRAIN, np.array(["yes", "no", "yes", "no"]), ["a", "b", "c"])

    with pytest.raises(ValueError, match=eval_label):
        ranker.rank_features(
            X_EVAL, np.array(["yes", eval_label, "no"]), scoring="accuracy"
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_features_are_rejected_in_fit_and_rank(monkeypatch, bad):
    ranker = ranker_with_models(monkeypatch, {"ok": LinearRegression()})
    X_bad = X_TRAIN.copy()
    X_bad[1, 2] = bad

    with pytest.raises(ValueError, match="finite"):
        ranker.fit(X_bad, Y_TRAIN, ["a", "b", "c"])

    fitted = fitted_ranker(monkeypatch)
    X_eval_bad = X_EVAL.copy()
    X_eval_bad[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        fitted.rank_features(X_eval_bad, Y_EVAL, scoring="r2")


def _run_shipped_linear(verbose):
    import warnings

    kwargs = {} if verbose is None else {"verbose": verbose}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker = FeatureRanker(task="reg", group="linear", **kwargs)
        ranker.fit(X_TRAIN, Y_TRAIN, ["a", "b", "c"])
        ranker.rank_features(X_EVAL, Y_EVAL, scoring="r2", n_repeats=1, random_state=0)


def test_verbose_false_writes_no_progress(capsys):
    _run_shipped_linear(verbose=False)

    err = capsys.readouterr().err
    assert "[*]" not in err and "it/s" not in err


def test_progress_is_shown_by_default(capsys):
    _run_shipped_linear(verbose=None)

    err = capsys.readouterr().err
    for step in ("[*] Initialize", "[*] Training models", "[*] Ranking models"):
        assert step in err


def test_verbose_must_be_bool():
    with pytest.raises(TypeError, match="verbose"):
        FeatureRanker(task="reg", verbose="no")


def test_quiet_run_still_warns_on_empty_consensus():
    ranker = FeatureRanker(
        task="reg", models={"tie": ConstantRegressor(0.0)}, verbose=False
    )
    ranker.fit(X_TRAIN, Y_TRAIN, ["a", "b", "c"])

    with pytest.warns(UserWarning, match="dummy baseline"):
        report = ranker.rank_features(
            X_EVAL, Y_EVAL, scoring="r2", n_repeats=1, random_state=0
        )
    assert report["consensus"] == []
