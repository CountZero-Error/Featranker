from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import get_scorer

from featranker import FeatureRanker, build_ranker
import featranker.importance as importance_module

OPTIONAL_NAMES = {"xgboost", "lightgbm", "catboost"}
from featranker.importance import _assign_ranks, _build_consensus


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


class FailingScoreRegressor(MeanRegressor):
    def score(self, X, y):
        raise ValueError("score exploded")


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


def test_ties_share_minimum_rank():
    assert _assign_ranks({"b": 2.0, "a": 2.0, "c": 1.0}) == {
        "a": 1,
        "b": 1,
        "c": 3,
    }


def test_consensus_uses_model_ranks_not_raw_importance():
    model_reports = {
        "large_scale": {
            "importance": {
                "a": {"mean": 1000.0, "rank": 1},
                "b": {"mean": 999.0, "rank": 2},
            }
        },
        "small_scale": {
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
    }
    assert consensus["b"]["mean_rank"] == 1.5
    assert "mean_importance" not in consensus["a"]


def test_model_report_contains_raw_importance_and_rank(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

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
        models={"ok": MeanRegressor(), "bad": FailingScoreRegressor()},
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


def test_legacy_prep_workflow_warns_and_marks_in_sample(monkeypatch, tmp_path):
    prep_path = tmp_path / "clinical_prep.py"
    prep_path.write_text(
        "class Prep:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.offset = kwargs.get('offset', 0)\n"
        "    def _calc_features(self):\n"
        "        return {\n"
        "            'age': [50, 60, 70, 80],\n"
        "            'dose': [1, 2, 3, 4],\n"
        "            'label': [2, 3, 4, 5],\n"
        "        }\n",
        encoding="utf-8",
    )

    def init_models(self):
        return {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {"ok": MeanRegressor()}, "tree": {}},
        }

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)
    ranker = build_ranker(
        task="reg",
        group="linear",
        prep_file=str(prep_path),
        prep_class="Prep",
        offset=1,
    )

    assert ranker.is_fitted_
    assert hasattr(ranker, "_legacy_X")
    assert hasattr(ranker, "_legacy_y")
    with pytest.warns(DeprecationWarning, match="in-sample"):
        report = ranker.rankFeatures()

    assert report["evaluation_mode"] == "in_sample"
    assert report["scoring"] == "r2"


def test_explicit_constructor_prep_file_remains_supported(monkeypatch, tmp_path):
    prep_path = tmp_path / "prep.py"
    prep_path.write_text(
        "class Prep:\n"
        "    def _calc_features(self):\n"
        "        return {'x': [1, 2, 3], 'label': [2, 4, 6]}\n",
        encoding="utf-8",
    )

    def init_models(self):
        return {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {"ok": MeanRegressor()}, "tree": {}},
        }

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)

    ranker = FeatureRanker(
        task="reg",
        group="linear",
        prep_file=str(prep_path),
        prep_class="Prep",
    )

    assert ranker.is_fitted_
    assert ranker.feature_names_ == ["x"]


def test_legacy_rank_features_unavailable_after_normal_fit(monkeypatch):
    ranker = fitted_ranker(monkeypatch)

    with pytest.raises(RuntimeError, match="evaluation data"):
        ranker.rankFeatures()


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


def test_optional_estimators_are_not_core_dependencies():
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    core, extras = project.split("[project.optional-dependencies]", maxsplit=1)

    assert all(f'  "{name}",' not in core for name in OPTIONAL_NAMES)
    assert 'xgboost = ["xgboost"]' in extras
    assert 'lightgbm = ["lightgbm"]' in extras
    assert 'catboost = ["catboost"]' in extras
    assert 'all = ["xgboost", "lightgbm", "catboost"]' in extras
    assert 'test = ["pytest", "pandas", "build"]' in extras
