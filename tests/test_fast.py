import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

import featranker.importance as importance_module
from featranker import ConsensusSelector, FeatureRanker

CONFIG = Path(__file__).parents[1] / "featranker" / "importance_config.yaml"
FAST_MODELS = {
    "clf": {
        "logistic_regression", "ridge_classifier", "lda", "naive_bayes_gaussian",
        "decision_tree", "random_forest", "hist_gradient_boosting",
    },
    "reg": {
        "ridge_regression", "svr_regression", "decision_tree_regressor",
        "random_forest_regressor", "hist_gradient_boosting_regressor",
    },
}
RNG = np.random.default_rng(0)
X_DATA = RNG.normal(size=(60, 3))
Y_REG = 2.0 * X_DATA[:, 0] + 0.1 * RNG.normal(size=60)
Y_CLF = (X_DATA[:, 0] + 0.1 * RNG.normal(size=60) > 0).astype(int)
Y_BY_TASK = {"clf": Y_CLF, "reg": Y_REG}


def test_fast_group_is_accepted():
    ranker = FeatureRanker("reg", group="fast", verbose=False)

    assert ranker.group == "fast"


def test_unknown_group_and_fast_with_models_raise():
    with pytest.raises(ValueError, match="fast"):
        FeatureRanker("reg", group="quick")
    with pytest.raises(ValueError, match="group"):
        FeatureRanker("reg", group="fast", models={"m": Ridge()})


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_fast_fits_exactly_the_representatives(task):
    ranker = FeatureRanker(task, group="fast", verbose=False)

    ranker.fit(X_DATA, Y_BY_TASK[task], ["a", "b", "c"])

    assert set(ranker.trained_models) == FAST_MODELS[task]
    assert ranker.fit_failures_ == []


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_fast_models_cover_each_family_once(task):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    entries = [e for g in config[{"clf": "classification", "reg": "regression"}[task]].values() for e in g]
    all_families = {entry["family"] for entry in entries}

    ranker = FeatureRanker(task, group="fast", verbose=False)

    fast_families = list(ranker.model_families_.values())
    assert sorted(fast_families) == sorted(all_families)
    assert {e["name"] for e in entries if e.get("fast")} == FAST_MODELS[task]


def test_fast_linear_representatives_keep_the_scaler():
    models = FeatureRanker("clf", group="fast", verbose=False).models["clf"]["linear"]

    assert isinstance(models["ridge_classifier"], Pipeline)
    assert models["ridge_classifier"].steps[0][0] == "standardscaler"


def test_fast_never_imports_optional_dependencies(monkeypatch):
    real_import = importance_module.importlib.import_module
    imported = []

    def import_without_optional(module_name):
        imported.append(module_name)
        if module_name in {"xgboost", "catboost"}:
            raise ModuleNotFoundError(f"No module named '{module_name}'")
        return real_import(module_name)

    monkeypatch.setattr(
        importance_module.importlib, "import_module", import_without_optional
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ranker = [FeatureRanker(t, group="fast", verbose=False) for t in ("clf", "reg")]

    assert all(r.initialization_failures_ == [] for r in ranker)
    assert not {"xgboost", "catboost"} & set(imported)


def test_rank_resampled_uses_the_fast_set_on_every_split():
    ranker = FeatureRanker("reg", group="fast", verbose=False)

    report = ranker.rank_resampled(
        X_DATA, Y_REG, scoring="r2", cv=2, n_repeats=1, random_state=0,
        feature_names=["a", "b", "c"],
    )

    assert len(report["splits"]) == 2
    for split in report["splits"]:
        assert set(split["report"]["models"]) == FAST_MODELS["reg"]


def test_selector_passes_fast_through():
    selector = ConsensusSelector(
        "reg", "r2", k=2, group="fast", cv=2, n_repeats=1, random_state=0,
        verbose=False,
    )

    selector.fit(X_DATA, Y_REG)

    for split in selector.report_["splits"]:
        assert set(split["report"]["models"]) == FAST_MODELS["reg"]
