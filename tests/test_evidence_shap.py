import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from featranker import FeatureRanker
from featranker.evidence import SHAP_ROWS, evenly_spaced_rows

pytest.importorskip("shap", exc_type=ImportError)

NAMES = ["a", "b", "c"]
RNG = np.random.default_rng(0)
X_DATA = RNG.normal(size=(180, 3))
SCORE = 3.0 * X_DATA[:, 0] + 0.5 * X_DATA[:, 1] + 0.2 * RNG.normal(size=180)
N_TRAIN = 120


def regressors():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "forest": RandomForestRegressor(n_estimators=15, random_state=0),
        "knn": KNeighborsRegressor(),
    }


def run(task, models, y, X=X_DATA, **kwargs):
    r = FeatureRanker(task=task, models=models, verbose=False, evidence=True,
                      shadows=kwargs.pop("shadows", False), shadow_seed=0)
    r.fit(X[:N_TRAIN], y[:N_TRAIN], NAMES)
    kwargs = {"scoring": "r2" if task == "reg" else "accuracy",
              "n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_features(X[N_TRAIN:], y[N_TRAIN:], **kwargs)


@pytest.fixture(scope="module")
def regression_report():
    return run("reg", regressors(), SCORE)


def test_linear_and_tree_models_have_shap_for_every_group(regression_report):
    models = regression_report["models"]

    assert models["ridge"]["shap"]["explainer"] == "linear"
    assert models["forest"]["shap"]["explainer"] == "tree"
    for name in ("ridge", "forest"):
        groups = models[name]["shap"]["groups"]
        assert set(groups) == set(NAMES)
        assert all(g["mean_abs"] > 0 for g in groups.values())
    block = regression_report["evidence"]["shap"]
    assert block["available"] is True
    assert block["n_rows"] == len(X_DATA) - N_TRAIN


def test_planted_positive_effect_has_positive_direction(regression_report):
    groups = regression_report["models"]["ridge"]["shap"]["groups"]

    assert groups["a"]["direction"] > 0.9
    assert groups["a"]["mean_abs"] > groups["c"]["mean_abs"]


def test_unsupported_model_is_listed_and_does_not_fail_the_ranking(regression_report):
    unsupported = regression_report["evidence"]["shap"]["unsupported"]

    assert "shap" not in regression_report["models"]["knn"]
    assert [u["model"] for u in unsupported] == ["knn"]
    assert unsupported[0]["reason"].startswith("InvalidModelError")
    assert len(unsupported[0]["reason"]) <= len("InvalidModelError: ") + 200
    assert regression_report["failures"] == []


def test_multi_column_group_sums_columns_and_has_null_direction():
    report = run("reg", regressors(), SCORE, feature_groups={"ab": ["a", "b"]})

    groups = report["models"]["ridge"]["shap"]["groups"]
    assert set(groups) == {"ab", "c"}
    assert groups["ab"]["direction"] is None
    assert isinstance(groups["c"]["direction"], float)
    # sum of columns: at least as large as the planted single column alone
    single = run("reg", regressors(), SCORE)["models"]["ridge"]["shap"]["groups"]
    assert groups["ab"]["mean_abs"] >= single["a"]["mean_abs"] * 0.9


def test_constant_feature_has_null_direction():
    X = X_DATA.copy()
    X[:, 2] = 1.0

    report = run("reg", {"ridge": regressors()["ridge"]}, SCORE, X=X)

    assert report["models"]["ridge"]["shap"]["groups"]["c"]["direction"] is None


def test_binary_classification_uses_positive_class():
    y = (SCORE > 0).astype(int)
    models = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression()),
        "forest": RandomForestClassifier(n_estimators=15, random_state=0),
        "knn": KNeighborsClassifier(),
    }

    report = run("clf", models, y)

    assert report["models"]["logit"]["shap"]["groups"]["a"]["direction"] > 0.9
    assert report["models"]["forest"]["shap"]["groups"]["a"]["direction"] > 0.5
    assert "shap" not in report["models"]["knn"]
    json.dumps(report, allow_nan=False)


def test_multiclass_averages_mean_abs_and_has_null_direction():
    y = np.digitize(SCORE, [-2.0, 2.0])
    models = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression()),
        "forest": RandomForestClassifier(n_estimators=15, random_state=0),
        "boosting": GradientBoostingClassifier(n_estimators=5, random_state=0),
    }

    report = run("clf", models, y)

    for name in ("logit", "forest"):
        groups = report["models"][name]["shap"]["groups"]
        assert set(groups) == set(NAMES)
        assert all(g["direction"] is None and g["mean_abs"] > 0 for g in groups.values())
    # whether or not shap can explain multiclass boosting, it is never fatal
    unsupported = [u["model"] for u in report["evidence"]["shap"]["unsupported"]]
    assert ("shap" in report["models"]["boosting"]) != ("boosting" in unsupported)
    assert "boosting" in report["models"]


def test_row_sample_is_capped_and_evenly_spaced():
    rows = evenly_spaced_rows(2500)

    assert len(rows) == SHAP_ROWS == 1000
    assert rows[0] == 0 and rows[-1] == 2499
    assert np.ptp(np.diff(rows)) <= 1
    assert list(evenly_spaced_rows(7)) == list(range(7))


def test_report_caps_explained_rows_at_1000():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1300, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=1300)
    r = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False, evidence=True)
    r.fit(X[:100], y[:100], NAMES)

    report = r.rank_features(X[100:], y[100:], scoring="r2", n_repeats=1, random_state=0)

    assert report["evidence"]["shap"]["n_rows"] == 1000


def test_shadow_columns_are_not_reported():
    report = run("reg", regressors(), SCORE, shadows=True)

    for name in ("ridge", "forest"):
        assert set(report["models"][name]["shap"]["groups"]) == set(NAMES)


def test_shap_does_not_depend_on_random_state():
    first = run("reg", regressors(), SCORE, random_state=0)
    second = run("reg", regressors(), SCORE, random_state=7)

    assert first["evidence"] == second["evidence"]
    for name in ("ridge", "forest"):
        assert first["models"][name]["shap"] == second["models"][name]["shap"]


def test_full_evidence_report_is_strict_json(regression_report):
    json.dumps(regression_report, allow_nan=False)


def test_pipeline_with_a_non_scaler_step_is_unsupported():
    models = {"pca_ridge": make_pipeline(PCA(n_components=3), Ridge())}

    report = run("reg", models, SCORE)

    assert "shap" not in report["models"]["pca_ridge"]
    (entry,) = report["evidence"]["shap"]["unsupported"]
    assert entry["model"] == "pca_ridge"
    assert "PCA" in entry["reason"] and "per-column scaler" in entry["reason"]
    assert report["failures"] == []


@pytest.mark.parametrize("step", ["passthrough", None])
def test_passthrough_pipeline_step_is_the_identity(step):
    models = {"ridge": Pipeline([("skip", step), ("ridge", Ridge())])}

    report = run("reg", models, SCORE)

    assert report["models"]["ridge"]["shap"]["explainer"] == "linear"
    assert report["models"]["ridge"]["shap"]["groups"]["a"]["direction"] > 0.9
    assert report["evidence"]["shap"]["unsupported"] == []


def test_one_step_pipeline_is_the_bare_estimator():
    report = run("reg", {"ridge": make_pipeline(Ridge())}, SCORE)

    assert report["models"]["ridge"]["shap"]["explainer"] == "linear"
    assert report["models"]["ridge"]["shap"]["groups"]["a"]["direction"] > 0.9


def test_scaled_tree_pipeline_is_explained_and_errors_name_the_real_class():
    models = {
        "scaled_forest": make_pipeline(
            MinMaxScaler(), RandomForestRegressor(n_estimators=15, random_state=0)
        ),
        "scaled_knn": make_pipeline(StandardScaler(), KNeighborsRegressor()),
    }

    report = run("reg", models, SCORE)

    assert report["models"]["scaled_forest"]["shap"]["explainer"] == "tree"
    assert set(report["models"]["scaled_forest"]["shap"]["groups"]) == set(NAMES)
    (entry,) = report["evidence"]["shap"]["unsupported"]
    assert entry["model"] == "scaled_knn"
    assert "KNeighborsRegressor" in entry["reason"]


def test_linear_shap_is_coefficient_times_distance_from_the_training_mean():
    rng = np.random.default_rng(3)
    X = rng.normal(loc=[5.0, -2.0, 1.0], scale=[2.0, 0.5, 3.0], size=(1500, 3))
    y = X @ [1.0, 2.0, 0.0] + rng.normal(size=1500)
    n_train = 1300  # more than the 1000-row cap, so the background is subsampled
    r = FeatureRanker("reg", models={"ridge": make_pipeline(StandardScaler(), Ridge())},
                      verbose=False, evidence=True)
    r.fit(X[:n_train], y[:n_train], NAMES)

    report = r.rank_features(X[n_train:], y[n_train:], scoring="r2", n_repeats=1,
                             random_state=0)

    pipeline = r.trained_models["ridge"]
    scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1]
    coef = ridge.coef_ / scaler.scale_  # in the original feature space
    background = X[:n_train][evenly_spaced_rows(n_train)]
    expected = np.abs(coef * (X[n_train:] - background.mean(axis=0))).mean(axis=0)
    groups = report["models"]["ridge"]["shap"]["groups"]
    for index, name in enumerate(NAMES):
        assert groups[name]["mean_abs"] == pytest.approx(expected[index], abs=1e-9)
