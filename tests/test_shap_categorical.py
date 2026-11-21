"""Acceptance check for shap-categorical (lead-written; builder must not edit)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

shap = pytest.importorskip("shap")

from featranker import FeatureRanker  # noqa: E402
from featranker.evidence import _shap_values  # noqa: E402


def frame(n=500, levels=8, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, levels, size=n)
    X = pd.DataFrame({"x": rng.normal(size=n), "c": codes.astype(str), "z": rng.normal(size=n)})
    y = (codes % 3).astype(float) + 2 * X["x"].to_numpy() + 0.1 * rng.normal(size=n)
    return X, y


def ranked(**kwargs):
    X, y = frame()
    ranker = FeatureRanker("reg", "linear", verbose=False, categorical=["c"], evidence=True, **kwargs)
    ranker.fit(X.iloc[:350], y[:350])
    return ranker, ranker.rank_features(X.iloc[350:], y[350:], scoring="r2", n_repeats=2, random_state=0)


def test_builtin_categorical_linear_models_get_shap():
    _, report = ranked()
    ridge = report["models"]["ridge_regression"]
    assert ridge["shap"]["explainer"] == "linear"
    assert set(ridge["shap"]["groups"]) == {"x", "c", "z"}
    assert ridge["shap"]["groups"]["c"]["direction"] is None
    for entry in report["evidence"]["shap"]["unsupported"]:
        assert "ColumnTransformer" not in entry["reason"], entry


def test_shap_values_are_summed_back_exactly():
    ranker, _ = ranked()
    model = ranker.trained_models["ridge_regression"]
    X_codes = ranker.train_X_[:, : len(ranker.feature_names_)]
    rows, background = X_codes[:50], X_codes[50:150]

    name, values = _shap_values(model, rows, background, shap)

    assert name == "linear"
    assert values.shape == (1, 50, X_codes.shape[1])
    expected = model.predict(rows) - model.predict(background).mean()
    np.testing.assert_allclose(values[0].sum(axis=1), expected, rtol=1e-6, atol=1e-9)


def test_with_shadows_still_supported():
    _, report = ranked(shadows=True, shadow_seed=0)
    assert report["models"]["ridge_regression"]["shap"]["explainer"] == "linear"


def test_user_column_transformer_stays_unsupported():
    X, y = frame()
    model = make_pipeline(ColumnTransformer([("s", StandardScaler(), [0])], remainder="passthrough"), Ridge())
    numeric = X[["x", "z"]]
    ranker = FeatureRanker("reg", models={"m": model}, verbose=False, evidence=True)
    ranker.fit(numeric.iloc[:350], y[:350])
    report = ranker.rank_features(numeric.iloc[350:], y[350:], scoring="r2", n_repeats=2, random_state=0)
    reasons = {e["model"]: e["reason"] for e in report["evidence"]["shap"]["unsupported"]}
    assert "ColumnTransformer" in reasons.get("m", "")


def test_dataframe_model_input_is_supported():
    _, report = ranked(model_input="dataframe")
    assert report["models"]["ridge_regression"]["shap"]["explainer"] == "linear"
    for entry in report["evidence"]["shap"]["unsupported"]:
        assert "feature_names_in_" not in entry["reason"], entry


def test_user_one_hot_pipeline_with_non_column_remainder_is_not_folded():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(0, 4, 300), rng.normal(size=(300, 3))]).astype(float)
    y = X[:, 1] + rng.normal(size=300)
    model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [0])], remainder=PCA(2)),
        Ridge(),
    ).fit(X, y)
    with pytest.raises(ValueError):
        _shap_values(model, X[:20], X[20:60], shap)
