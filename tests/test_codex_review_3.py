"""Acceptance check for codex-review-3 (lead-written; builder must not edit)."""
import json
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker
from featranker.ablation import _median

ROOT = Path(__file__).resolve().parents[1]
X = np.random.default_rng(0).normal(size=(150, 3))
Y = X[:, 0] + X[:, 1]
NAMES = ["a", "b", "c"]


def fitted(models, **kwargs):
    return FeatureRanker("reg", models=models, verbose=False, **kwargs).fit(
        X[:100], Y[:100], feature_names=NAMES)


def ranked(ranker, scoring="r2"):
    return ranker.rank_features(X[100:], Y[100:], scoring=scoring, n_repeats=2, random_state=0)


def ablated(ranker, report, **kwargs):
    return ranker.ablate(X[:100], Y[:100], X[100:], Y[100:], report, **{"ks": [3], **kwargs})


# ablate refits what fit() used


@pytest.mark.parametrize("nested", [False, True])
def test_changed_user_estimator_does_not_change_ablation(nested):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)) if nested else Ridge(alpha=1)
    ranker = fitted({"m": model})
    report = ranked(ranker)
    before = ablated(ranker, report)["ablation"]["curve"]
    model.set_params(**{"ridge__alpha" if nested else "alpha": 1e9})
    assert ablated(ranker, report)["ablation"]["curve"] == before


def test_changed_builtin_template_does_not_change_ablation():
    frame = pd.DataFrame({"c": (np.arange(150) % 4).astype(str), "x": X[:, 0], "z": X[:, 1]})
    ranker = FeatureRanker("reg", "linear", verbose=False, categorical=["c"]).fit(frame[:100], Y[:100])
    report = ranker.rank_features(frame[100:], Y[100:], scoring="r2", n_repeats=2, random_state=0)
    args = (frame[:100], Y[:100], frame[100:], Y[100:], report)
    before = ranker.ablate(*args, ks=[3])["ablation"]
    for model in ranker.models["reg"]["linear"].values():
        for key in model.get_params(deep=True):
            if key.endswith("__alpha"):
                model.set_params(**{key: 1e9})
    after = ranker.ablate(*args, ks=[3])["ablation"]
    score = "ridge_regression"
    assert score in before["models"]
    assert after["curve"][0]["scores"][score] == before["curve"][0]["scores"][score]


def test_refit_follows_the_latest_fit():
    model = Ridge(alpha=1e9)
    ranker = fitted({"m": model})
    model.set_params(alpha=1)
    ranker.fit(X[:100], Y[:100], feature_names=NAMES)
    out = ablated(ranker, ranked(ranker))
    assert out["ablation"]["curve"][0]["median_score"] > 0.99


# schema_version, not provenance, decides the old-report warning


def schema_warnings(ranker, report):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ablated(ranker, report)
    assert out["ablation"]["curve"]
    return [w for w in caught if "report.schema.json" in str(w.message)]


def edited(report, version, keep_stamp):
    out = json.loads(json.dumps(report))
    out["provenance"]["featranker"] = version
    if not keep_stamp:
        del out["schema_version"]
    return out


@pytest.mark.parametrize("version", ["1.21.2", "2.0.0", None])
def test_stamp_decides(version):
    ranker = fitted({"m": Ridge()})
    report = ranked(ranker)
    assert report["schema_version"] == 2
    assert not schema_warnings(ranker, edited(report, version, True))
    hits = schema_warnings(ranker, edited(report, version, False))
    assert hits and all(issubclass(w.category, UserWarning) for w in hits)


def test_stamp_in_resampled_and_splits_and_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((files("featranker") / "report.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    report = FeatureRanker("reg", models={"m": Ridge()}, verbose=False).rank_resampled(
        X, Y, scoring="r2", cv=3, n_repeats=2, random_state=0, feature_names=NAMES)
    assert report["schema_version"] == 2
    ranker = fitted({"m": Ridge()})
    ablation = ablated(ranker, ranked(ranker))
    for good in (report, ablation):
        plain = json.loads(json.dumps(good, allow_nan=False))
        validator.validate(plain)
        del plain["schema_version"]
        assert not validator.is_valid(plain)


# Medians keep both extremes


def test_median_extremes():
    assert _median({"a": 5e-324, "b": 5e-324}) == 5e-324
    assert _median({"a": 1e308, "b": 1e308}) == 1e308
    assert _median({"a": -1.7e308, "b": -1.7e308}) == -1.7e308
    assert _median({"a": 1.0, "b": 2.0}) == 1.5


def test_tiny_scores_through_ablate():
    def tiny(est, X_, y_):
        return 0.0 if isinstance(est, DummyRegressor) else 5e-324

    ranker = fitted({"r": Ridge(), "l": Ridge(alpha=2)})
    row = ablated(ranker, ranked(ranker, tiny), ks=[1], scoring=tiny)["ablation"]["curve"][0]
    assert row["median_score"] == row["paired_median_score"] == 5e-324


# SHAP


class ReverseScaler(StandardScaler):
    def transform(self, X, copy=None):
        return super().transform(X)[:, ::-1]


class MyPipeline(Pipeline):
    pass


class ReversingPipeline(Pipeline):
    def fit(self, X, y=None, **params):
        return super().fit(np.asarray(X)[:, ::-1], y, **params)

    def predict(self, X, **params):
        return super().predict(np.asarray(X)[:, ::-1], **params)


def builtin_encoder():
    frame = pd.DataFrame({"c": (np.arange(100) % 4).astype(str), "x": X[:100, 0]})
    ranker = FeatureRanker("reg", "linear", verbose=False, categorical=["c"]).fit(frame, Y[:100])
    return clone(ranker.trained_models["ridge_regression"][0].transformers[0][1])


def coded():
    rng = np.random.default_rng(2)
    data = np.column_stack([rng.integers(4, size=150), rng.normal(size=(150, 2))]).astype(float)
    return data, data[:, 1]


def shap_report(model):
    pytest.importorskip("shap")
    data, y = coded()
    ranker = FeatureRanker("reg", models={"m": model}, verbose=False, evidence=True)
    ranker.fit(data[:100], y[:100], feature_names=["code", "signal", "noise"])
    if isinstance(getattr(model, "steps", [("", None)])[0][1], ColumnTransformer):
        ranker.user_models = None  # only a built-in model's one-hot step is folded (codex-review-9)
    return ranker.rank_features(data[100:], y[100:], scoring="r2", n_repeats=2, random_state=0)


def fold_model(remainder):
    return make_pipeline(
        ColumnTransformer([("cat", builtin_encoder(), [0])], remainder=remainder), Ridge())


@pytest.mark.parametrize("build", [
    lambda: fold_model(ReverseScaler()),
    lambda: fold_model(make_pipeline(ReverseScaler())),
    lambda: fold_model(MyPipeline([("s", StandardScaler())])),
    lambda: make_pipeline(ReverseScaler(), Ridge()),
    lambda: ReversingPipeline([("s", StandardScaler()), ("r", Ridge())]),
    lambda: ReversingPipeline([("r", Ridge())]),
])
def test_reordering_steps_unsupported(build):
    report = shap_report(build())
    assert [e["model"] for e in report["evidence"]["shap"]["unsupported"]] == ["m"]
    assert "shap" not in report["models"]["m"]


@pytest.mark.parametrize("build", [
    lambda: fold_model(StandardScaler()),
    lambda: fold_model("passthrough"),
    lambda: make_pipeline(StandardScaler(), Ridge()),
])
def test_known_steps_put_signal_first(build):
    report = shap_report(build())
    assert report["evidence"]["shap"]["unsupported"] == []
    groups = report["models"]["m"]["shap"]["groups"]
    assert groups["signal"]["mean_abs"] > 10 * groups["noise"]["mean_abs"]


def test_builtin_categorical_pipelines_stay_supported():
    pytest.importorskip("shap")
    frame = pd.DataFrame({"c": (np.arange(150) % 4).astype(str), "x": X[:, 0], "z": X[:, 1]})
    ranker = FeatureRanker("reg", "linear", verbose=False, categorical=["c"], evidence=True)
    report = ranker.fit(frame[:100], Y[:100]).rank_features(
        frame[100:], Y[100:], scoring="r2", n_repeats=2, random_state=0)
    reasons = {e["model"]: e["reason"] for e in report["evidence"]["shap"]["unsupported"]}
    assert "ridge_regression" not in reasons
    assert "shap" in report["models"]["ridge_regression"]


def huge_report(columns):
    pytest.importorskip("shap")
    base = np.tile([-1.0, 1.0], 25)[:, None]
    train, evaluation = np.repeat(base[:12], columns, axis=1), 5 * np.repeat(base, columns, axis=1)
    names = [f"x{i}" for i in range(columns)]
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False, evidence=True)
    ranker.fit(train, 1e307 * train[:, 0], feature_names=names)
    return ranker.rank_features(
        evaluation, 1e307 * evaluation[:, 0], n_repeats=2, random_state=0,
        scoring=lambda est, X_, y_: r2_score(y_ / 1e307, est.predict(X_) / 1e307))


def test_huge_contributions_have_finite_mean():
    report = huge_report(1)
    json.dumps(report, allow_nan=False)
    assert 4e307 < report["models"]["m"]["shap"]["groups"]["x0"]["mean_abs"] < 5.1e307


def test_group_overflow_raises():
    from featranker.evidence import _group_summary

    stacked = np.full((1, 4, 2), 1.5e308)
    with pytest.raises(ValueError):
        _group_summary(stacked, np.ones((4, 2)), {"g": [0, 1]})
    multi = np.full((3, 4, 1), 1.5e308)
    assert _group_summary(multi, np.ones((4, 1)), {"g": [0]})["g"]["mean_abs"] == pytest.approx(1.5e308)


# Docs


def test_docs_match_artifacts():
    def text(path):
        return (ROOT / path).read_text(encoding="utf-8")

    assert "10 of the 80 regression null runs" in text("docs/calibration.md")
    assert "were not slower" not in text("docs/benchmark.md")
    assert "20:37:58" in text("benchmarks/realdata/DEVIATIONS.md")
    assert "schema_version" in text("docs/api.md") and "schema_version" in text("docs/report.md")
