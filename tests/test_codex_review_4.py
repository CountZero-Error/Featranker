"""Acceptance check for codex-review-4 (lead-written; builder must not edit)."""
import copy
import json
import warnings
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model as lm
from sklearn.compose import TransformedTargetRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, NuSVC

from featranker import FeatureRanker
from featranker.importance import _resolve_scorer, _resolve_scorers

RNG = np.random.default_rng(0)
X = RNG.normal(size=(240, 3))
Y = X[:, 0] + 0.3 * X[:, 1] + 0.1 * RNG.normal(size=240)
NAMES = ["a", "b", "c"]


def ranked(task, model, y, X=X, **kwargs):
    ranker = FeatureRanker(task, models={"m": model}, evidence=True, verbose=False, **kwargs)
    ranker.fit(X[:160], y[:160], feature_names=NAMES)
    return ranker.rank_features(
        X[160:], y[160:], scoring="r2" if task == "reg" else "accuracy", n_repeats=2, random_state=0)


def is_supported(report):
    json.dumps(report, allow_nan=False)
    unsupported = [entry["model"] for entry in report["evidence"]["shap"]["unsupported"]]
    assert ("shap" in report["models"]["m"]) != ("m" in unsupported)
    return not unsupported


def reason(report):
    return report["evidence"]["shap"]["unsupported"][0]["reason"]


class ReversePipeline(Pipeline):
    def fit(self, X, y=None, **kwargs):
        return super().fit(np.asarray(X)[:, ::-1], y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(np.asarray(X)[:, ::-1], **kwargs)

    coef_ = property(lambda self: self.steps[-1][1].coef_)
    intercept_ = property(lambda self: self.steps[-1][1].intercept_)


class HarmlessPipeline(Pipeline):
    coef_ = property(lambda self: self.steps[-1][1].coef_)
    intercept_ = property(lambda self: self.steps[-1][1].intercept_)


class ReverseRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        return super().fit(np.asarray(X)[:, ::-1], y, sample_weight)

    def predict(self, X):
        return super().predict(np.asarray(X)[:, ::-1])


class ReverseLogistic(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        return super().fit(np.asarray(X)[:, ::-1], y, sample_weight)

    def decision_function(self, X):
        return super().decision_function(np.asarray(X)[:, ::-1])


def ridge_in(cls):
    return cls([("ridge", Ridge())])


# SHAP


@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
@pytest.mark.parametrize(
    "model", [ridge_in(ReversePipeline), ridge_in(HarmlessPipeline), ReverseRidge()],
    ids=["reverse-pipeline", "harmless-pipeline", "reverse-ridge"])
def test_column_changing_regressor_is_unsupported(shap_installed, wrap, model):
    assert not is_supported(ranked("reg", wrap(model), Y))


class ReversePoisson(lm.PoissonRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(np.asarray(X)[:, ::-1], y, sample_weight)

    def predict(self, X):
        return super().predict(np.asarray(X)[:, ::-1])


@pytest.mark.parametrize("offset", [1e4, 1e6, 1e8])
@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
def test_large_offset_does_not_hide_changed_columns(shap_installed, wrap, offset):
    assert not is_supported(ranked("reg", wrap(ReverseRidge()), Y + offset))
    assert is_supported(ranked("reg", wrap(Ridge()), Y + offset))


def test_large_offset_does_not_hide_changed_columns_behind_a_link(shap_installed):
    for offset in (1e6, 3e7, 1e9):
        assert not is_supported(ranked("reg", ReversePoisson(), np.exp(Y) + offset))
        assert is_supported(ranked("reg", lm.PoissonRegressor(), np.exp(Y) + offset))


@pytest.mark.parametrize("factor", [1.0, 1e3, 1e14])
@pytest.mark.parametrize("scaled", [False, True])
def test_output_scale_does_not_hide_changed_columns_behind_a_link(shap_installed, scaled, factor):
    for cls, expected in ((ReversePoisson, False), (lm.PoissonRegressor, True)):
        model = cls(alpha=0)
        model = make_pipeline(StandardScaler(), model) if scaled else model
        assert is_supported(ranked("reg", model, np.exp(Y) * factor)) is expected


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_column_changing_classifier_is_unsupported(shap_installed, scaled, n_classes):
    y = np.digitize(Y, [0.0] if n_classes == 2 else [-0.5, 0.5])
    model = make_pipeline(StandardScaler(), ReverseLogistic()) if scaled else ReverseLogistic()
    assert not is_supported(ranked("clf", model, y))


@pytest.mark.parametrize("offset", [0.0, 1e6])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("model", [
    Ridge(), lm.LinearRegression(), lm.Lasso(0.01), lm.Lasso(1e3), lm.ElasticNet(0.01),
    lm.BayesianRidge(), lm.ARDRegression(), lm.HuberRegressor(max_iter=1000), lm.Lars(),
    lm.OrthogonalMatchingPursuit(), lm.QuantileRegressor(alpha=0.0), lm.TheilSenRegressor(),
    lm.RidgeCV(), lm.LassoCV(), lm.SGDRegressor(), SVR(kernel="linear"),
], ids=lambda m: type(m).__name__)
def test_linear_regressors_stay_supported(shap_installed, model, scaled, offset):
    if type(model) is lm.SGDRegressor and offset:
        pytest.skip("diverges")
    assert is_supported(ranked(
        "reg", make_pipeline(StandardScaler(), model) if scaled else model, Y + offset))


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("model", [
    lm.PoissonRegressor(), lm.GammaRegressor(), lm.TweedieRegressor(power=1.5, link="log"),
    lm.TweedieRegressor(power=0),
], ids=str)
def test_link_regressors_stay_supported(shap_installed, model, scaled):
    model = make_pipeline(StandardScaler(), model) if scaled else model
    assert is_supported(ranked("reg", model, np.exp(Y)))


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
@pytest.mark.parametrize("model", [
    LogisticRegression(), lm.RidgeClassifier(), lm.SGDClassifier(random_state=0),
    lm.Perceptron(), LinearSVC(), LinearDiscriminantAnalysis(),
], ids=lambda m: type(m).__name__)
def test_linear_classifiers_stay_supported(shap_installed, model, n_classes, scaled):
    y = np.digitize(Y, {2: [0.0], 3: [-0.5, 0.5], 4: [-0.6, 0.0, 0.6]}[n_classes])
    model = make_pipeline(StandardScaler(), model) if scaled else model
    assert is_supported(ranked("clf", model, y))


def test_float32_and_named_input_stay_supported(shap_installed):
    assert is_supported(ranked("reg", Ridge(), (Y + 1e4).astype(np.float32), X.astype(np.float32)))
    assert is_supported(ranked("reg", Ridge(), Y, model_input="dataframe"))


def test_supported_values_are_the_closed_form(shap_installed):
    report = ranked("reg", Ridge(), Y)
    model = Ridge().fit(X[:160], Y[:160])
    expected = np.abs((X[160:] - X[:160].mean(axis=0)) * model.coef_).mean(axis=0)
    got = [report["models"]["m"]["shap"]["groups"][n]["mean_abs"] for n in NAMES]
    assert got == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("shape", ["ovr", "ovo"])
@pytest.mark.parametrize("n_classes", [3, 4])
@pytest.mark.parametrize("cls", [SVC, NuSVC])
def test_multiclass_svc_is_unsupported(shap_installed, cls, n_classes, shape):
    y = np.digitize(Y, [-0.5, 0.5] if n_classes == 3 else [-0.6, 0.0, 0.6])
    kwargs = {"nu": 0.1} if cls is NuSVC else {}
    model = cls(kernel="linear", decision_function_shape=shape, **kwargs)
    report = ranked("clf", make_pipeline(StandardScaler(), model), y)
    assert not is_supported(report)
    assert "one-vs-one" in reason(report)


def test_binary_svc_stays_supported(shap_installed):
    for model in (SVC(kernel="linear"), make_pipeline(StandardScaler(), SVC(kernel="linear"))):
        assert is_supported(ranked("clf", model, (Y > 0).astype(int)))


@pytest.mark.parametrize("task", ["reg", "clf"])
def test_builtin_categorical_linear_models_stay_supported(shap_installed, task):
    frame = pd.DataFrame({"c": (np.arange(240) % 4).astype(str), "x": X[:, 0], "z": X[:, 1]})
    y = Y if task == "reg" else np.digitize(Y, [-0.5, 0.5])
    ranker = FeatureRanker(task, "linear", evidence=True, verbose=False, categorical=["c"])
    ranker.fit(frame[:160], y[:160])
    report = ranker.rank_features(frame[160:], y[160:], n_repeats=2, random_state=0,
                                  scoring="r2" if task == "reg" else "accuracy")
    fitted = {n for n, m in report["models"].items()}
    unsupported = {e["model"] for e in report["evidence"]["shap"]["unsupported"]}
    assert len(fitted - unsupported) >= (7 if task == "reg" else 3)
    assert not [e for e in report["evidence"]["shap"]["unsupported"] if "reproduce" in e["reason"]]


@pytest.fixture
def shap_installed():
    return pytest.importorskip("shap")


# ablate() and changed components


def forest(cls):
    return cls(n_estimators=5, bootstrap=False, random_state=0)


def simple(models, y=Y):
    return FeatureRanker("reg", models=models, verbose=False).fit(
        X[:160], y[:160], feature_names=NAMES)


def report_of(ranker):
    return ranker.rank_features(X[160:], Y[160:], scoring="r2", n_repeats=2, random_state=0)


def ablated(ranker, report):
    return ranker.ablate(X[:160], Y[:160], X[160:], Y[160:], report, ks=[3])


def test_components_record_nested_classes():
    entry = report_of(simple({"m": make_pipeline(StandardScaler(), Ridge()), "r": Ridge()}))["models"]
    assert entry["m"]["components"] == {"standardscaler": "StandardScaler", "ridge": "Ridge"}
    assert entry["r"]["components"] == {}
    assert entry["m"]["params"]["ridge__alpha"] == 1.0
    assert not {"steps", "ridge", "standardscaler"} & set(entry["m"]["params"])


def test_changed_component_is_refused():
    pipe = Pipeline([("forest", forest(ExtraTreesRegressor))])
    ranker = simple({"m": pipe})
    report = report_of(ranker)
    ablated(ranker, report)
    pipe.steps[0] = ("forest", forest(RandomForestRegressor))
    ranker.fit(X[:160], Y[:160], feature_names=NAMES)
    with pytest.raises(ValueError, match="'m'"):
        ablated(ranker, report)


SWAPS = {
    "forest": (lambda: Pipeline([("step", StandardScaler()), ("ridge", Ridge())]),
               "ridge", forest(RandomForestRegressor)),
    "passthrough": (lambda: Pipeline([("step", StandardScaler()), ("ridge", Ridge())]),
                    "step", "passthrough"),
    "none": (lambda: Pipeline([("step", StandardScaler()), ("ridge", Ridge())]), "step", None),
    "stacking": (lambda: StackingRegressor([("r", Ridge())], final_estimator=Ridge()),
                 "final_estimator", None),
    "bagging": (lambda: BaggingRegressor(Ridge(), random_state=0), "estimator", None),
    "target": (lambda: TransformedTargetRegressor(Ridge(), transformer=StandardScaler()),
               "transformer", None),
}


@pytest.mark.parametrize("version", [None, "1.4.0"])
@pytest.mark.parametrize("make,key,value", SWAPS.values(), ids=SWAPS)
def test_changed_component_is_refused_on_any_sklearn_version(make, key, value, version):
    model = make()
    ranker = simple({"m": model})
    report = copy.deepcopy(report_of(ranker))
    if version:
        report["provenance"]["scikit-learn"] = version
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ablated(ranker, report)
    model.set_params(**{key: value})
    ranker.fit(X[:160], Y[:160], feature_names=NAMES)
    with pytest.raises(ValueError, match="'m'"):
        ablated(ranker, report)


def test_old_report_without_components_still_ablates():
    ranker = simple({"m": make_pipeline(StandardScaler(), Ridge())})
    report = copy.deepcopy(report_of(ranker))
    del report["schema_version"], report["models"]["m"]["components"]
    with pytest.warns(UserWarning, match="report.schema.json"):
        assert ablated(ranker, report)["ablation"]["curve"]


class Duck:
    """No get_params."""

    def fit(self, X, y):
        self.model_ = Ridge().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class OddParam(Ridge):
    def __init__(self, alpha=1.0, pairs=None):
        super().__init__(alpha=alpha)
        self.pairs = pairs


@pytest.mark.parametrize("change", ["none", "version", "schema"])
@pytest.mark.parametrize("make", [Duck, lambda: OddParam(pairs=[()])], ids=["duck", "odd"])
def test_unchanged_unusual_models_still_ablate(make, change):
    ranker = simple({"m": make()})
    report = copy.deepcopy(report_of(ranker))
    if change == "version":
        report["provenance"]["scikit-learn"] = "1.4.0"
    if change == "schema":
        del report["schema_version"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ablated(ranker, report)["ablation"]["curve"]


def test_components_are_required_by_the_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((files("featranker") / "report.schema.json").read_text(encoding="utf-8"))
    report = json.loads(json.dumps(report_of(simple({"m": make_pipeline(StandardScaler(), Ridge())}))))
    jsonschema.validate(report, schema)
    del report["models"]["m"]["components"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, schema)


class FailsOnDemand(Ridge):
    failing = False

    def fit(self, X, y, sample_weight=None):
        if FailsOnDemand.failing:
            raise RuntimeError("on demand")
        return super().fit(X, y, sample_weight)


def test_voter_missing_from_latest_fit_is_refused(monkeypatch):
    ranker = simple({"m": FailsOnDemand(), "other": Ridge()})
    report = report_of(ranker)
    ablated(ranker, report)
    monkeypatch.setattr(FailsOnDemand, "failing", True)
    ranker.fit(X[:160], Y[:160], feature_names=NAMES)
    with pytest.raises(ValueError, match="'m'"):
        ablated(ranker, report)


# scorer names


class WrappedScorer:
    __name__ = None

    def __init__(self):
        self._score_func = r2_score

    def __call__(self, estimator, X, y):
        return 2 * r2_score(y, estimator.predict(X))


def test_custom_scorer_is_named_by_its_class():
    assert _resolve_scorer(WrappedScorer())[1] == "WrappedScorer"
    names = [name for _, name in _resolve_scorers([WrappedScorer(), make_scorer(r2_score)])]
    assert names == ["WrappedScorer", "r2_score"]


def test_function_attribute_does_not_rename_it():
    def double(estimator, X, y):
        return 0.0

    double._score_func = r2_score
    assert _resolve_scorer(double)[1] == "double"
