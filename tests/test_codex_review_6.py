"""Acceptance check for codex-review-6 (lead-written; builder must not edit)."""
import enum
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn import linear_model as lm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVC

from featranker import FeatureRanker

from test_codex_review_5 import ReverseRidge, ranked, shap_installed, supported  # noqa: F401

RNG = np.random.default_rng(0)
X = RNG.normal(size=(300, 3))
Z = X[:, 0] + 0.3 * X[:, 1] + 0.1 * RNG.normal(size=300)


class SameRidge(Ridge):
    """Harmless, still not a library class."""


class AuxiliaryOutputs(LinearRegression):
    def fit(self, X, y):
        return super().fit(X, np.c_[10 * X[:, 1], 20 * X[:, 1], y])

    def predict(self, X):
        return super().predict(X)[:, -1]


class Duck(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.inner_ = LogisticRegression().fit(X, y)
        self.classes_, self.coef_, self.intercept_ = (
            self.inner_.classes_, self.inner_.coef_, self.inner_.intercept_)
        return self

    def predict(self, X):
        return self.inner_.predict(X)

    def predict_proba(self, X):
        return self.inner_.predict_proba(X)


def run(model, y, task="reg"):
    return ranked(model, X[:200], y[:200], X[200:], y[200:], task)


WRAPS = [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)]


@pytest.mark.parametrize("wrap", WRAPS)
def test_custom_linear_estimators_are_unsupported(shap_installed, wrap):
    for model in (SameRidge(), ReverseRidge(), AuxiliaryOutputs()):
        assert not supported(run(wrap(model), Z))
    assert not supported(run(wrap(Duck()), (Z > 0).astype(int), "clf"))


def test_reported_cases(shap_installed):
    u, v = np.random.default_rng(0).uniform(-2, 2, 130), np.linspace(-1.8, 1.8, 43)
    both = lambda a: np.c_[np.exp(a), a]  # noqa: E731
    model = ReverseRidge(alpha=0, fit_intercept=False)
    assert not supported(ranked(model, both(u), np.exp(u), both(v), np.exp(v)))
    a = np.linspace(-3, 4, 41)
    assert not supported(ranked(ReverseRidge(), np.c_[a, a + 10], a, np.c_[a, a - 25], a))


def test_wrapper_around_a_custom_class_is_unsupported(shap_installed):
    frozen = pytest.importorskip("sklearn.frozen")
    inner = ReverseRidge().fit(X[:200], Z[:200])
    for wrap in WRAPS:
        assert not supported(run(wrap(frozen.FrozenEstimator(inner)), Z))


@pytest.mark.parametrize("wrap", WRAPS)
def test_library_models_have_closed_form_values(shap_installed, wrap):
    cases = [(m, Z, "reg") for m in (
        Ridge(), LinearRegression(), lm.Lasso(0.01), lm.HuberRegressor(), SVR(kernel="linear"))]
    cases += [(m, np.exp(Z), "reg") for m in (
        lm.PoissonRegressor(), lm.GammaRegressor(), lm.TweedieRegressor(power=1.5, link="log"))]
    cases += [(m, (Z > 0).astype(int), "clf") for m in (
        LogisticRegression(), LinearDiscriminantAnalysis())]
    for model, y, task in cases:
        report = run(wrap(model), y, task)
        assert supported(report), model
        fitted = wrap(model).fit(X[:200], y[:200])
        steps = fitted if isinstance(fitted, Pipeline) else [None, fitted]
        coef = np.ravel(steps[-1].coef_)
        data = fitted[:-1].transform(X) if isinstance(fitted, Pipeline) else X
        expected = np.abs((data[200:] - data[:200].mean(axis=0)) * coef).mean(axis=0)
        got = [report["models"]["m"]["shap"]["groups"][n]["mean_abs"] for n in "abc"]
        assert got == pytest.approx(expected, rel=1e-4), model


@pytest.mark.parametrize("wrap", WRAPS)
def test_gblinear(shap_installed, wrap):
    xgboost = pytest.importorskip("xgboost")
    XGBC = lambda **k: xgboost.XGBClassifier(booster="gblinear", **k)  # noqa: E731
    assert supported(run(wrap(xgboost.XGBRegressor(booster="gblinear")), Z))
    for objective in ("binary:logistic", "binary:hinge"):
        y = (Z > 0).astype(int)
        report = ranked(wrap(XGBC(objective=objective)), X[:200], y[:200], X[200:], y[200:],
                        "clf", scoring="accuracy")
        assert supported(report)
    three = np.digitize(Z, [-0.5, 0.5])
    assert not supported(run(wrap(XGBC()), three, "clf"))


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("missing,expected", [
    (-999, False), (-999.0, False), (0.0, False), (np.float32(-1), False),
    (np.nan, True), (float("nan"), True), (np.float32("nan"), True)])
def test_gblinear_with_a_missing_sentinel_is_unsupported(shap_installed, wrap, missing, expected):
    xgboost = pytest.importorskip("xgboost")
    data = X.copy()
    if not expected:
        data[::5, 0] = float(missing)
    model = xgboost.XGBRegressor(booster="gblinear", missing=missing)
    assert supported(ranked(wrap(model), data[:200], Z[:200], data[200:], Z[200:])) is expected


@pytest.mark.parametrize("wrap", WRAPS)
def test_lda_with_a_covariance_estimator_stays_supported(shap_installed, wrap):
    from sklearn.covariance import LedoitWolf
    model = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=LedoitWolf())
    assert supported(run(wrap(model), (Z > 0).astype(int), "clf"))


def test_offsets_and_redundant_columns_are_supported(shap_installed):
    for model in (Ridge(), SVR(kernel="linear", max_iter=2000), LinearRegression()):
        wide = np.c_[X[:, :2] + 1e7, X[:, 0] + 1e7]
        assert supported(ranked(model, wide[:200], Z[:200], wide[200:], Z[200:]))


def test_multiclass_linear_svc_stays_supported(shap_installed):
    y = np.digitize(Z, [-0.5, 0.5])
    assert supported(ranked(LinearSVC(), X[:200], y[:200], X[200:], y[200:], "clf", scoring="accuracy"))


# ablate()


W = np.random.default_rng(42).normal(size=(250, 6))
SPLIT = (W[:180], W[:180, 0], W[180:], W[180:, 0])
NAMES = list("abcdef")


def fit(model):
    return FeatureRanker("reg", models={"m": model}, verbose=False).fit(
        SPLIT[0], SPLIT[1], feature_names=NAMES)


def report_of(model):
    return fit(model).rank_features(*SPLIT[2:], scoring="r2", n_repeats=2, random_state=0)


def ablate(model, report):
    return fit(model).ablate(*SPLIT, report, ks=[6])


def forest(**kwargs):
    return RandomForestRegressor(**{"n_estimators": 10, "random_state": 0, **kwargs})


@pytest.mark.parametrize("version", [None, "1.4.0"])
@pytest.mark.parametrize("old,new", [(1, 1.0), (1.0, 1), (np.int64(1), np.float64(1.0))])
def test_integer_and_fraction_differ(old, new, version):
    report = json.loads(json.dumps(report_of(forest(max_features=old))))
    if version:
        report["provenance"]["scikit-learn"] = version
        with pytest.warns(UserWarning, match="'m'"):
            ablate(forest(max_features=new), report)
        return
    with pytest.raises(ValueError, match="'m'"):
        ablate(forest(max_features=new), report)


class Size(enum.IntEnum):
    ONE = 1


class Name(str, enum.Enum):
    SQRT = "sqrt"


class WithExtra(Ridge):
    def __init__(self, alpha=1.0, extra=None):
        super().__init__(alpha=alpha)
        self.extra = extra


@pytest.mark.parametrize("round_trip", [False, True])
@pytest.mark.parametrize("value", [
    1, 1.0, True, None, "x", np.int64(1), np.float32(0.5), np.float64(0.5), np.bool_(True),
    np.str_("x"), Size.ONE, Name.SQRT, np.longdouble(0.1), np.int8(-3), [1], -0.0, 10 ** 30,
], ids=repr)
def test_unchanged_parameters_still_ablate(value, round_trip):
    report = report_of(WithExtra(extra=value))
    report = json.loads(json.dumps(report, allow_nan=False)) if round_trip else report
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert ablate(WithExtra(extra=value), report)["ablation"]["curve"]


@pytest.mark.parametrize("old,new", [(True, 1), (1, True), (0, False), (None, 0), ("1", 1)])
def test_scalar_types_differ(old, new):
    with pytest.raises(ValueError, match="'m'"):
        ablate(WithExtra(extra=new), report_of(WithExtra(extra=old)))


def legacy(report, dropped=()):
    report = json.loads(json.dumps(report))
    del report["schema_version"], report["models"]["m"]["components"]
    for key in dropped:
        del report["models"]["m"]["params"][key]
    return report


@pytest.mark.parametrize("value", [np.int64(10), np.int32(10)])
def test_pre_2_report_with_a_numpy_parameter_still_ablates(value):
    report = legacy(report_of(forest(n_estimators=value)), ["n_estimators"])
    with pytest.warns(UserWarning, match="report.schema.json"):
        assert ablate(forest(n_estimators=value), report)["ablation"]["curve"]


def test_pre_2_report_keeps_float64_and_is_still_checked():
    report = legacy(report_of(Ridge(alpha=np.float64(0.5))))
    with pytest.warns(UserWarning, match="report.schema.json"):
        assert ablate(Ridge(alpha=np.float64(0.5)), report)["ablation"]["curve"]
    for other in (Ridge(alpha=np.float64(0.6)), Ridge(alpha=0.6)):
        with pytest.raises(ValueError, match="'m'"):
            ablate(other, report)
    report = legacy(report_of(forest(max_features=1)))
    with pytest.raises(ValueError, match="'m'"):
        ablate(forest(max_features=1.0), report)


def test_2x_report_missing_a_parameter_is_refused():
    report = json.loads(json.dumps(report_of(forest(n_estimators=np.int64(10)))))
    del report["models"]["m"]["params"]["n_estimators"]
    with pytest.raises(ValueError, match="'m'"):
        ablate(forest(n_estimators=np.int64(10)), report)


# benchmark


def test_rf_arm_never_selects_an_all_missing_column():
    path = Path(__file__).parents[1] / "benchmarks" / "realdata" / "run.py"
    spec = importlib.util.spec_from_file_location("realdata_run_6", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["realdata_run_6"] = module
    spec.loader.exec_module(module)
    x = np.arange(40.0)
    data = np.c_[np.full(40, np.nan), np.zeros(40), np.ones(40), x]
    kept = module._select("rf_median", data, x, "reg", 0, True, 20)
    assert 3 in kept and 0 not in kept and max(kept) <= 3
    data[:5, 1] = np.nan
    assert 0 not in module._select("rf_median", data, x, "reg", 0, True, 20)
    assert 0 not in module._select("rf_median", data, (x > 20).astype(int), "clf", 0, True, 20)
