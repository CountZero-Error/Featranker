"""Acceptance check for codex-review-5 (lead-written; builder must not edit)."""
import itertools
import json
import warnings

import numpy as np
import pytest
from sklearn import linear_model as lm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from featranker import FeatureRanker

PATTERNS = np.array([[1, -1], [-1, -1], [1, 1], [-1, 1]], float)
TRAIN = np.tile(PATTERNS, (20, 1))
LINE = np.linspace(-1, 1, 41)


@pytest.fixture
def shap_installed():
    return pytest.importorskip("shap")


def swap(X):
    return np.asarray(X)[:, ::-1]


class ReverseRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        return super().fit(swap(X), y, sample_weight)

    def predict(self, X):
        return super().predict(swap(X))


class ReversePoisson(lm.PoissonRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(swap(X), y, sample_weight)

    def predict(self, X):
        return super().predict(swap(X))


def ranked(model, X_train, y_train, X, y, task="reg", scoring=None, **kwargs):
    names = list("abcdef")[: X.shape[1]]
    ranker = FeatureRanker(task, models={"m": model}, evidence=True, verbose=False)
    ranker.fit(X_train, y_train, feature_names=names)
    report = ranker.rank_features(
        X, y, scoring=scoring or ("r2" if task == "reg" else "neg_log_loss"),
        n_repeats=2, random_state=0, **kwargs)
    json.dumps(report, allow_nan=False)
    return report


def supported(report):
    result = "shap" in report["models"]["m"]
    assert result != bool(report["evidence"]["shap"]["unsupported"])
    return result


EVALS = {
    "monotone": np.c_[LINE, LINE ** 3],
    "equal": np.c_[LINE, LINE],
    "exp": np.c_[LINE, np.exp(3 * LINE)],
    **{f"order{i}": PATTERNS[list(p)] for i, p in enumerate(itertools.permutations(range(4)))},
}


@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
@pytest.mark.parametrize("X", EVALS.values(), ids=EVALS)
def test_swapped_columns_never_pass(shap_installed, X, wrap):
    # offset 5: outputs are positive, so the log link is tried too
    for offset in (0.0, 5.0):
        y_train, y = TRAIN[:, 0] + offset, X[:, 0] + offset
        assert not supported(ranked(wrap(ReverseRidge()), TRAIN, y_train, X, y))
        assert supported(ranked(wrap(Ridge()), TRAIN, y_train, X, y))
    y_train, y = np.exp(TRAIN[:, 0]), np.exp(X[:, 0])
    assert not supported(ranked(wrap(ReversePoisson()), TRAIN, y_train, X, y))
    assert supported(ranked(wrap(lm.PoissonRegressor()), TRAIN, y_train, X, y))


def test_row_order_does_not_matter(shap_installed):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.exp(X[:, 0])
    for model in (lm.PoissonRegressor(), lm.GammaRegressor(), Ridge()):
        for order in (np.arange(60), rng.permutation(60), np.argsort(X[140:, 0])):
            assert supported(ranked(model, X[:140], y[:140], X[140:][order], y[140:][order]))


def test_one_huge_output_does_not_hide_changed_columns(shap_installed):
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 2))
    X = rng.normal(size=(60, 2))
    X[0] = [12.0, 12.0]
    for cls, expected in ((ReversePoisson, False), (lm.PoissonRegressor, True)):
        model = cls(alpha=1e-3, max_iter=1000)
        assert supported(ranked(
            model, train, np.exp(train[:, 0]), X, np.exp(X[:, 0]))) is expected


def test_gblinear_stays_supported(shap_installed):
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    z = X[:, 0] + 0.3 * X[:, 1]
    cases = [
        (xgboost.XGBRegressor(booster="gblinear"), z, "reg"),
        (xgboost.XGBRegressor(booster="gblinear", objective="count:poisson"), np.exp(z), "reg"),
        (xgboost.XGBRegressor(booster="gblinear", objective="reg:logistic"), 1 / (1 + np.exp(-z)), "reg"),
        (xgboost.XGBClassifier(booster="gblinear"), (z > 0).astype(int), "clf"),
    ]
    for model, y, task in cases:
        for wrapped in (model, make_pipeline(StandardScaler(), model)):
            assert supported(ranked(wrapped, X[:200], y[:200], X[200:], y[200:], task))


class Float32Reverse(ReverseRidge):
    def predict(self, X):
        return super().predict(X).astype(np.float32)


@pytest.mark.parametrize("base,relative,cls", [
    (1e-13, 4e-3, ReverseRidge), (3e-14, 2e-2, ReverseRidge), (1e-12, 8e-4, ReverseRidge),
    (2e-5, 2e-2, Float32Reverse), (1e-4, 4e-3, Float32Reverse), (1e-3, 4e-4, Float32Reverse),
])
def test_tiny_outputs_do_not_hide_changed_columns(shap_installed, base, relative, cls):
    X = np.random.default_rng(0).normal(size=(300, 2))
    y = base * (1 + relative * np.clip(X[:, 0], -2, 2))
    assert not supported(ranked(cls(), X[:200], y[:200], X[200:], y[200:]))
    if cls is ReverseRidge:
        assert supported(ranked(Ridge(), X[:200], y[:200], X[200:], y[200:]))


@pytest.mark.parametrize("wrap,alpha", [
    (lambda m: m, 1e-3), (lambda m: make_pipeline(StandardScaler(), m), 1e-6)])
def test_underflowing_predictions_stay_supported(shap_installed, wrap, alpha):
    rng = np.random.default_rng(0)
    X = rng.lognormal(sigma=2.5, size=(1000, 3))
    y = rng.poisson(np.exp(1 - 0.8 * X[:, 0])).astype(float)
    model = wrap(lm.PoissonRegressor(alpha=alpha, max_iter=5000))
    assert supported(ranked(model, X[:700], y[:700], X[700:], y[700:]))


@pytest.mark.parametrize("value", [1e3, 4e6])
@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
def test_equal_outlier_rows_do_not_hide_changed_columns(shap_installed, wrap, value):
    class ReverseOLS(lm.LinearRegression):
        def fit(self, X, y, sample_weight=None):
            return super().fit(swap(X), y, sample_weight)

        def predict(self, X):
            return super().predict(swap(X))

    X = np.random.default_rng(0).normal(size=(300, 2))
    X[0] = X[250] = value
    y = X[:, 0] + 0.01 * np.random.default_rng(1).normal(size=300)
    for model, expected in ((ReverseOLS(), False), (lm.LinearRegression(), True)):
        assert supported(ranked(wrap(model), X[:200], y[:200], X[200:], y[200:])) is expected


@pytest.mark.parametrize("scaler", [MinMaxScaler, StandardScaler])
@pytest.mark.parametrize("seed", range(4))
def test_redundant_columns_stay_supported(shap_installed, scaler, seed):
    rng = np.random.default_rng(seed)
    cm = 170 + 10 * rng.normal(size=600)
    X = np.c_[cm, cm / 2.54, rng.normal(size=600)]
    y = 0.1 * (cm - 170) + 0.5 * X[:, 2] + 0.3 * rng.normal(size=600)
    assert supported(ranked(
        make_pipeline(scaler(), lm.LinearRegression()), X[:400], y[:400], X[400:], y[400:],
        feature_groups={"height": ["a", "b"], "u": ["c"]}))


class Cancelling(RegressorMixin, BaseEstimator):
    """Exposes coefficients whose absolute sum overflows; predicts from another column."""

    def fit(self, X, y):
        self.coef_, self.intercept_ = np.array([1e308, -1e308, 1e307, 0.0]), 0.0
        return self

    def predict(self, X):
        return np.asarray(X)[:, 3] * 1e307


def test_rounding_allowance_stays_finite(shap_installed):
    X = np.array([[u, u, v, w] for u, v, w in itertools.product([-1.0, 1.0], repeat=3)])
    y = 1e307 * X[:, 3]
    scorer = make_scorer(lambda a, b: r2_score(a / 1e307, b / 1e307))
    report = ranked(Cancelling(), X, y, X, y, scoring=scorer,
                    feature_groups={"g": ["a", "b"], "v": ["c"], "w": ["d"]})
    assert not supported(report)


# ablate()


D = np.random.default_rng(42).normal(size=(150, 2))
SPLIT = (D[:100], D[:100, 0], D[100:], D[100:, 0])


def fit(model):
    return FeatureRanker("reg", models={"m": model}, verbose=False).fit(
        SPLIT[0], SPLIT[1], feature_names=["a", "b"])


@pytest.mark.parametrize("key,old,new,value", [
    ("alpha", np.float32(0.01), np.float32(50000), float(np.float32(0.01))),
    ("alpha", np.float64(0.01), np.float64(50000), 0.01),
    ("alpha", np.longdouble(0.01), np.longdouble(50000), 0.01),
    ("max_iter", np.int64(100), np.int64(5), 100),
    ("fit_intercept", np.bool_(True), np.bool_(False), True),
])
@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
def test_numpy_scalar_params_are_compared(wrap, key, old, new, value):
    inner = Ridge(**{key: old})
    model = wrap(inner)
    ranker = fit(model)
    report = ranker.rank_features(*SPLIT[2:], scoring="r2", n_repeats=2, random_state=0)
    params = report["models"]["m"]["params"]
    recorded = params[key if model is inner else f"ridge__{key}"]
    assert recorded == value and type(recorded) is type(value)
    json.dumps(report, allow_nan=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ranker.ablate(*SPLIT, json.loads(json.dumps(report)), ks=[2])
    inner.set_params(**{key: new})
    ranker.fit(SPLIT[0], SPLIT[1], feature_names=["a", "b"])
    with pytest.raises(ValueError, match="'m'"):
        ranker.ablate(*SPLIT, report, ks=[2])


class WithExtra(Ridge):
    def __init__(self, alpha=1.0, extra=None):
        super().__init__(alpha=alpha)
        self.extra = extra


def test_other_numpy_values_stay_out_of_params():
    for extra in (np.float32("nan"), np.float64("inf"), np.complex64(1), np.datetime64("2020-01-01"),
                  np.timedelta64(1, "D"), np.timedelta64("NaT"), np.timedelta64(3, "ns"),
                  np.longdouble("inf"), np.array([1.0])):
        ranker = fit(WithExtra(extra=extra))
        report = ranker.rank_features(*SPLIT[2:], scoring="r2", n_repeats=2, random_state=0)
        json.dumps(report, allow_nan=False)
        assert "extra" not in report["models"]["m"]["params"]
