import inspect

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, make_scorer, mean_absolute_error

import featranker.importance as importance_module
from featranker import FeatureRanker

N, N_TRAIN, N_REPEATS = 300, 100, 3
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, 3))
Y = 2 * X[:, 0] + X[:, 1] + 0.1 * RNG.normal(size=N)
YC = (X[:, 0] + 0.5 * RNG.normal(size=N) > 0).astype(int)
W = RNG.uniform(0.5, 2.0, size=N)
NAMES = ["a", "b", "c"]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, N)
COPIES = 1 + len(NAMES) * N_REPEATS  # baseline + every group x repeat
# The per-model self-check also scores the baseline once per scorer, directly.

CALLS: dict = {}
_DEPTH = [0]


def _recorded(base, method):
    def call(self, *args, **kwargs):
        # Count outermost calls only: LogisticRegression.predict calls decision_function.
        if _DEPTH[0] == 0:
            CALLS[method] = CALLS.get(method, 0) + 1
        _DEPTH[0] += 1
        try:
            return getattr(base, method)(self, *args, **kwargs)
        finally:
            _DEPTH[0] -= 1

    call.__name__ = method  # scikit-learn reads the response method's name
    return call


class RecordingRidge(Ridge):
    predict = _recorded(Ridge, "predict")


class RecordingLogistic(LogisticRegression):
    predict = _recorded(LogisticRegression, "predict")
    predict_proba = _recorded(LogisticRegression, "predict_proba")
    decision_function = _recorded(LogisticRegression, "decision_function")


def failing_error(y_true, y_pred):
    raise ValueError("cacheable boom")


def kw_error(y_true, y_pred, **kwargs):
    """Takes weights through **kwargs only, so scikit-learn 1.7 would not pass them."""
    return -np.average(np.abs(y_true - y_pred), weights=kwargs.get("sample_weight"))


def plain_boom(model, X_, y_):
    raise ValueError("plain boom")


def build(task, model, **kwargs):
    ranker = FeatureRanker(task, models={"m": model}, verbose=False, **kwargs)
    return ranker.fit(X[TRAIN], (Y if task == "reg" else YC)[TRAIN], feature_names=NAMES)


def rank(task="reg", scoring=("r2", "neg_mean_absolute_error"), model=None, **kwargs):
    model = model or (Ridge() if task == "reg" else LogisticRegression())
    ranker = build(task, model, shadows=kwargs.pop("shadows", False), shadow_seed=0)
    y = (Y if task == "reg" else YC)[EVAL]
    return ranker.rank_features(
        X[EVAL], y, scoring=list(scoring), n_repeats=N_REPEATS, random_state=0, **kwargs
    )


def without_cache(monkeypatch, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", None)
        return rank(**kwargs)


def test_regression_scorers_share_one_predict_per_copy():
    ranker = build("reg", RecordingRidge())
    CALLS.clear()
    ranker.rank_features(
        X[EVAL], Y[EVAL], scoring=["r2", "neg_mean_absolute_error"],
        n_repeats=N_REPEATS, random_state=0,
    )
    assert CALLS["predict"] == COPIES + 2


def test_fallback_predicts_once_per_scorer(monkeypatch):
    ranker = build("reg", RecordingRidge())
    CALLS.clear()
    monkeypatch.setattr(importance_module, "_MultimetricScorer", None)
    ranker.rank_features(
        X[EVAL], Y[EVAL], scoring=["r2", "neg_mean_absolute_error"],
        n_repeats=N_REPEATS, random_state=0,
    )
    assert CALLS["predict"] == 2 * COPIES


def test_classification_mixes_one_call_per_response_method():
    ranker = build("clf", RecordingLogistic())
    CALLS.clear()
    ranker.rank_features(
        X[EVAL], YC[EVAL], scoring=["accuracy", "roc_auc", "neg_log_loss", "f1"],
        n_repeats=N_REPEATS, random_state=0,
    )
    assert CALLS == {
        "predict": COPIES + 2, "decision_function": COPIES + 1, "predict_proba": COPIES + 1,
    }


SCENARIOS = {
    "plain": {},
    "shadows": {"shadows": True},
    "weights": {"sample_weight": W[EVAL]},
    "subsample": {"max_samples": 40},
    "classification": {
        "task": "clf", "scoring": ("accuracy", "roc_auc", "neg_log_loss", "f1"),
    },
    "callable": {"scoring": ("r2", lambda m, X_, y_: -float(np.mean(m.predict(X_) - y_)))},
    "weights+shadows+classification": {
        "task": "clf", "shadows": True, "sample_weight": W[EVAL],
        "scoring": ("accuracy", "f1", "neg_log_loss"),
    },
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_report_equals_the_uncached_report(monkeypatch, name):
    kwargs = SCENARIOS[name]
    assert dict(rank(**kwargs)) == dict(without_cache(monkeypatch, **kwargs))


def test_resampled_report_equals_the_uncached_report(monkeypatch):
    def resampled():
        return FeatureRanker(
            "reg", models={"m": Ridge()}, verbose=False, shadows=True, shadow_seed=0
        ).rank_resampled(
            X, Y, scoring=["r2", "neg_mean_absolute_error"], cv=3, n_repeats=2,
            random_state=0, feature_names=NAMES,
        )

    cached = dict(resampled())
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", None)
        assert cached == dict(resampled())


def test_secondary_failures_keep_their_entries(monkeypatch):
    scoring = (
        "r2", make_scorer(failing_error), plain_boom, "neg_mean_absolute_error",
        "neg_mean_squared_error",
    )
    cached = rank(scoring=scoring)
    assert dict(cached) == dict(without_cache(monkeypatch, scoring=scoring))
    messages = sorted(f["message"] for f in cached["failures"])
    assert messages == [
        "scorer failing_error: cacheable boom", "scorer plain_boom: plain boom",
    ]
    secondary = cached["models"]["m"]["secondary"]
    assert secondary["failing_error"] is None and secondary["plain_boom"] is None
    assert secondary["neg_mean_absolute_error"] and secondary["neg_mean_squared_error"]


def test_primary_failure_keeps_its_message(monkeypatch):
    scoring = (make_scorer(failing_error), "r2", "neg_mean_absolute_error")
    with pytest.raises(RuntimeError) as cached:
        rank(scoring=scoring)
    with pytest.raises(RuntimeError) as uncached:
        without_cache(monkeypatch, scoring=scoring)
    assert str(cached.value) == str(uncached.value)
    assert '"stage": "ranking"' in str(cached.value) and "cacheable boom" in str(cached.value)


def test_missing_response_method_falls_back_without_a_crash(monkeypatch):
    class NoProba(Ridge):
        pass

    # roc_auc needs decision_function or predict_proba, which a regressor lacks.
    scoring = ("r2", "roc_auc", "neg_mean_absolute_error")
    cached = rank(scoring=scoring, model=NoProba())
    assert dict(cached) == dict(without_cache(monkeypatch, scoring=scoring, model=NoProba()))
    assert cached["models"]["m"]["secondary"]["roc_auc"] is None
    assert cached["models"]["m"]["secondary"]["neg_mean_absolute_error"] is not None


def test_cached_scores_match_manual_scoring():
    ranker = build("reg", Ridge())
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring=["r2", "neg_mean_absolute_error"],
        n_repeats=N_REPEATS, random_state=0,
    )
    expected = -mean_absolute_error(Y[EVAL], ranker.trained_models["m"].predict(X[EVAL]))
    entry = report["models"]["m"]["secondary"]["neg_mean_absolute_error"]
    assert entry["evaluation_score"] == expected


def ap_scorer(pos_label, method="predict_proba"):
    return make_scorer(average_precision_score, response_method=method, pos_label=pos_label)


@pytest.fixture
def trust_sharing(monkeypatch):
    """Switch the baseline self-check off so the grouping rules are tested alone."""
    monkeypatch.setattr(importance_module, "_shared_matches", lambda direct, shared: True)


def test_pos_label_zero_does_not_reuse_the_default_probability_column(
    monkeypatch, trust_sharing
):
    scoring = ("neg_log_loss", ap_scorer(0))
    cached = rank("clf", scoring=scoring)
    assert dict(cached) == dict(without_cache(monkeypatch, task="clf", scoring=scoring))


def test_decision_function_pos_label_zero_does_not_flip_roc_auc(monkeypatch, trust_sharing):
    scoring = ("roc_auc", ap_scorer(0, "decision_function"))
    cached = rank("clf", scoring=scoring)
    assert dict(cached) == dict(without_cache(monkeypatch, task="clf", scoring=scoring))


class DropsUnnamedWeights:
    """Like scikit-learn 1.7: sample_weight reaches only score functions that name it."""

    def __init__(self, *, scorers, raise_exc=True):
        self.scorers = scorers

    def __call__(self, model, X_, y_, **kwargs):
        out = {}
        for key, scorer in self.scorers.items():
            named = "sample_weight" in inspect.signature(scorer._score_func).parameters
            out[key] = scorer(model, X_, y_, **(kwargs if named else {}))
        return out


def test_weights_skip_scorers_whose_function_takes_no_weights(monkeypatch, trust_sharing):
    scoring = ("r2", "max_error", make_scorer(kw_error), "neg_mean_absolute_error")
    kwargs = {"scoring": scoring, "sample_weight": W[EVAL]}
    expected = dict(without_cache(monkeypatch, **kwargs))
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", DropsUnnamedWeights)
        cached = rank(**kwargs)
    assert dict(cached) == expected
    assert cached["models"]["m"]["secondary"]["max_error"] is None


class RaisesOnConstruction:
    def __init__(self, **kwargs):
        raise TypeError("signature changed")


def test_unusable_multimetric_scorer_falls_back(monkeypatch):
    expected = dict(without_cache(monkeypatch))
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", RaisesOnConstruction)
        assert dict(rank()) == expected


class ShiftedScores(DropsUnnamedWeights):
    def __call__(self, model, X_, y_, **kwargs):
        return {k: v + 1e-9 for k, v in super().__call__(model, X_, y_, **kwargs).items()}


def test_inexact_shared_scores_disable_sharing_for_the_model(monkeypatch):
    expected = dict(without_cache(monkeypatch))
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", ShiftedScores)
        assert dict(rank()) == expected
