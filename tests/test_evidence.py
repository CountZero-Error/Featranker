import builtins
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from featranker import FeatureRanker
from featranker.importance import _float_target, _validate_X, _validate_y
from featranker.evidence import CORRELATION_THRESHOLD, correlation_evidence

# Without the shap package every evidence run warns; that case has its own test.
pytestmark = pytest.mark.filterwarnings("ignore:shap is not installed:UserWarning")

EVIDENCE_KEYS = {"evidence", "train_score", "overfit_gap", "shap"}

RNG = np.random.default_rng(0)
X_DATA = RNG.normal(size=(120, 3))
Y_DATA = 2.0 * X_DATA[:, 0] + RNG.normal(size=120)  # noisy, so a deep tree overfits
NAMES = ["a", "b", "c"]
N_TRAIN = 90


def ranker(models=None, **kwargs):
    return FeatureRanker(
        task="reg", models=models or {"lin": LinearRegression()}, verbose=False, **kwargs
    )


def run(r, X=X_DATA, y=Y_DATA, names=NAMES, **kwargs):
    r.fit(X[:N_TRAIN], y[:N_TRAIN], names)
    kwargs = {"scoring": "r2", "n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_features(X[N_TRAIN:], y[N_TRAIN:], **kwargs)


def all_keys(obj):
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in all_keys(v)}
    return set()


# -- flag -------------------------------------------------------------------


def test_evidence_defaults_to_false_and_adds_no_keys():
    r = ranker()
    report = run(r)

    assert r.evidence is False
    assert not (all_keys(report) & EVIDENCE_KEYS)
    assert not hasattr(r, "train_X_")


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_evidence_rejects_non_bool(value):
    with pytest.raises(TypeError, match="evidence"):
        ranker(evidence=value)


def test_evidence_report_only_adds_keys():
    plain = run(ranker())
    rich = run(ranker(evidence=True))

    assert set(rich) - set(plain) == {"evidence"}
    assert set(rich["models"]["lin"]) - set(plain["models"]["lin"]) >= {
        "train_score", "overfit_gap"
    }
    for key in plain:
        if key != "models":
            assert rich[key] == plain[key]


def test_ranker_unpickled_without_the_attribute_is_not_evidence():
    r = ranker()
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES)
    state = pickle.loads(pickle.dumps(r)).__dict__.copy()
    del state["evidence"]
    old = FeatureRanker.__new__(FeatureRanker)

    old.__setstate__(state)
    report = old.rank_features(
        X_DATA[N_TRAIN:], Y_DATA[N_TRAIN:], scoring="r2", n_repeats=2, random_state=0
    )

    assert old.evidence is False
    assert not (all_keys(report) & EVIDENCE_KEYS)


def test_fit_keeps_training_data_only_with_evidence():
    r = ranker(shadows=True, shadow_seed=0, evidence=True)
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES)

    assert r.train_X_.shape == (N_TRAIN, 2 * len(NAMES))  # as the models saw it
    assert np.array_equal(r.train_y_, Y_DATA[:N_TRAIN])


def evidence_of_fit_then_mutation(make_train, mutate, models=None, **ranker_kwargs):
    """Reports from a ranker whose training data is mutated after fit() and from a control."""
    models = models or {"lin": LinearRegression(), "tree": DecisionTreeRegressor(random_state=0)}
    reports = []
    for should_mutate in (True, False):
        X_train, y_train = make_train()
        r = ranker(models, evidence=True, **ranker_kwargs)
        r.fit(X_train, y_train, NAMES)
        if should_mutate:
            mutate(X_train, y_train)
        reports.append(r.rank_features(
            X_DATA[N_TRAIN:], Y_DATA[N_TRAIN:], scoring="r2", n_repeats=2, random_state=0
        ))
    return reports


def zero_out(X_train, y_train):
    X_train[:] = 0
    y_train[:] = 0


def test_mutating_numpy_training_arrays_after_fit_leaves_evidence_unchanged():
    mutated, control = evidence_of_fit_then_mutation(
        lambda: (X_DATA[:N_TRAIN].copy(), Y_DATA[:N_TRAIN].copy()), zero_out
    )

    assert mutated["models"]["tree"]["train_score"] == 1.0
    assert mutated["models"] == control["models"]
    assert mutated["evidence"] == control["evidence"]


def test_mutating_a_float_dataframe_after_fit_leaves_evidence_unchanged():
    mutated, control = evidence_of_fit_then_mutation(
        lambda: (pd.DataFrame(X_DATA[:N_TRAIN].copy(), columns=NAMES), Y_DATA[:N_TRAIN].copy()),
        zero_out,
    )

    assert mutated["models"]["tree"]["train_score"] == 1.0
    assert mutated["models"] == control["models"]
    assert mutated["evidence"] == control["evidence"]


@pytest.mark.parametrize("shadows", [False, True])
def test_model_that_stores_its_training_data_is_unaffected_by_caller_mutation(shadows):
    mutated, control = evidence_of_fit_then_mutation(
        lambda: (X_DATA[:N_TRAIN].copy(), Y_DATA[:N_TRAIN].copy()), zero_out,
        models={"knn": KNeighborsRegressor()}, shadows=shadows, shadow_seed=0,
    )

    knn = mutated["models"]["knn"]
    assert knn["train_score"] > 0.5  # a zeroed training set would score about 0
    assert knn == control["models"]["knn"]  # evaluation_score, importances, train_score
    assert mutated["evidence"] == control["evidence"]


def test_validation_returns_arrays_that_share_no_memory_with_the_caller():
    X = X_DATA.copy()
    y = Y_DATA.copy()
    frame = pd.DataFrame(X_DATA.copy(), columns=NAMES)

    values, _ = _validate_X(X, feature_names=NAMES)
    frame_values, _ = _validate_X(frame)
    target = _validate_y(y, len(y))

    assert not np.shares_memory(values, X)
    assert not np.shares_memory(frame_values, frame.to_numpy())
    assert not np.shares_memory(target, y)
    assert not np.shares_memory(_float_target(target), y)


def test_rank_features_works_on_copies_of_the_evaluation_data():
    seen = []

    class RecordingKNN(KNeighborsRegressor):
        def predict(self, X):
            seen.append(X)
            return super().predict(X)

    r = ranker({"knn": RecordingKNN()}, evidence=True)
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES)
    X_eval, y_eval = X_DATA[N_TRAIN:].copy(), Y_DATA[N_TRAIN:].copy()
    seen.clear()

    r.rank_features(X_eval, y_eval, scoring="r2", n_repeats=1, random_state=0)

    assert seen and not any(np.shares_memory(batch, X_eval) for batch in seen)


# -- train score ------------------------------------------------------------


def test_unpruned_tree_overfits():
    report = run(ranker({"tree": DecisionTreeRegressor(random_state=0)}, evidence=True))
    model = report["models"]["tree"]

    assert model["train_score"] == 1.0
    assert model["overfit_gap"] == pytest.approx(1.0 - model["evaluation_score"])
    assert model["overfit_gap"] > 0


def test_train_score_works_with_shadows():
    report = run(
        ranker({"tree": DecisionTreeRegressor(random_state=0)},
               shadows=True, shadow_seed=0, evidence=True)
    )

    assert report["models"]["tree"]["train_score"] == 1.0
    assert "shadow_max" in report["models"]["tree"]


def test_train_scoring_error_is_an_evidence_failure_and_keeps_the_model():
    def scorer(model, X, y):
        if len(X) == N_TRAIN:
            raise ValueError("no training scores")
        return model.score(X, y)

    report = run(ranker(evidence=True), scoring=scorer)

    model = report["models"]["lin"]
    assert model["train_score"] is None and model["overfit_gap"] is None
    assert model["importance"] and report["consensus"]
    assert [(f["model"], f["stage"], f["error_type"]) for f in report["failures"]] == [
        ("lin", "evidence", "ValueError")
    ]


# -- correlated groups ------------------------------------------------------


def singletons(n):
    return {chr(ord("a") + i): [i] for i in range(n)}


def test_planted_copy_pair_is_found():
    rng = np.random.default_rng(1)
    e = rng.normal(size=200)
    X = np.column_stack([e, e + 0.1 * rng.normal(size=200), rng.normal(size=200)])

    block = correlation_evidence(X, {"e": [0], "f": [1], "g": [2]})

    assert block["method"] == "spearman" and block["data"] == "train"
    assert block["threshold"] == CORRELATION_THRESHOLD == 0.5
    assert [p["groups"] for p in block["pairs"]] == [["e", "f"]]
    assert block["pairs"][0]["rho"] > 0.9


def test_independent_features_give_no_pairs():
    X = np.random.default_rng(2).normal(size=(300, 4))

    assert correlation_evidence(X, singletons(4))["pairs"] == []


def test_multi_column_group_uses_strongest_member_pair_with_its_sign():
    rng = np.random.default_rng(3)
    b = rng.normal(size=300)
    X = np.column_stack(
        [rng.normal(size=300), -b + 0.5 * rng.normal(size=300), b]
    )

    pairs = correlation_evidence(X, {"a": [0, 1], "b": [2]})["pairs"]

    assert [p["groups"] for p in pairs] == [["a", "b"]]
    assert -1.0 <= pairs[0]["rho"] < -0.8


def test_constant_column_is_skipped_without_warning_or_nan():
    rng = np.random.default_rng(4)
    e = rng.normal(size=100)
    X = np.column_stack([np.ones(100), e, e])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pairs = correlation_evidence(X, singletons(3))["pairs"]

    assert [p["groups"] for p in pairs] == [["b", "c"]]


def test_pairs_sort_by_abs_rho_then_names():
    rng = np.random.default_rng(5)
    e = rng.normal(size=300)
    X = np.column_stack([e, e, e, e + 1.0 * rng.normal(size=300)])

    pairs = correlation_evidence(X, singletons(4))["pairs"]

    assert [p["groups"] for p in pairs[:3]] == [["a", "b"], ["a", "c"], ["b", "c"]]
    assert [abs(p["rho"]) for p in pairs] == sorted(
        (abs(p["rho"]) for p in pairs), reverse=True
    )


def test_report_correlation_ignores_shadow_columns_and_uses_training_rows():
    rng = np.random.default_rng(6)
    e = rng.normal(size=120)
    X = np.column_stack([e, e + 0.1 * rng.normal(size=120), rng.normal(size=120)])
    y = e + rng.normal(size=120)

    plain = run(ranker(evidence=True), X=X, y=y)
    shadowed = run(ranker(evidence=True, shadows=True, shadow_seed=0), X=X, y=y)

    assert plain["evidence"]["correlation"]["pairs"] == shadowed["evidence"]["correlation"]["pairs"]
    assert [p["groups"] for p in plain["evidence"]["correlation"]["pairs"]] == [["a", "b"]]


# -- shap missing -----------------------------------------------------------


def test_missing_shap_warns_and_keeps_the_rest_of_the_evidence(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)  # makes `import shap` fail

    with pytest.warns(UserWarning, match=r"featranker\[shap\]"):
        report = run(ranker(evidence=True))

    assert report["evidence"]["shap"] == {"available": False}
    assert report["evidence"]["correlation"]["pairs"] == []
    assert report["models"]["lin"]["train_score"] is not None
    assert "shap" not in report["models"]["lin"]
    json.dumps(report, allow_nan=False)


# -- resampled ----------------------------------------------------------------


def test_resampled_puts_evidence_in_every_split_only():
    r = ranker(evidence=True)

    report = r.rank_resampled(
        X_DATA, Y_DATA, scoring="r2", cv=3, n_repeats=2, random_state=0,
        feature_names=NAMES,
    )

    assert "evidence" not in report
    assert len(report["splits"]) == 3
    for entry in report["splits"]:
        assert "pairs" in entry["report"]["evidence"]["correlation"]
        assert entry["report"]["models"]["lin"]["train_score"] is not None


def test_resampled_without_evidence_is_unchanged():
    report = ranker().rank_resampled(
        X_DATA, Y_DATA, scoring="r2", cv=3, n_repeats=2, random_state=0,
        feature_names=NAMES,
    )

    assert not (all_keys(report) & EVIDENCE_KEYS)


def test_parallel_models_give_the_same_evidence():
    models = {"lin": LinearRegression(), "tree": DecisionTreeRegressor(random_state=0)}

    sequential = run(ranker(models, evidence=True))
    parallel = run(ranker(models, evidence=True, n_jobs=2))

    assert parallel["models"]["tree"]["train_score"] == 1.0
    assert parallel["evidence"]["correlation"] == sequential["evidence"]["correlation"]
    for name in models:
        assert parallel["models"][name]["overfit_gap"] == sequential["models"][name]["overfit_gap"]


def test_shap_import_that_crashes_is_a_warning_not_an_error(monkeypatch):
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "shap":
            raise AttributeError("module 'numpy' has no attribute 'broken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.warns(UserWarning, match=r"AttributeError.*featranker\[shap\]"):
        report = run(ranker(evidence=True))

    assert report["evidence"]["shap"] == {"available": False}
    assert report["models"]["lin"]["train_score"] is not None


def test_evidence_ranker_survives_a_pickle_round_trip():
    r = ranker(evidence=True)
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES)

    loaded = pickle.loads(pickle.dumps(r))
    report = loaded.rank_features(
        X_DATA[N_TRAIN:], Y_DATA[N_TRAIN:], scoring="r2", n_repeats=2, random_state=0
    )

    assert np.array_equal(loaded.train_X_, r.train_X_)
    assert np.array_equal(loaded.train_y_, r.train_y_)
    assert report["models"]["lin"]["train_score"] is not None
    assert "correlation" in report["evidence"]
