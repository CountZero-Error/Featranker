import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import featranker.importance as importance_module
from featranker import ConsensusSelector, FeatureRanker, summarize

N = 120
N_TRAIN = 80
RNG = np.random.default_rng(0)
FRAME = pd.DataFrame(RNG.normal(size=(N, 3)), columns=["a", "b", "c"])
Y = 2 * FRAME["a"].to_numpy() + FRAME["b"].to_numpy() + 0.1 * RNG.normal(size=N)
TRAIN, EVAL = FRAME.iloc[:N_TRAIN], FRAME.iloc[N_TRAIN:]


def fit_rank(r, **kwargs):
    r.fit(TRAIN, Y[:N_TRAIN])
    kwargs = {"n_repeats": 2, "random_state": 0, "scoring": "r2", **kwargs}
    return r.rank_features(EVAL, Y[N_TRAIN:], **kwargs)


class Recorder(RegressorMixin, BaseEstimator):
    """Logs the type and columns of every matrix it is fit or scored on."""

    def fit(self, X, y):
        self.fit_seen_ = [(type(X).__name__, list(getattr(X, "columns", [])))]
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        self.fit_seen_.append((type(X).__name__, list(getattr(X, "columns", []))))
        return np.full(len(X), self.mean_)


def by_name_pipeline(remainder="passthrough", columns=["a", "b"]):
    return make_pipeline(
        ColumnTransformer([("num", StandardScaler(), columns)], remainder=remainder),
        Ridge(),
    )


# -- option ----------------------------------------------------------------


def test_model_input_defaults_to_array_and_is_not_recorded():
    r = FeatureRanker("reg", "fast", verbose=False)
    assert r.model_input == "array"
    assert "model_input" not in fit_rank(r)


@pytest.mark.parametrize("value", ["frame", "DATAFRAME", None, True])
def test_invalid_model_input_raises_at_construction(value):
    with pytest.raises(ValueError, match="model_input"):
        FeatureRanker("reg", verbose=False, model_input=value)


def test_dataframe_without_pandas_raises_import_error(monkeypatch):
    monkeypatch.setattr(importance_module, "_pd", None)
    with pytest.raises(ImportError, match="pandas"):
        FeatureRanker("reg", verbose=False, model_input="dataframe")


def test_ranker_unpickled_without_model_input_behaves_as_array():
    state = pickle.loads(pickle.dumps(FeatureRanker("reg", verbose=False))).__dict__.copy()
    state.pop("model_input")  # what a 1.9 pickle holds
    old = FeatureRanker.__new__(FeatureRanker)
    old.__setstate__(state)
    assert old.model_input == "array"


# -- named columns ---------------------------------------------------------


def test_dataframe_mode_gives_every_call_named_columns_with_shadows():
    r = FeatureRanker(
        "reg", models={"rec": Recorder()}, verbose=False, shadows=True,
        shadow_seed=0, evidence=True, model_input="dataframe",
    )
    report = fit_rank(r)
    expected = ["a", "b", "c", "shadow__a", "shadow__b", "shadow__c"]
    assert report["model_input"] == "dataframe"
    seen = r.trained_models["rec"].fit_seen_
    assert len(seen) > 3  # fit, baseline, permuted repeats, train score
    assert all(kind == "DataFrame" and cols == expected for kind, cols in seen)


def test_array_mode_gives_arrays():
    r = FeatureRanker("reg", models={"rec": Recorder()}, verbose=False)
    fit_rank(r)
    assert {kind for kind, _ in r.trained_models["rec"].fit_seen_} == {"ndarray"}


def test_dataframe_mode_without_shadows_uses_feature_names_only():
    r = FeatureRanker("reg", models={"rec": Recorder()}, verbose=False, model_input="dataframe")
    fit_rank(r)
    assert all(cols == ["a", "b", "c"] for _, cols in r.trained_models["rec"].fit_seen_)


def test_numpy_input_uses_given_names_in_dataframe_mode():
    r = FeatureRanker("reg", models={"rec": Recorder()}, verbose=False, model_input="dataframe")
    r.fit(TRAIN.to_numpy(), Y[:N_TRAIN], feature_names=["p", "q", "r"])
    assert r.trained_models["rec"].fit_seen_[0][1] == ["p", "q", "r"]


def test_select_by_name_ranks_every_group():
    r = FeatureRanker(
        "reg", models={"named": by_name_pipeline()}, verbose=False,
        shadows=True, shadow_seed=0, model_input="dataframe",
    )
    report = fit_rank(r)
    assert set(report["models"]["named"]["importance"]) == {"a", "b", "c"}
    assert not [f for f in report["failures"] if f["stage"] == "shadows"]


def test_array_mode_rejects_name_selection_as_before():
    r = FeatureRanker("reg", models={"named": by_name_pipeline()}, verbose=False)
    with pytest.raises(RuntimeError, match="No models fit"):
        fit_rank(r)


def test_built_in_reports_equal_between_modes():
    reports = []
    for mode in ("array", "dataframe"):
        r = FeatureRanker(
            "reg", "fast", verbose=False, shadows=True, shadow_seed=3, model_input=mode
        )
        reports.append(fit_rank(r))
    array, frame = reports
    assert array["consensus"] == frame["consensus"]
    for name in array["models"]:
        assert array["models"][name]["importance"] == frame["models"][name]["importance"]
        assert array["models"][name]["shadow_importance"] == frame["models"][name]["shadow_importance"]


def test_shap_summary_unchanged_between_modes():
    pytest.importorskip("shap")
    reports = []
    # SHAP must get DataFrames for models fit on one, or sklearn warns.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*feature names.*", category=UserWarning)
        for mode in ("array", "dataframe"):
            r = FeatureRanker(
                "reg", "fast", verbose=False, shadows=True, shadow_seed=3,
                evidence=True, model_input=mode,
            )
            reports.append(fit_rank(r))
    array, frame = reports
    assert array["evidence"]["shap"]["available"]
    assert array["evidence"]["shap"] == frame["evidence"]["shap"]
    for name, entry in array["models"].items():
        assert entry.get("shap") == frame["models"][name].get("shap")
        assert entry["train_score"] == frame["models"][name]["train_score"]


def test_shadow_prefixed_feature_name_is_rejected_in_dataframe_mode():
    frame = TRAIN.rename(columns={"a": "shadow__a"})
    r = FeatureRanker("reg", "fast", verbose=False, shadows=True, model_input="dataframe")
    with pytest.raises(ValueError, match="shadow__"):
        r.fit(frame, Y[:N_TRAIN])
    # Array mode and shadows off are unaffected.
    FeatureRanker("reg", "fast", verbose=False, shadows=True).fit(frame, Y[:N_TRAIN])
    FeatureRanker("reg", "fast", verbose=False, model_input="dataframe").fit(
        frame, Y[:N_TRAIN]
    )


def test_categorical_and_impute_rank_the_same_in_both_modes():
    rng = np.random.default_rng(1)
    frame = FRAME.copy()
    frame["kind"] = rng.choice(["x", "y", "z"], size=N)
    frame.loc[frame.index[::7], "a"] = np.nan
    y = Y + (frame["kind"] == "x").to_numpy()
    consensus = []
    for mode in ("array", "dataframe"):
        r = FeatureRanker(
            "reg", "fast", verbose=False, missing="impute", categorical=["kind"],
            model_input=mode,
        )
        r.fit(frame.iloc[:N_TRAIN], y[:N_TRAIN])
        report = r.rank_features(
            frame.iloc[N_TRAIN:], y[N_TRAIN:], n_repeats=2, random_state=0, scoring="r2"
        )
        consensus.append(report["consensus"])
    assert consensus[0] and consensus[0] == consensus[1]


# -- shadow compatibility --------------------------------------------------


def shadow_failures(report):
    return [f for f in report["failures"] if f["stage"] == "shadows"]


def ranked(model, mode="array", **kwargs):
    models = {"keep": make_pipeline(StandardScaler(), Ridge()), "m": model}
    r = FeatureRanker(
        "reg", models=models, verbose=False, shadows=True, shadow_seed=0,
        model_input=mode, **kwargs,
    )
    return fit_rank(r)


@pytest.mark.parametrize("mode", ["array", "dataframe"])
def test_dropped_shadows_exclude_model_from_shadow_evidence(mode):
    selector = ["a", "b", "c"] if mode == "dataframe" else [0, 1, 2]
    report = ranked(by_name_pipeline("drop", selector), mode)
    model = report["models"]["m"]
    assert "shadow_max" not in model and "shadow_importance" not in model
    assert all("above_shadow" not in e for e in model["importance"].values())
    [failure] = shadow_failures(report)
    assert failure["model"] == "m" and failure["error_type"] == "ShadowColumnsDropped"
    # Still votes; the consensus shadow shares come from the compatible model only.
    assert all(row["n_models"] == 2 for row in report["consensus"])
    assert all(list(row["family_above_shadow"]) == ["keep"] for row in report["consensus"])


@pytest.mark.parametrize("mode", ["array", "dataframe"])
def test_remainder_passthrough_keeps_shadows(mode):
    selector = ["a", "b"] if mode == "dataframe" else [0, 1]
    report = ranked(by_name_pipeline("passthrough", selector), mode)
    assert not shadow_failures(report)
    assert "shadow_max" in report["models"]["m"]


def test_nested_pipeline_is_inspected():
    inner = by_name_pipeline("drop", [0, 1, 2])
    report = ranked(Pipeline([("inner", inner)]))
    assert [f["model"] for f in shadow_failures(report)] == ["m"]


def test_drop_transformer_does_not_count_as_use():
    model = make_pipeline(
        ColumnTransformer([("x", "drop", [3, 4, 5])], remainder="passthrough"), Ridge()
    )
    # Shadows are dropped explicitly, though the remainder keeps the real columns.
    assert [f["model"] for f in shadow_failures(ranked(model))] == ["m"]


@pytest.mark.parametrize(
    "columns",
    [[0, 1, 2], slice(0, 3), np.array([True] * 3 + [False] * 3), lambda X: [0, 1, 2]],
)
def test_selector_kinds_resolve(columns):
    report = ranked(by_name_pipeline("drop", columns))
    assert [f["model"] for f in shadow_failures(report)] == ["m"]


def test_selector_covering_the_shadows_is_compatible():
    report = ranked(by_name_pipeline("drop", slice(0, 6)))
    assert not shadow_failures(report)


def test_unresolvable_selector_cannot_be_verified_and_counts_as_dropped():
    model = by_name_pipeline("drop", [0, 1, 2]).fit(np.ones((10, 6)), np.arange(10))
    del model[0]._transformer_to_input_indices
    assert importance_module._shadow_status(model, 3, 6) == "unverified"


def test_shadows_off_has_no_check():
    r = FeatureRanker(
        "reg", models={"m": by_name_pipeline("drop", [0, 1, 2])}, verbose=False
    )
    assert not shadow_failures(fit_rank(r))


def test_summary_reports_shadows_stage_as_warning():
    report = ranked(by_name_pipeline("drop", [0, 1, 2]))
    [finding] = [f for f in summarize(report) if f["code"] == "failures"]
    assert finding["severity"] == "warning"
    assert finding["evidence"] == {"stage": "shadows", "models": ["m"]}


# -- pass-through ----------------------------------------------------------


def selector(**kwargs):
    return ConsensusSelector(
        "reg", "r2", k=2, models={"m": by_name_pipeline()}, cv=3, n_repeats=2,
        random_state=0, **kwargs,
    )


def test_selector_selects_with_named_pipeline():
    fitted = selector(model_input="dataframe").fit(FRAME, Y)
    assert len(fitted.selected_groups_) == 2
    assert fitted.report_["splits"][0]["report"]["model_input"] == "dataframe"


def test_selector_default_clone_params_and_pickle():
    s = selector(model_input="dataframe")
    assert ConsensusSelector("reg", "r2", k=1).model_input == "array"
    assert s.get_params()["model_input"] == "dataframe"
    from sklearn.base import clone

    assert clone(s).model_input == "dataframe"
    state = pickle.loads(pickle.dumps(s)).__dict__.copy()
    state.pop("model_input")
    old = ConsensusSelector.__new__(ConsensusSelector)
    old.__setstate__(state)
    assert old.model_input == "array"


def test_selector_in_pipeline_cross_validates():
    pipe = make_pipeline(selector(model_input="dataframe"), Ridge())
    assert np.isfinite(cross_val_score(pipe, FRAME, Y, cv=2)).all()


def test_rank_resampled_uses_mode_in_every_split():
    r = FeatureRanker(
        "reg", models={"named": by_name_pipeline()}, verbose=False, model_input="dataframe"
    )
    report = r.rank_resampled(FRAME, Y, scoring="r2", cv=3, n_repeats=2, random_state=0)
    assert report["model_input"] == "dataframe"
    assert not report["failures"]
    assert all(s["report"]["model_input"] == "dataframe" for s in report["splits"])


def test_default_resampled_report_has_no_model_input():
    r = FeatureRanker("reg", "fast", verbose=False)
    report = r.rank_resampled(FRAME, Y, scoring="r2", cv=3, n_repeats=2, random_state=0)
    assert "model_input" not in report
