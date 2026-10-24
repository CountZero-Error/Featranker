import json
import pathlib
import pickle
import re

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import featranker.importance as importance_module
from featranker import ConsensusSelector, FeatureRanker, Report, summarize
from featranker.importance import _fingerprint

N, N_TRAIN, N_FEATURES = 200, 120, 4
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, N_FEATURES))
Y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * RNG.normal(size=N)
W = RNG.uniform(0.5, 2.0, size=N)
NAMES = [f"x{i}" for i in range(N_FEATURES)]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, None)
REAL = list(range(N_FEATURES))
SHADOWS = list(range(N_FEATURES, 2 * N_FEATURES))
ALL = list(range(2 * N_FEATURES))


# -- nested shadow-drop check -------------------------------------------------


def shadow_failures(report):
    return [f for f in report["failures"] if f["stage"] == "shadows"]


def shadow_ranker(model, **options):
    models = {"keep": make_pipeline(StandardScaler(), Ridge()), "m": model}
    ranker = FeatureRanker(
        "reg", models=models, verbose=False, shadows=True, shadow_seed=0, **options
    )
    return ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)


def rank_shadows(ranker):
    return ranker.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=2, random_state=0)


def ranked_with(model, **options):
    return rank_shadows(shadow_ranker(model, **options))


def ct(transformers, remainder="drop"):
    return ColumnTransformer(transformers, remainder=remainder)


def outer(inner, columns=ALL):
    return make_pipeline(ct([("inner", inner, columns)]), Ridge())


def test_inner_transformer_dropping_shadows_is_a_shadows_failure():
    report = ranked_with(outer(ct([("real", StandardScaler(), REAL)])))

    [failure] = shadow_failures(report)
    assert failure["model"] == "m" and failure["error_type"] == "ShadowColumnsDropped"
    assert "does not use every shadow column" in failure["message"]
    model = report["models"]["m"]
    assert "shadow_max" not in model and "shadow_importance" not in model


def test_nested_passthrough_keeps_shadows():
    report = ranked_with(outer(ct([("all", "passthrough", ALL)])))

    assert not shadow_failures(report)
    assert "shadow_max" in report["models"]["m"]


def test_nested_remainder_passthrough_keeps_shadows():
    report = ranked_with(outer(ct([("real", StandardScaler(), REAL)], remainder="passthrough")))

    assert not shadow_failures(report)


def test_pipeline_inside_a_branch_is_followed():
    inner = ct([("real", StandardScaler(), REAL)])

    report = ranked_with(outer(make_pipeline(StandardScaler(), inner)))

    assert [f["model"] for f in shadow_failures(report)] == ["m"]


def test_branch_columns_map_back_to_model_input():
    # The outer branch sees model columns 2..7; the inner transformer's
    # columns [0, 1] are model columns 2 and 3.
    branch = list(range(2, 8))
    kept = outer(ct([("first", StandardScaler(), [0, 1])], remainder="passthrough"), branch)
    dropped = outer(ct([("first", StandardScaler(), [0, 1])]), branch)

    assert not shadow_failures(ranked_with(kept))
    assert [f["model"] for f in shadow_failures(ranked_with(dropped))] == ["m"]


def test_unverifiable_model_is_a_failure_that_says_so():
    ranker = shadow_ranker(outer(ct([("all", "passthrough", ALL)])))
    model = ranker.trained_models["m"]
    del model[0]._transformer_to_input_indices

    assert importance_module._shadow_status(model, N_FEATURES, 2 * N_FEATURES) == "unverified"
    [failure] = shadow_failures(rank_shadows(ranker))
    assert failure["model"] == "m" and failure["error_type"] == "ShadowColumnsDropped"
    assert "could not be verified" in failure["message"]


def message_of_the_shadows_failure(report):
    [failure] = shadow_failures(report)
    return failure["message"]


def reordering_first_step():
    return ct([("shadow", "passthrough", SHADOWS), ("real", "passthrough", REAL)])


def test_later_transformer_after_a_reordering_step_cannot_be_verified():
    # After the reorder, positions 4..7 are the real columns: this keeps no shadow.
    drops = make_pipeline(reordering_first_step(), ct([("k", "passthrough", SHADOWS)]), Ridge())
    # ... and this keeps every shadow, though a naive mapping says it does not.
    keeps = make_pipeline(reordering_first_step(), ct([("k", "passthrough", REAL)]), Ridge())

    for model in (drops, keeps):
        assert "could not be verified" in message_of_the_shadows_failure(ranked_with(model))


def test_later_transformer_after_a_one_to_one_scaler_is_still_verified():
    dropping = make_pipeline(StandardScaler(), ct([("real", "passthrough", REAL)]), Ridge())
    keeping = make_pipeline(MinMaxScaler(), ct([("all", "passthrough", ALL)]), Ridge())

    assert "does not use every shadow" in message_of_the_shadows_failure(ranked_with(dropping))
    assert not shadow_failures(ranked_with(keeping))


# -- non-ColumnTransformer estimators ------------------------------------------


def test_selector_in_a_branch_that_drops_shadows_is_a_failure():
    selector = SelectKBest(f_regression, k=2)

    report = ranked_with(outer(selector))

    assert "does not use every shadow" in message_of_the_shadows_failure(report)
    assert [f["model"] for f in shadow_failures(ranked_with(make_pipeline(selector, Ridge())))] == ["m"]


def test_selector_keeping_everything_is_not_a_failure():
    assert not shadow_failures(ranked_with(outer(SelectKBest(f_regression, k="all"))))


def test_feature_union_of_transformers_keeping_only_real_columns_is_a_failure():
    union = make_union(ct([("r", "passthrough", REAL)]), ct([("r2", StandardScaler(), REAL)]))

    report = ranked_with(make_pipeline(union, Ridge()))

    assert "does not use every shadow" in message_of_the_shadows_failure(report)


def test_feature_union_is_the_union_of_its_parts():
    union = make_union(ct([("r", "passthrough", REAL)]), ct([("s", "passthrough", SHADOWS)]))

    assert not shadow_failures(ranked_with(make_pipeline(union, Ridge())))


def test_meta_estimator_wrapping_a_column_transformer_cannot_be_verified():
    inner = make_pipeline(ct([("r", "passthrough", REAL)]), Ridge())

    report = ranked_with(TransformedTargetRegressor(regressor=inner))

    assert "could not be verified" in message_of_the_shadows_failure(report)


def test_callable_columns_in_a_nested_branch_are_resolved():
    everything = lambda Z: list(range(Z.shape[1]))  # noqa: E731
    real_only = lambda Z: list(range(Z.shape[1] // 2))  # noqa: E731

    assert not shadow_failures(ranked_with(outer(ct([("a", "passthrough", everything)]))))
    assert [f["model"] for f in shadow_failures(
        ranked_with(outer(ct([("r", "passthrough", real_only)])))
    )] == ["m"]


def test_named_columns_and_selectors_in_a_nested_branch_of_dataframe_input():
    pytest.importorskip("pandas")
    every = make_column_selector(pattern=".*")
    real = make_column_selector(pattern="^x")
    names = NAMES + [f"shadow__{n}" for n in NAMES]
    options = {"model_input": "dataframe"}

    assert not shadow_failures(ranked_with(outer(ct([("a", "passthrough", every)]), names), **options))
    assert not shadow_failures(ranked_with(outer(ct([("a", "passthrough", names)]), names), **options))
    for columns in (real, NAMES):
        dropped = ranked_with(outer(ct([("r", "passthrough", columns)]), names), **options)
        assert [f["model"] for f in shadow_failures(dropped)] == ["m"]


@pytest.mark.parametrize("group", ["linear", "fast", "all"])
@pytest.mark.parametrize("model_input", ["array", "dataframe"])
def test_built_in_categorical_pipelines_keep_the_shadows(group, model_input):
    pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    values = np.empty((90, 3), dtype=object)
    values[:, 0] = rng.choice(list("abc"), 90)
    values[:, 1:] = rng.normal(size=(90, 2))
    y = rng.normal(size=90)
    ranker = FeatureRanker(
        "reg", group=group, verbose=False, shadows=True, shadow_seed=0,
        categorical=["c"], model_input=model_input,
    ).fit(values[:60], y[:60], feature_names=["c", "a", "b"])

    report = ranker.rank_features(values[60:], y[60:], scoring="r2", n_repeats=1, random_state=0)

    assert not shadow_failures(report)


# -- fingerprint scheme 2 -----------------------------------------------------

BIG = 2**53


def test_large_integer_labels_that_differ_only_by_swap_have_different_fingerprints():
    features = np.zeros((4, 1))
    y = np.array([BIG, BIG + 1, BIG, BIG + 1], dtype=np.int64)
    swapped = np.array([BIG + 1, BIG, BIG + 1, BIG], dtype=np.int64)

    assert _fingerprint(["a"], features, y) != _fingerprint(["a"], features, swapped)
    assert _fingerprint(["a"], features, y, scheme=1) == _fingerprint(
        ["a"], features, swapped, scheme=1
    )


def test_unsigned_and_bool_columns_hash_losslessly():
    features = np.zeros((2, 1))
    big = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    flags = np.array([True, False])

    assert _fingerprint(["a"], features, big) != _fingerprint(["a"], features, big[::-1].copy())
    assert _fingerprint(["a"], features, flags) != _fingerprint(["a"], features, ~flags)


def test_float_columns_hash_as_before_with_one_nan_pattern():
    values = np.array([[1.0, np.nan], [2.5, -np.nan]])
    y = np.array([0.1, np.nan])

    assert _fingerprint(["a", "b"], values, y) == _fingerprint(["a", "b"], values, y, scheme=1)


def test_fingerprints_depend_on_dtype_kind_not_size():
    features = np.zeros((2, 1))
    values = [1, 0]

    assert _fingerprint(["a"], features, np.array(values, np.int32)) == _fingerprint(
        ["a"], features, np.array(values, np.int64)
    )
    assert _fingerprint(["a"], features, np.array(values, bool)) == _fingerprint(
        ["a"], features, np.array(values, np.int64)
    )


def test_integer_and_float_columns_do_not_collide():
    features = np.zeros((2, 1))

    assert _fingerprint(["a"], features, np.array([1, 2])) != _fingerprint(
        ["a"], features, np.array([1.0, 2.0])
    )


def single():
    ranker = FeatureRanker("reg", models={"lr": LinearRegression()}, verbose=False)
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    return ranker, ranker.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=1)


def test_run_blocks_record_the_scheme():
    ranker, report = single()
    resampled = ranker.rank_resampled(
        X, Y, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1, random_state=0
    )
    selector = ConsensusSelector(
        "reg", "r2", k=2, models={"lr": LinearRegression()}, cv=3, n_repeats=1, random_state=0
    ).fit(X, Y)

    assert report["run"]["fingerprint_scheme"] == 2
    assert resampled["run"]["fingerprint_scheme"] == 2
    assert selector.report_["run"]["fingerprint_scheme"] == 2
    assert all(
        entry["report"]["run"]["fingerprint_scheme"] == 2 for entry in resampled["splits"]
    )
    json.dumps(report["run"], allow_nan=False)


def test_ablate_accepts_a_report_made_with_the_previous_scheme():
    labels = np.round(Y * 10).astype(np.int64)
    ranker = FeatureRanker("reg", models={"lr": LinearRegression()}, verbose=False)
    ranker.fit(X[TRAIN], labels[TRAIN], feature_names=NAMES)
    report = ranker.rank_features(X[EVAL], labels[EVAL], scoring="r2", n_repeats=1)
    run = {k: v for k, v in report["run"].items() if k != "fingerprint_scheme"}
    run["train_data_sha256"] = _fingerprint(NAMES, X[TRAIN], labels[TRAIN], scheme=1)
    run["data_sha256"] = _fingerprint(NAMES, X[EVAL], labels[EVAL], scheme=1)
    old = Report({**report, "run": run})
    assert run["train_data_sha256"] != report["run"]["train_data_sha256"]

    block = ranker.ablate(X[TRAIN], labels[TRAIN], X[EVAL], labels[EVAL], old, ks=[1])["ablation"]

    assert block["same_holdout"] is True
    with pytest.raises(ValueError, match="training data"):
        ranker.ablate(X[TRAIN], labels[TRAIN] + 1, X[EVAL], labels[EVAL], old, ks=[1])


def test_a_ranker_pickled_before_scheme_2_ranks_and_ablates_with_scheme_1():
    labels = np.round(Y * 10).astype(np.int64)
    ranker = FeatureRanker("reg", models={"lr": LinearRegression()}, verbose=False)
    ranker.fit(X[TRAIN], labels[TRAIN], feature_names=NAMES)
    assert ranker.fingerprint_scheme_ == 2
    ranker.train_data_sha256_ = _fingerprint(NAMES, X[TRAIN], labels[TRAIN], scheme=1)
    del ranker.__dict__["fingerprint_scheme_"]

    loaded = pickle.loads(pickle.dumps(ranker))
    report = loaded.rank_features(X[EVAL], labels[EVAL], scoring="r2", n_repeats=1)

    assert loaded.fingerprint_scheme_ == 1
    assert report["run"]["fingerprint_scheme"] == 1
    assert report["run"]["data_sha256"] == _fingerprint(NAMES, X[EVAL], labels[EVAL], scheme=1)
    block = loaded.ablate(X[TRAIN], labels[TRAIN], X[EVAL], labels[EVAL], report, ks=[1])["ablation"]
    assert block["same_holdout"] is True


# -- ablation provenance and paired curve ------------------------------------


class FailsOnOneColumn(RegressorMixin, BaseEstimator):
    """Raises on fit when it gets a single column."""

    def fit(self, X, y):
        if np.shape(X)[1] < 2:
            raise ValueError("needs two columns")
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def ablation_setup(models, eval_weight=None):
    ranker = FeatureRanker("reg", models=models, verbose=False)
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    report = ranker.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0, sample_weight=eval_weight
    )
    return ranker, report


def ablate(ranker, report, evaluation=EVAL, **kwargs):
    return ranker.ablate(X[TRAIN], Y[TRAIN], X[evaluation], Y[evaluation], report, **kwargs)


def test_provenance_keys_and_same_holdout():
    ranker, report = ablation_setup({"lr": LinearRegression()})

    block = ablate(ranker, report, ks=[1, 2])["ablation"]

    assert block["same_holdout"] is True
    assert block["data_sha256"] == report["run"]["data_sha256"]
    assert block["weights_sha256"] is None
    assert block["n_eval_rows"] == N - N_TRAIN
    assert block["fingerprint_scheme"] == 2
    assert list(block)[:8] == [
        "scoring", "models", "baseline_score", "n_groups", "fingerprint_scheme",
        "data_sha256", "weights_sha256", "n_eval_rows",
    ]


def test_a_third_data_set_is_not_the_holdout():
    ranker, report = ablation_setup({"lr": LinearRegression()})
    third = slice(0, 90)

    block = ablate(ranker, report, third, ks=[1])["ablation"]

    assert block["same_holdout"] is False
    assert block["data_sha256"] == _fingerprint(NAMES, X[third], Y[third])
    assert block["n_eval_rows"] == 90


def test_weights_are_fingerprinted_and_decide_same_holdout():
    ranker, report = ablation_setup({"lr": LinearRegression()}, eval_weight=W[EVAL])

    block = ablate(ranker, report, ks=[1], eval_sample_weight=W[EVAL])["ablation"]

    assert block["same_holdout"] is True
    assert block["weights_sha256"] == report["run"]["weights_sha256"]
    # The same rows with other weights are refused, as before.
    with pytest.raises(ValueError, match="eval_sample_weight"):
        ablate(ranker, report, ks=[1], eval_sample_weight=W[EVAL] + 1)


def test_other_weights_on_a_third_set_are_recorded_and_not_the_holdout():
    ranker, report = ablation_setup({"lr": LinearRegression()})
    third = slice(0, 90)

    block = ablate(ranker, report, third, ks=[1], eval_sample_weight=W[third])["ablation"]

    assert block["same_holdout"] is False
    assert len(block["weights_sha256"]) == 64


def cohort_result():
    models = {"lr": LinearRegression(), "boom": FailsOnOneColumn(), "boom2": FailsOnOneColumn()}
    ranker, report = ablation_setup(models)
    return ablate(ranker, report, ks=[1, 2, N_FEATURES])


def test_paired_curve_uses_the_models_scored_at_every_k():
    block = cohort_result()["ablation"]

    assert block["paired_models"] == ["lr"]
    assert [row["n_models_scored"] for row in block["curve"]] == [1, 3, 3]
    assert [row["paired_median_score"] for row in block["curve"]] == [
        row["scores"]["lr"] for row in block["curve"]
    ]
    json.dumps(block, allow_nan=False)


def test_no_paired_models_gives_null_paired_medians():
    ranker, report = ablation_setup({"boom": FailsOnOneColumn(), "boom2": FailsOnOneColumn()})

    block = ablate(ranker, report, ks=[1, 2])["ablation"]

    assert block["paired_models"] == []
    assert [row["n_models_scored"] for row in block["curve"]] == [0, 2]
    assert [row["paired_median_score"] for row in block["curve"]] == [None, None]


def test_summary_uses_the_paired_median_and_warns_about_the_cohort():
    result = cohort_result()
    findings = summarize(result)

    [curve] = [f for f in findings if f["code"] == "ablation_curve"]
    [cohort] = [f for f in findings if f["code"] == "ablation_cohort_changed"]
    assert cohort["severity"] == "warning"
    assert cohort["evidence"]["rows"] == [
        {"k": 1, "n_models_scored": 1, "models": ["lr"]},
        {"k": 2, "n_models_scored": 3, "models": ["lr", "boom", "boom2"]},
        {"k": N_FEATURES, "n_models_scored": 3, "models": ["lr", "boom", "boom2"]},
    ]
    paired = {row["k"]: row["paired_median_score"] for row in result["ablation"]["curve"]}
    best_k = min(paired, key=lambda k: (-paired[k], k))
    assert curve["evidence"]["best_k"] == best_k
    assert curve["evidence"]["best_median_score"] == paired[best_k]
    assert "paired" in curve["message"]
    json.dumps(findings, allow_nan=False)


def test_summary_without_paired_models_falls_back_to_the_median_and_says_so():
    ranker, report = ablation_setup({"boom": FailsOnOneColumn(), "boom2": FailsOnOneColumn()})
    result = ablate(ranker, report, ks=[1, 2])

    [curve] = [f for f in summarize(result) if f["code"] == "ablation_curve"]

    assert result["ablation"]["paired_models"] == []
    assert "unpaired" in curve["message"]


def test_summary_of_a_block_without_paired_models_does_not_claim_a_failure():
    result = cohort_result()
    block = {
        key: value for key, value in result["ablation"].items()
        if key not in ("paired_models", "data_sha256", "weights_sha256")
    }
    block["curve"] = [
        {k: v for k, v in row.items() if k not in ("paired_median_score", "n_models_scored")}
        for row in block["curve"]
    ]

    [curve] = [f for f in summarize(Report({**result, "ablation": block})) if f["code"] == "ablation_curve"]

    assert "predates the paired curve" in curve["message"]
    assert "no model scored at every k" not in curve["message"]


class FailsAtWidth(RegressorMixin, BaseEstimator):
    """Raises on fit when it gets exactly `width` columns."""

    def __init__(self, width=1):
        self.width = width

    def fit(self, X, y):
        if np.shape(X)[1] == self.width:
            raise ValueError("bad width")
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def test_cohort_warning_compares_model_sets_not_counts():
    models = {"lr": LinearRegression(), "a": FailsAtWidth(2), "b": FailsAtWidth(1)}
    ranker, report = ablation_setup(models)

    result = ablate(ranker, report, ks=[1, 2])
    cohort = [f for f in summarize(result) if f["code"] == "ablation_cohort_changed"]

    assert [row["n_models_scored"] for row in result["ablation"]["curve"]] == [2, 2]
    [finding] = cohort
    assert [row["models"] for row in finding["evidence"]["rows"]] == [["lr", "a"], ["lr", "b"]]
    assert "set of scored models changes" in finding["message"]


def test_stable_cohort_has_no_cohort_warning():
    ranker, report = ablation_setup({"lr": LinearRegression()})

    result = ablate(ranker, report, ks=[1, 2])

    assert "ablation_cohort_changed" not in [f["code"] for f in summarize(result)]


# -- zero-weight subsample -----------------------------------------------------

N_EVAL = N - N_TRAIN


def positive_only_outside_the_sample(size, random_state, n_rows=N_EVAL):
    """Weights positive on one row the subsample does not draw."""
    drawn = np.random.default_rng([random_state, 2]).choice(n_rows, size, replace=False)
    weights = np.zeros(n_rows)
    weights[next(i for i in range(n_rows) if i not in drawn)] = 1.0
    return weights


def fitted_ridge():
    ranker = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False)
    return ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)


def test_max_samples_with_all_zero_sampled_weights_raises_before_scoring():
    weights = positive_only_outside_the_sample(5, 0)

    with pytest.raises(ValueError, match="max_samples.*sample_weight"):
        fitted_ridge().rank_features(
            X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0,
            max_samples=5, sample_weight=weights,
        )


def test_max_samples_with_a_positive_sampled_weight_works():
    report = fitted_ridge().rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0,
        max_samples=5, sample_weight=np.ones(N_EVAL),
    )

    assert report["n_eval_rows"] == 5


def test_resampled_split_with_zero_sampled_weights_fails_naming_the_option():
    weights = np.concatenate([np.ones(N_TRAIN), positive_only_outside_the_sample(5, 0)])
    splits = [(np.arange(N_TRAIN), np.arange(N_TRAIN, N))]
    ranker = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False)

    with pytest.raises(RuntimeError, match="max_samples"):
        ranker.rank_resampled(
            X, Y, feature_names=NAMES, scoring="r2", cv=splits, n_repeats=1, random_state=0,
            max_samples=5, sample_weight=weights,
        )


def test_resampled_run_keeps_the_split_that_can_be_scored():
    # Split 0 evaluates rows 100..149 (random_state 0 draws its sample); split 1 rows 150..199.
    weights = np.ones(N)
    weights[100:150] = positive_only_outside_the_sample(5, 0, 50)
    splits = [(np.arange(100), np.arange(100, 150)), (np.arange(100), np.arange(150, N))]
    ranker = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False)

    report = ranker.rank_resampled(
        X, Y, feature_names=NAMES, scoring="r2", cv=splits, n_repeats=1, random_state=0,
        max_samples=5, sample_weight=weights,
    )

    [failure] = [f for f in report["failures"] if f["stage"] == "split"]
    assert failure["split"] == 0 and "max_samples" in failure["message"]
    assert [entry["split"] for entry in report["splits"]] == [1]


# -- dependency floor -----------------------------------------------------------


def test_pyproject_declares_the_scikit_learn_floor():
    text = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert re.search(r'"scikit-learn>=1\.5"', text)
