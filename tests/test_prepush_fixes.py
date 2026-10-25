"""1.21.1 prepush-fixes: regressions found by Signal's pre-push review of 1.17-1.21."""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import featranker.evidence as evidence_module
import featranker.importance as importance_module
from featranker import FeatureRanker, Report, summarize
from featranker.importance import _fingerprint, _labels_fingerprint

N, N_TRAIN = 160, 80
RNG = np.random.default_rng(3)
X = RNG.normal(size=(N, 2))
Y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * RNG.normal(size=N)
NAMES = ["a", "b"]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, None)


# -- 1. a selector after an untracked column step is not certified -----------------


def shadow_failures(report):
    return [f for f in report["failures"] if f["stage"] == "shadows"]


def ranked_with(model):
    ranker = FeatureRanker(
        "reg", models={"m": model}, verbose=False, shadows=True, shadow_seed=0
    ).fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    return ranker.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=2, random_state=0)


def passthrough_all():
    return ColumnTransformer([("all", "passthrough", [0, 1, 2, 3])])


def test_selector_after_a_passthrough_column_transformer_is_not_certified():
    model = Pipeline([
        ("columns", passthrough_all()),
        ("select", SelectKBest(f_regression, k=2)),
        ("model", Ridge()),
    ])

    [failure] = shadow_failures(ranked_with(model))

    assert failure["model"] == "m" and "could not be verified" in failure["message"]


def test_feature_union_and_nested_selector_after_an_untracked_step_are_not_certified():
    from sklearn.pipeline import make_union

    union = make_pipeline(passthrough_all(), make_union("passthrough", "passthrough"), Ridge())
    nested = make_pipeline(
        passthrough_all(), make_pipeline(SelectKBest(f_regression, k=2), Ridge())
    )

    for model in (union, nested):
        [failure] = shadow_failures(ranked_with(model))
        assert "could not be verified" in failure["message"]


def test_column_free_steps_after_an_untracked_step_stay_verified():
    model = make_pipeline(passthrough_all(), Ridge())

    assert not shadow_failures(ranked_with(model))


def test_built_in_pipelines_stay_verified():
    for task, target in (("reg", Y), ("clf", (Y > 0).astype(int))):
        for frame in (False, True):
            ranker = FeatureRanker(
                task, "all", verbose=False, shadows=True, shadow_seed=0,
                **({"model_input": "dataframe"} if frame else {}),
            )
            ranker.fit(X[TRAIN], target[TRAIN], feature_names=NAMES)
            scoring = "r2" if task == "reg" else "accuracy"
            report = ranker.rank_features(
                X[EVAL], target[EVAL], scoring=scoring, n_repeats=1, random_state=0
            )
            assert not shadow_failures(report)


def test_built_in_categorical_pipelines_stay_verified():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "x": rng.normal(size=120),
        "c": rng.choice(list("abcdefgh"), size=120),
    })
    target = frame["x"].to_numpy() + (frame["c"] == "a").to_numpy()
    for cap in (None, 3):
        ranker = FeatureRanker(
            "reg", "all", verbose=False, shadows=True, shadow_seed=0,
            categorical=["c"], max_categories=cap,
        ).fit(frame[:60], target[:60])
        report = ranker.rank_features(
            frame[60:], target[60:], scoring="r2", n_repeats=1, random_state=0
        )
        assert not shadow_failures(report)


def selecting_regressor():
    return make_pipeline(SelectKBest(f_regression, k=2), Ridge())


class WrappedModel(TransformerMixin, BaseEstimator):
    """A transformer that hides a selecting model among its parameters."""

    def __init__(self, inner=None):
        self.inner = inner

    def fit(self, X, y=None):
        self.inner_ = self.inner.fit(X, y)
        return self

    def transform(self, X):
        return self.inner_.predict(X).reshape(-1, 1)


def wrapped_selecting_models():
    wrapped = TransformedTargetRegressor(regressor=selecting_regressor())
    return {
        "meta-estimator as the model": wrapped,
        "meta-estimator after a scaler": make_pipeline(StandardScaler(), wrapped),
        "bagging of a selecting pipeline": BaggingRegressor(
            selecting_regressor(), n_estimators=2, random_state=0
        ),
        "wrapped in a feature union part": make_pipeline(
            FeatureUnion([("w", WrappedModel(selecting_regressor()))]), Ridge()
        ),
        "wrapped in a column transformer branch": make_pipeline(
            ColumnTransformer([("w", WrappedModel(selecting_regressor()), [0, 1, 2, 3])]),
            Ridge(),
        ),
    }


@pytest.mark.parametrize("label", list(wrapped_selecting_models()))
def test_selectors_wrapped_in_meta_estimators_are_not_certified(label):
    model = wrapped_selecting_models()[label]

    [failure] = shadow_failures(ranked_with(model))

    assert "could not be verified" in failure["message"]


# -- 2. only scikit-learn's own scorer classes share predictions -------------------


def clipped_mse_scorer(floor=-0.5):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    class Clipped(type(scorer)):
        def __call__(self, *args, **kwargs):
            return max(super().__call__(*args, **kwargs), floor)

    scorer.__class__ = Clipped  # same state, overridden __call__
    return scorer


def rank_with_scoring(scoring):
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False)
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    return ranker.rank_features(
        X[EVAL], Y[EVAL], scoring=scoring, n_repeats=3, random_state=0
    )


def test_a_scorer_subclass_that_overrides_call_is_not_shared(monkeypatch):
    scoring = ["r2", clipped_mse_scorer()]
    cached = rank_with_scoring(scoring)
    with monkeypatch.context() as patch:
        patch.setattr(importance_module, "_MultimetricScorer", None)
        uncached = rank_with_scoring(scoring)

    assert dict(cached) == dict(uncached)


def test_native_scorers_still_share_and_subclasses_are_direct():
    native = make_scorer(mean_squared_error, greater_is_better=False)

    assert importance_module._share_key(native, True, False) is None
    assert importance_module._share_key(clipped_mse_scorer(), True, False) is importance_module._DIRECT


# -- 3. ablate() checks the training rows in both directions ----------------------


def fitted(**fit_options):
    ranker = FeatureRanker("reg", models={"m": LinearRegression()}, verbose=False)
    return ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, **fit_options)


def rank(ranker):
    return ranker.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0)


def ablate(ranker, report):
    return ranker.ablate(X[TRAIN], Y[TRAIN], X[EVAL], Y[EVAL], report, ks=[1])


def test_full_report_with_a_later_subsampled_refit_is_refused():
    ranker = fitted()
    report = rank(ranker)
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, max_samples=10, random_state=0)

    with pytest.raises(ValueError, match="training rows"):
        ablate(ranker, report)


def test_subsampled_report_with_a_later_full_refit_is_refused():
    ranker = fitted(max_samples=10, random_state=0)
    report = rank(ranker)
    assert "train_rows_sha256" in report["run"]
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)

    with pytest.raises(ValueError, match="training rows"):
        ablate(ranker, report)


def test_matching_training_rows_are_accepted():
    for options in ({}, {"max_samples": 10, "random_state": 0}):
        ranker = fitted(**options)
        assert ablate(ranker, rank(ranker))["ablation"]["curve"]


def pre_1_21_report(ranker):
    report = rank(ranker)
    run = {k: v for k, v in report["run"].items() if k != "fingerprint_scheme"}
    run.pop("train_rows_sha256", None)
    run["train_data_sha256"] = _fingerprint(NAMES, X[TRAIN], Y[TRAIN], scheme=1)
    run["data_sha256"] = _fingerprint(NAMES, X[EVAL], Y[EVAL], scheme=1)
    return Report({**report, "run": run})


def test_report_before_scheme_keys_is_refused_for_a_subsampled_ranker():
    old = pre_1_21_report(fitted())  # no version before 1.21 could subsample training rows
    subsampled = fitted(max_samples=10, random_state=0)

    with pytest.raises(ValueError, match="training rows"):
        ablate(subsampled, old)


def test_report_before_scheme_keys_is_accepted_for_a_full_ranker():
    full = fitted()

    assert ablate(full, pre_1_21_report(full))["ablation"]["curve"]


# -- 4. shuffle-label fingerprints follow the ranker's scheme ---------------------


def test_scheme_1_ranker_hashes_shuffle_labels_with_scheme_1():
    train_groups = np.repeat(np.arange(N_TRAIN // 4), 4)
    eval_groups = np.repeat(np.arange((N - N_TRAIN) // 4), 4)
    ranker = FeatureRanker(
        "reg", models={"m": LinearRegression()}, verbose=False, shuffle="within_groups"
    )
    ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, shuffle_groups=train_groups)
    ranker.train_data_sha256_ = _fingerprint(NAMES, X[TRAIN], Y[TRAIN], scheme=1)
    ranker.train_shuffle_groups_sha256_ = _labels_fingerprint(train_groups, 1)
    del ranker.__dict__["fingerprint_scheme_"]

    loaded = pickle.loads(pickle.dumps(ranker))
    run = loaded.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=1, random_state=0, shuffle_groups=eval_groups
    )["run"]

    assert run["fingerprint_scheme"] == 1
    assert run["train_shuffle_groups_sha256"] == _labels_fingerprint(train_groups, 1)
    assert run["shuffle_groups_sha256"] == _labels_fingerprint(eval_groups, 1)
    assert _labels_fingerprint(eval_groups, 1) != _labels_fingerprint(eval_groups, 2)


def test_labels_fingerprint_defaults_to_the_current_scheme():
    labels = np.arange(5)

    assert _labels_fingerprint(labels) == _labels_fingerprint(labels, 2)
    assert _labels_fingerprint(None, 1) is None


# -- 6. Spearman keeps the dtype of the ranks -------------------------------------


def float32_ranks(values, axis=0):
    """What SciPy 1.18 returns for float32 input: float32 ranks."""
    return rankdata(values, axis=axis).astype(np.float32)


def old_spearman(values, ranker):
    centered = ranker(values, axis=0) - (values.shape[0] + 1) / 2
    norm = np.sqrt((centered**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(centered.T @ centered / np.outer(norm, norm), -1.0, 1.0)


@pytest.mark.parametrize("ranks", [rankdata, float32_ranks], ids=["scipy", "float32 ranks"])
@pytest.mark.parametrize("shape", [(50, 4), (2000, 300)])  # one block, several blocks
def test_spearman_equals_the_previous_construction_on_float32(monkeypatch, ranks, shape):
    values = np.random.default_rng(1).normal(size=shape).astype(np.float32)
    monkeypatch.setattr(evidence_module, "rankdata", ranks)

    new = evidence_module._spearman_matrix(values)
    old = old_spearman(values, ranks)

    assert new.dtype == old.dtype
    assert np.array_equal(new, old)


# -- 7. a resampled constant_within_groups finding names the splits ---------------

N_GROUPS, GROUP_SIZE = 15, 4
PATIENTS = np.repeat(np.arange(N_GROUPS), GROUP_SIZE)
ROW_RNG = np.random.default_rng(5)
PATIENT_LEVEL = np.repeat(ROW_RNG.normal(size=N_GROUPS), GROUP_SIZE)
X_PAT = np.column_stack([PATIENT_LEVEL, ROW_RNG.normal(size=len(PATIENTS))])
# Patients 10..14 also vary inside the patient, so "p" is constant within groups
# on the rows of split 0 (patients 0..4) and not on those of split 1 (patients 10..14).
X_PAT[PATIENTS >= 10, 0] += ROW_RNG.normal(size=(PATIENTS >= 10).sum())
Y_PAT = 2.0 * X_PAT[:, 0] + X_PAT[:, 1] + 0.1 * ROW_RNG.normal(size=len(PATIENTS))


def resampled_with_one_constant_split():
    rows = np.arange(len(PATIENTS))
    splits = [
        (rows[PATIENTS >= 5], rows[PATIENTS < 5]),
        (rows[PATIENTS < 10], rows[PATIENTS >= 10]),
    ]
    ranker = FeatureRanker(
        "reg", models={"m": LinearRegression()}, verbose=False, shuffle="within_groups"
    )
    with pytest.warns(UserWarning, match="constant within"):
        return ranker.rank_resampled(
            X_PAT, Y_PAT, scoring="r2", cv=splits, groups=PATIENTS, feature_names=["p", "v"],
            n_repeats=1, random_state=0,
        )


@pytest.mark.filterwarnings("ignore:No model beat")
def test_resampled_finding_names_the_splits_where_a_group_was_constant():
    result = resampled_with_one_constant_split()
    first, second = (entry["report"] for entry in result["splits"])
    assert first["constant_within_groups"] == ["p"] and second["constant_within_groups"] == []
    assert first["models"]["m"]["importance"]["p"]["mean"] == 0.0

    [finding] = [f for f in summarize(result) if f["code"] == "constant_within_groups"]

    assert finding["evidence"] == {"groups": ["p"], "splits": {"p": [0]}}
    assert "in at least one split" in finding["message"]
    assert "in those splits" in finding["message"]


# -- 8. the category cap makes the infrequent column at equality -------------------


def unseen_row(levels, max_categories):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, levels, 300)
    frame = pd.DataFrame({"x": rng.normal(size=300), "c": codes.astype(str)})
    ranker = FeatureRanker(
        "reg", "fast", verbose=False, categorical=["c"], max_categories=max_categories
    ).fit(frame, (codes % 3).astype(float) + 0.1 * rng.normal(size=300))
    names = set(ranker.models["reg"]["linear"])
    model = next(m for n, m in ranker.trained_models.items() if n in names)
    encoder = model.steps[0][1].named_transformers_["cat"]
    return encoder.transform(np.array([[1e6]])).toarray()[0]


def test_unseen_level_joins_the_infrequent_column_at_exactly_max_categories_levels():
    row = unseen_row(levels=3, max_categories=3)

    assert row.sum() == 1 and row[-1] == 1


def test_unseen_level_is_all_zeros_below_max_categories_levels():
    assert unseen_row(levels=2, max_categories=3).sum() == 0
