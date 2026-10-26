"""Regression tests for the 1.21.2 review fixes (OpenSpec change review-fixes)."""
import copy
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker
from featranker.importance import _resolve_splits

RNG = np.random.default_rng(17)
X = RNG.normal(size=(180, 3)) + [100, 200, 300]
Y = 3 * X[:, 0] + 2 * X[:, 1] + 0.1 * RNG.normal(size=180)
NAMES = ["a", "b", "c"]


def fit_ranker(models, **kwargs):
    return FeatureRanker("reg", models=models, verbose=False, **kwargs).fit(
        X[:120], Y[:120], feature_names=NAMES
    )


def rank(ranker, **kwargs):
    kwargs.setdefault("random_state", 0)
    return ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=2, **kwargs)


def score_of(report, name):
    return report["models"][name]["evaluation_score"]


# F1: models cannot corrupt shared inputs


def test_in_place_estimator_before_ridge_leaves_ridge_unchanged():
    alone = score_of(rank(fit_ranker({"clean": Ridge()})), "clean")
    X_before = X.copy()

    report = rank(fit_ranker({"mutator": LinearRegression(copy_X=False), "clean": Ridge()}))

    assert score_of(report, "clean") == alone
    assert np.array_equal(X, X_before)
    failed = {f["model"] for f in report["failures"] if f["stage"] == "fit"}
    assert failed <= {"mutator"}
    assert "mutator" in failed or score_of(report, "mutator") > 0.9


class WritesToX(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        X[:, 0] = 0.0
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_estimator_that_writes_to_x_is_a_fit_failure_and_others_are_unchanged():
    alone = score_of(rank(fit_ranker({"clean": Ridge()})), "clean")
    original = X.copy()

    report = rank(fit_ranker({"writer": WritesToX(), "clean": Ridge()}))

    assert score_of(report, "clean") == alone
    assert [f["model"] for f in report["failures"] if f["stage"] == "fit"] == ["writer"]
    assert np.array_equal(X, original)


def test_callers_arrays_stay_writable():
    features, target = X.copy(), Y.copy()
    FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadows=True, shadow_seed=0).fit(
        features, target, feature_names=NAMES
    )
    assert features.flags.writeable and target.flags.writeable


def nocopy_models():
    return {
        "lr": LinearRegression(copy_X=False),
        "lasso": Lasso(alpha=0.01, copy_X=False),
        "scaler": make_pipeline(StandardScaler(copy=False), Ridge()),
    }


@pytest.mark.parametrize("mode", ["array", "dataframe"])
def test_in_place_models_cannot_corrupt_ridge_in_either_input_mode(mode):
    if mode == "dataframe":
        pytest.importorskip("pandas")
    alone = score_of(rank(fit_ranker({"clean": Ridge()}, model_input=mode)), "clean")

    report = rank(fit_ranker({**nocopy_models(), "clean": Ridge()}, model_input=mode))

    assert score_of(report, "clean") == alone
    assert not [f for f in report["failures"] if f["model"] == "clean"]


def test_dataframe_input_cannot_be_made_writable_again():
    pytest.importorskip("pandas")
    from featranker.evidence import _model_X

    frame = _model_X(X.copy(), NAMES)
    values = np.asarray(frame)
    assert not values.flags.writeable
    with pytest.raises(ValueError):
        values.setflags(write=True)


def test_shap_evidence_models_cannot_corrupt_each_other():
    pytest.importorskip("shap")

    def ridge_shap(models):
        ranker = FeatureRanker("reg", models=models, verbose=False, evidence=True)
        ranker.fit(X[:120], Y[:120], feature_names=NAMES)
        report = rank(ranker)
        return {g: v["mean_abs"] for g, v in report["models"]["ridge"]["shap"]["groups"].items()}

    alone = ridge_shap({"ridge": Ridge()})
    after = ridge_shap({"ss": make_pipeline(StandardScaler(copy=False), Ridge()), "ridge": Ridge()})
    assert after == alone


# F2: shadow rules need shadow evidence


@pytest.mark.parametrize(
    "rule", [{"min_shadow_share": 0.5}, {"max_shadow_q": 0.05}, {"min_shadow_families": 1}]
)
def test_shadow_rule_without_shadow_evidence_raises_value_error(rule):
    drop = make_pipeline(ColumnTransformer([("real", "passthrough", [0, 1, 2])]), Ridge())
    selector = ConsensusSelector(
        "reg", "r2", models={"drop": drop}, cv=3, n_repeats=2,
        random_state=0, shadow_seed=0, **rule,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=next(iter(rule))) as info:
            selector.fit(X, Y)
    assert "shadow columns" in str(info.value)


# F3: ablation checks model parameters


def test_ablate_rejects_a_model_with_other_parameters_under_the_same_name():
    report = rank(fit_ranker({"m": Ridge(alpha=1)}))
    other = fit_ranker({"m": Ridge(alpha=1e9)})
    with pytest.raises(ValueError, match="'m'"):
        other.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[3])


def test_ablate_accepts_matching_parameters():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    out = ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], rank(ranker), ks=[3])
    assert out["ablation"]["curve"][0]["scores"]["m"] is not None


def doctored(report, version, params):
    other = copy.deepcopy(report)
    other["provenance"]["scikit-learn"] = version
    other["models"]["m"]["params"] = params
    return other


def test_ablate_compares_only_shared_params_across_scikit_learn_versions():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = rank(ranker)
    params = report["models"]["m"]["params"]
    older = doctored(report, "0.0.1", {**params, "removed_in_new_version": 1})
    del older["models"]["m"]["params"]["tol"]

    out = ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], older, ks=[3])
    assert out["ablation"]["curve"][0]["scores"]["m"] is not None


def test_ablate_warns_about_a_shared_param_that_differs_across_versions():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = rank(ranker)
    older = doctored(report, "0.0.1", {**report["models"]["m"]["params"], "alpha": 5})

    with pytest.warns(UserWarning, match=r"0\.0\.1.*'m'"):
        out = ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], older, ks=[3])
    assert out["ablation"]["curve"]


def test_ablate_tolerates_a_null_provenance():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = rank(ranker)
    report["provenance"] = None
    ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[3])


def test_ablate_rejects_a_differing_param_under_the_same_version():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = rank(ranker)
    same = doctored(
        report, report["provenance"]["scikit-learn"],
        {**report["models"]["m"]["params"], "alpha": 5},
    )
    with pytest.raises(ValueError, match="'m'"):
        ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], same, ks=[3])


def test_ablate_rejects_a_missing_param_under_the_same_version():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = rank(ranker)
    same = doctored(report, report["provenance"]["scikit-learn"], {"alpha": 1})

    with pytest.raises(ValueError, match="'m'"):
        ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], same, ks=[3])


# F4: repeated splits are rejected


def test_rank_resampled_rejects_a_repeated_split_before_fitting():
    split = (np.arange(120), np.arange(120, 180))
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadows=True, shadow_seed=0)
    with pytest.raises(ValueError, match="identical"):
        ranker.rank_resampled(X, Y, feature_names=NAMES, scoring="r2", cv=[split, split])
    assert not ranker.is_fitted_


def test_repeated_split_message_names_a_possible_cause():
    split = (np.arange(10), np.arange(10, 20))
    with pytest.raises(ValueError, match="by chance"):
        _resolve_splits([split, split], np.zeros((20, 1)), np.zeros(20), None, "reg", 0)


def test_distinct_splits_are_kept():
    first = (np.arange(60), np.arange(60, 120))
    second = (np.arange(60, 120), np.arange(60))
    assert len(_resolve_splits([first, second], X, Y, None, "reg", 0)) == 2


# B-a: experimental shadow q


def test_max_shadow_q_warns_and_marks_the_shadow_test_experimental():
    selector = ConsensusSelector(
        "reg", "r2", models={"m": Ridge()}, cv=3, n_repeats=2, random_state=0,
        shadow_seed=0, max_shadow_q=0.05,
    )
    with pytest.warns(UserWarning, match="experimental"):
        selector.fit(X, Y)
    assert selector.report_["shadow_test"]["experimental"] is True


def test_other_shadow_rules_do_not_warn_experimental():
    selector = ConsensusSelector(
        "reg", "r2", models={"m": Ridge()}, cv=3, n_repeats=2, random_state=0,
        shadow_seed=0, min_shadow_share=0.5,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*experimental")
        selector.fit(X, Y)


# F5-a: opaque transformers are unverified


def shadow_failures(pipeline):
    report = rank(fit_ranker({"m": pipeline}, shadows=True, shadow_seed=0))
    return [f for f in report["failures"] if f["stage"] == "shadows"]


class OpaqueTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :3]


def test_function_transformer_that_drops_shadows_is_unverified():
    failures = shadow_failures(make_pipeline(FunctionTransformer(lambda a: a[:, :3]), Ridge()))
    assert len(failures) == 1 and "could not be verified" in failures[0]["message"]


def test_transformer_defined_outside_scikit_learn_is_unverified():
    failures = shadow_failures(make_pipeline(OpaqueTransformer(), Ridge()))
    assert len(failures) == 1 and "could not be verified" in failures[0]["message"]


def test_scikit_learn_scaler_stays_verified():
    assert shadow_failures(make_pipeline(StandardScaler(), Ridge())) == []


def test_final_predictor_from_any_library_stays_verified():
    assert shadow_failures(make_pipeline(WritesNothing())) == []


class WritesNothing(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def drop_first_three():
    return FunctionTransformer(lambda a: np.asarray(a)[:, :3])


OPAQUE_MODELS = {
    "target_regressor_with_function_transformer_inside": lambda: TransformedTargetRegressor(
        regressor=make_pipeline(drop_first_three(), Ridge()),
        transformer=FunctionTransformer(np.log1p, np.expm1),
    ),
    "scaler_then_function_transformer": lambda: make_pipeline(
        StandardScaler(), drop_first_three(), Ridge()
    ),
    "polynomial_then_function_transformer": lambda: make_pipeline(
        PolynomialFeatures(1, include_bias=False), drop_first_three(), Ridge()
    ),
    "imputer_indicator_then_function_transformer": lambda: make_pipeline(
        SimpleImputer(add_indicator=True), drop_first_three(), Ridge()
    ),
    "polynomial_then_custom": lambda: make_pipeline(
        PolynomialFeatures(1, include_bias=False), OpaqueTransformer(), Ridge()
    ),
    "target_regressor": lambda: TransformedTargetRegressor(
        regressor=make_pipeline(drop_first_three(), Ridge())
    ),
    "bagging": lambda: BaggingRegressor(
        make_pipeline(drop_first_three(), Ridge()), n_estimators=2
    ),
    "grid_search": lambda: GridSearchCV(
        make_pipeline(OpaqueTransformer(), Ridge()), {"ridge__alpha": [1.0]}, cv=2
    ),
}


def test_target_transformer_of_a_target_regressor_is_not_opaque():
    target_only = TransformedTargetRegressor(
        Ridge(), transformer=FunctionTransformer(np.log1p, np.expm1)
    )
    assert shadow_failures(target_only) == []


@pytest.mark.parametrize("name", sorted(OPAQUE_MODELS))
def test_opaque_transformer_anywhere_in_a_model_is_unverified(name):
    failures = shadow_failures(OPAQUE_MODELS[name]())
    assert len(failures) == 1 and "could not be verified" in failures[0]["message"]


# F7: NumPy integers keep reports strict JSON


def strict_json(report):
    json.dumps(report, allow_nan=False)


def test_rank_features_with_numpy_integers_is_strict_json():
    ranker = fit_ranker({"m": Ridge()}, shadows=True, shadow_seed=np.int64(3))
    report = ranker.rank_features(
        X[120:], Y[120:], scoring="r2", n_repeats=np.int64(2),
        random_state=np.int64(0), max_samples=np.int64(40),
    )
    strict_json(report)
    assert type(report["random_state"]) is int and type(report["n_repeats"]) is int
    assert type(report["shadow_seed"]) is int


def test_fit_with_numpy_random_state_records_plain_int():
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False).fit(
        X[:120], Y[:120], feature_names=NAMES, max_samples=np.int64(60),
        random_state=np.int64(1),
    )
    strict_json(ranker.train_subsample_)
    report = rank(ranker)
    strict_json(report)
    assert type(report["run"]["train_random_state"]) is int


def test_rank_resampled_with_numpy_integers_is_strict_json():
    ranker = FeatureRanker(
        "reg", models={"m": Ridge()}, verbose=False, shadows=True, shadow_seed=np.int64(1)
    )
    report = ranker.rank_resampled(
        X, Y, feature_names=NAMES, scoring="r2", cv=np.int64(3), n_repeats=np.int64(2),
        random_state=np.int64(0), top_k=np.int64(2), max_samples=np.int64(30),
        max_train_samples=np.int64(60),
    )
    strict_json(report)
    for key in ("random_state", "n_repeats", "top_k", "shadow_seed"):
        assert type(report[key]) is int, key


def test_ablate_with_numpy_integers_is_strict_json():
    ranker = fit_ranker({"m": Ridge()})
    report = rank(ranker, random_state=np.int64(0))
    out = ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=np.array([1, 3]))
    strict_json(out)


def test_selector_with_numpy_integers_is_strict_json():
    selector = ConsensusSelector(
        "reg", "r2", models={"m": Ridge()}, k=np.int64(2), cv=3, n_repeats=np.int64(2),
        random_state=np.int64(0), shadow_seed=np.int64(0), min_shadow_share=0.0,
        max_samples=np.int64(50),
    )
    selector.fit(X, Y)
    strict_json(selector.report_)


def test_booleans_are_still_rejected():
    ranker = fit_ranker({"m": Ridge()})
    with pytest.raises(ValueError, match="n_repeats"):
        ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=True)
    with pytest.raises(TypeError, match="shadow_seed"):
        FeatureRanker("reg", models={"m": Ridge()}, shadow_seed=True)


# F8: CLI checks the output path first


def cli_argv(tmp_path, *extra):
    (tmp_path / "train.csv").write_text("a,target\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")
    (tmp_path / "eval.csv").write_text("a,target\n3,0\n4,1\n5,0\n6,1\n", encoding="utf-8")
    return [
        "featranker", "--task", "reg", "--scoring", "r2",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", *extra,
    ]


class ExplodingRanker:
    def __init__(self, *args, **kwargs):
        raise AssertionError("ranking must not start")


def test_cli_rejects_a_missing_output_folder_before_ranking(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_module, "FeatureRanker", ExplodingRanker)
    missing = tmp_path / "missing" / "dir" / "report.json"
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--output", str(missing)))

    with pytest.raises(SystemExit) as info:
        cli_module.main()

    assert info.value.code == 2
    assert "missing" in capsys.readouterr().err


def test_cli_reports_a_write_error_as_a_usage_error(monkeypatch, tmp_path, capsys):
    blocked = tmp_path / "report.json"
    blocked.mkdir()
    (tmp_path / "train.csv").write_text("a,target\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_run_ranking", lambda args: {"ok": True})
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--output", str(blocked)))

    with pytest.raises(SystemExit) as info:
        cli_module.main()

    assert info.value.code == 2
    assert "report.json" in capsys.readouterr().err


@pytest.mark.parametrize("output", ["", "folder", "folder.json", "..", "."])
def test_cli_rejects_an_output_that_is_a_directory_or_empty_before_ranking(
    monkeypatch, tmp_path, capsys, output
):
    monkeypatch.setattr(cli_module, "FeatureRanker", ExplodingRanker)
    (tmp_path / "folder.json").mkdir()
    target = output if output in {"", "..", "."} else str(tmp_path / output)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--output", target))

    with pytest.raises(SystemExit) as info:
        cli_module.main()

    assert info.value.code == 2
    assert "output" in capsys.readouterr().err


# F6: benchmark tests are collected


def test_pytest_collects_the_benchmarks_folder():
    config = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not config.exists():
        pytest.skip("source checkout only")
    assert re.search(r'testpaths\s*=\s*\[[^\]]*"benchmarks"', config.read_text(encoding="utf-8"))
