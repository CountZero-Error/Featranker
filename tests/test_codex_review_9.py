"""Acceptance check for codex-review-9 (lead-written; builder must not edit)."""
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from featranker import ConsensusSelector, FeatureRanker
from test_codex_review_7 import load_run

pd = pytest.importorskip("pandas")
DURATION = np.timedelta64(2, "ns")


def data(task="reg"):
    rng = np.random.default_rng(0)
    x, c = rng.normal(size=300), rng.integers(0, 20, 300)
    y = x + c % 5 + 0.1 * rng.normal(size=300)
    frame = pd.DataFrame({"x": x, "c": [f"l{i:02d}" for i in c]})
    return frame, y if task == "reg" else (y > 2).astype(int)


def fitted(task="reg", group="fast", **kwargs):
    X, y = data(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker = FeatureRanker(task, group, categorical=["c"], verbose=False, **kwargs)
        return ranker.fit(X[:200], y[:200]), X, y


def full_k(ranker, X, y, change=lambda ranker: None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ranker.rank_features(X[200:], y[200:], scoring="r2" if ranker.task == "reg" else "accuracy",
                                      n_repeats=1)
        change(ranker)
        curve = ranker.ablate(X[:200], y[:200], X[200:], y[200:], report, ks=[2])["ablation"]["curve"]
    return {n: (s, report["models"][n]["evaluation_score"]) for n, s in curve[0]["scores"].items()
            if n in ("ridge_regression", "linear_regression", "logistic_regression", "lasso", "elastic_net")}


# ablation


@pytest.mark.parametrize("task", ["reg", "clf"])
@pytest.mark.parametrize("fit_cap,later", [(None, 2), (3, None), (3, 10)])
def test_cap_assigned_after_fit_is_not_used(task, fit_cap, later):
    ranker, X, y = fitted(task, max_categories=fit_cap)
    scores = full_k(ranker, X, y, lambda r: setattr(r, "max_categories", later))
    assert scores
    for score, reported in scores.values():
        assert score == pytest.approx(reported, abs=1e-9)
    assert ranker.max_categories == later


def test_refit_after_assigning_uses_the_new_cap():
    ranker, X, y = fitted()
    ranker.max_categories = 2
    ranker.fit(X[:200], y[:200])
    for score, reported in full_k(ranker, X, y).values():
        assert score == pytest.approx(reported, abs=1e-9)


def test_cap_assigned_to_a_pickled_impute_ranker_is_not_used():
    X, y = data()
    X = X.copy()
    X.loc[::7, "x"] = np.nan
    ranker = FeatureRanker("reg", "fast", categorical=["c"], missing="impute", verbose=False)
    ranker.fit(X[:200], y[:200])
    report = ranker.rank_features(X[200:], y[200:], scoring="r2", n_repeats=1)
    baseline = ranker.ablate(X[:200], y[:200], X[200:], y[200:], report, ks=[2])["ablation"]["curve"]
    pickled = pickle.loads(pickle.dumps(ranker))
    pickled.max_categories = 2
    again = pickled.ablate(X[:200], y[:200], X[200:], y[200:], report, ks=[2])["ablation"]["curve"]
    assert again[0]["scores"]["ridge_regression"] == baseline[0]["scores"]["ridge_regression"]


@pytest.mark.parametrize("task", ["reg", "clf"])
def test_dense_fit_is_refitted_dense_for_a_2x_report(task):
    # What a ranker pickled by 1.21 looks like: built-in pipelines with sparse_threshold 0.
    ranker, X, y = fitted(task)
    from featranker import importance

    original = importance._FORCE_DENSE_ONE_HOT
    importance._FORCE_DENSE_ONE_HOT = True
    try:
        dense, _, _ = fitted(task)
    finally:
        importance._FORCE_DENSE_ONE_HOT = original
    seen = []
    real = FeatureRanker.fit

    def spy(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        seen.append({m.get_params().get("columntransformer__sparse_threshold")
                     for m in self.trained_models.values()} - {None})
        return out

    FeatureRanker.fit = spy
    try:
        scores = full_k(dense, X, y)
        dense_seen, seen[:] = list(seen), []
        full_k(ranker, X, y)
    finally:
        FeatureRanker.fit = real
    assert dense_seen == [{0}] and 1.0 in seen[0]
    for score, reported in scores.values():
        assert score == pytest.approx(reported, abs=1e-12)


def test_ranker_pickled_without_recorded_settings():
    ranker, X, y = fitted(max_categories=4)
    known = {"task", "group", "n_jobs", "verbose", "shadows", "shadow_seed", "shadow_mode", "evidence", "missing",
             "model_input", "shuffle", "families", "max_categories", "categorical"}
    state = {k: v for k, v in vars(ranker).items()
             if k in known or not k.startswith("_")}
    old = FeatureRanker.__new__(FeatureRanker)
    old.__setstate__(state)
    for score, reported in full_k(old, X, y).values():
        assert score == pytest.approx(reported, abs=1e-9)


# SHAP


class Reversed(DecisionTreeRegressor):
    def fit(self, X, y, **kwargs):
        return super().fit(np.asarray(X)[:, ::-1], y, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(np.asarray(X)[:, ::-1], **kwargs)


class Forest(RandomForestRegressor):
    pass


def evidence(model):
    Z = np.random.default_rng(93).normal(size=(150, 2))
    ranker = FeatureRanker("reg", models={"m": model}, evidence=True, verbose=False)
    ranker.fit(Z[:100], 3 * Z[:100, 0], feature_names=["a", "b"])
    report = ranker.rank_features(Z[100:], 3 * Z[100:, 0], scoring="r2", n_repeats=1)
    return report["evidence"]["shap"], report["models"]["m"].get("shap")


@pytest.mark.parametrize("wrap", [lambda m: m, lambda m: make_pipeline(StandardScaler(), m)])
@pytest.mark.parametrize("model", [Reversed(max_depth=2), Forest(n_estimators=3)])
def test_tree_subclasses_are_unsupported(wrap, model):
    pytest.importorskip("shap")
    block, entry = evidence(wrap(clone(model)))
    assert [u["model"] for u in block["unsupported"]] == ["m"] and entry is None


def test_library_trees_are_still_supported():
    pytest.importorskip("shap")
    models = [DecisionTreeRegressor(max_depth=2), RandomForestRegressor(n_estimators=3)]
    for package, name in (("xgboost", "XGBRegressor"), ("lightgbm", "LGBMRegressor"),
                          ("catboost", "CatBoostRegressor")):
        try:
            module = __import__(package)
        except Exception:
            continue
        options = {"iterations": 5, "verbose": False, "allow_writing_files": False} if package == "catboost" else {
            "n_estimators": 5}
        models.append(getattr(module, name)(**options))
    for model in models:
        block, entry = evidence(model)
        assert block["unsupported"] == [], type(model)
        assert entry["groups"]["a"]["mean_abs"] > entry["groups"]["b"]["mean_abs"]


@pytest.mark.parametrize("version", ["v0.46.0", "V0.48"])
def test_version_prefix_does_not_pass_the_floor(monkeypatch, version):
    shap = pytest.importorskip("shap")
    monkeypatch.setattr(shap, "__version__", version)
    with pytest.warns(UserWarning, match=r"0\.49"):
        block, entry = evidence(Ridge())
    assert block == {"available": False} and entry is None


def test_user_column_transformer_is_unsupported_whatever_its_encoder():
    pytest.importorskip("shap")
    built, X, y = fitted(group="linear", evidence=True)
    encoder = built.trained_models["ridge_regression"][0].transformers[0][1]
    report = built.rank_features(X[200:], y[200:], scoring="r2", n_repeats=1)
    assert "shap" in report["models"]["ridge_regression"]
    codes = np.c_[X["x"], X["c"].astype("category").cat.codes]
    user = make_pipeline(ColumnTransformer([("cat", clone(encoder), [1])], remainder="passthrough"), Ridge())
    for model in (user, clone(built.trained_models["ridge_regression"])):
        ranker = FeatureRanker("reg", models={"m": model}, evidence=True, verbose=False)
        ranker.fit(codes[:200], y[:200], feature_names=["x", "c"])
        report = ranker.rank_features(codes[200:], y[200:], scoring="r2", n_repeats=1)
        assert [u["model"] for u in report["evidence"]["shap"]["unsupported"]] == ["m"]


# integer options


X3 = np.random.default_rng(0).normal(size=(60, 3))


def plain(**kwargs):
    return FeatureRanker("reg", models={"r": Ridge()}, verbose=False, **kwargs)


def call_fit(**kwargs):
    plain().fit(X3, X3[:, 0], feature_names=list("abc"), **kwargs)


def call_rank(**kwargs):
    plain().fit(X3, X3[:, 0], feature_names=list("abc")).rank_features(
        X3, X3[:, 0], scoring="r2", **{"n_repeats": 1, **kwargs})


def call_resampled(**kwargs):
    plain().rank_resampled(X3, X3[:, 0], scoring="r2", feature_names=list("abc"), **{"n_repeats": 1, **kwargs})


def call_selector(**kwargs):
    ConsensusSelector("reg", "r2", models={"r": Ridge()}, **{"k": 1, "n_repeats": 1, **kwargs}).fit(X3, X3[:, 0])


@pytest.mark.parametrize("call,option", [
    (call_fit, "random_state"), (call_fit, "max_samples"),
    (call_rank, "random_state"), (call_rank, "n_repeats"), (call_rank, "max_samples"),
    (call_resampled, "random_state"), (call_resampled, "n_repeats"), (call_resampled, "top_k"),
    (call_resampled, "max_samples"), (call_resampled, "max_train_samples"), (call_resampled, "cv"),
    (call_selector, "random_state"), (call_selector, "shadow_seed"), (call_selector, "n_repeats"),
    (call_selector, "k"), (call_selector, "cv"), (call_selector, "min_shadow_families"),
    (call_selector, "min_shadow_share"), (call_selector, "max_shadow_q"), (call_selector, "max_samples"),
    (plain, "shadow_seed"), (plain, "max_categories"),
])
def test_durations_are_refused(call, option):
    with pytest.raises((TypeError, ValueError)):
        call(**{option: DURATION})


def test_duration_refused_for_ks():
    ranker = plain().fit(X3, X3[:, 0], feature_names=list("abc"))
    report = ranker.rank_features(X3, X3[:, 0], scoring="r2", n_repeats=1)
    with pytest.raises((TypeError, ValueError)):
        ranker.ablate(X3, X3[:, 0], X3, X3[:, 0], report, ks=[DURATION])


def test_numpy_integers_still_work():
    call_fit(random_state=np.uint64(3), max_samples=np.int32(30))
    call_rank(random_state=np.int8(3), n_repeats=np.int64(2), max_samples=np.int64(30))
    call_resampled(cv=np.int64(3), top_k=np.int16(2), n_repeats=np.int64(1), max_train_samples=np.float64(0.5))
    assert plain(max_categories=np.int64(5)).max_categories == 5


@pytest.mark.parametrize("huge", [2 ** 63, 10 ** 5000], ids=["2**63", "10**5000"])
def test_huge_caps_are_refused(huge):
    with pytest.raises(ValueError, match="max_categories"):
        plain(max_categories=huge)
    with pytest.raises(ValueError, match="min_shadow_families"):
        call_selector(min_shadow_families=huge)


class Untouchable:
    def __array__(self, *args, **kwargs):
        raise AssertionError("data touched")

    def __len__(self):
        raise AssertionError("data touched")

    def __getattr__(self, name):
        raise AssertionError("data touched")


@pytest.mark.parametrize("option,value", [
    ("random_state", -1), ("n_repeats", 2 ** 63), ("top_k", 2 ** 63), ("max_samples", 2 ** 63),
    ("max_train_samples", 0),
])
def test_rank_resampled_checks_options_first(option, value):
    with pytest.raises(ValueError, match="max_samples" if "samples" in option else option):
        plain().rank_resampled(Untouchable(), Untouchable(), scoring="r2", **{option: value})


@pytest.mark.parametrize("option,value", [("random_state", -1), ("max_samples", 2 ** 63)])
def test_fit_and_rank_check_options_first(option, value):
    with pytest.raises(ValueError, match=option):
        plain().fit(Untouchable(), Untouchable(), **{option: value})
    ranker = plain().fit(X3, X3[:, 0], feature_names=list("abc"))
    with pytest.raises(ValueError, match=option):
        ranker.rank_features(Untouchable(), Untouchable(), scoring="r2", **{option: value})


@pytest.mark.parametrize("option,value", [
    ("--n-repeats", 2 ** 63), ("--max-samples", 2 ** 63), ("--max-train-samples", 0), ("--random-state", -1),
    ("--max-categories", 10 ** 4000),
])
def test_cli_checks_options_before_reading_files(monkeypatch, capsys, option, value):
    from featranker import cli

    def fail(*args, **kwargs):
        raise AssertionError("file read")

    monkeypatch.setattr(cli, "_load_csv", fail)
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--train-data", "t.csv", "--eval-data", "e.csv", "--target", "y", "--task", "reg",
        "--group", "fast", "--scoring", "r2", f"{option}={value}"])
    with pytest.raises(SystemExit) as caught:
        sys.exit(cli.main())
    assert caught.value.code == 2
    assert "error: " in capsys.readouterr().err


# benchmark


@pytest.fixture
def bench(tmp_path, monkeypatch):
    run = load_run()
    values = np.random.default_rng(0).normal(size=(569, 30))
    y = np.r_[np.zeros(568, dtype=int), 1]  # unsplittable: error rows only, fast
    monkeypatch.setattr(run, "load", lambda *a, **k: (values, y, [f"f{i}" for i in range(30)]))
    out = tmp_path / "run.json"

    def main(*arguments, datasets="1510"):
        monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", datasets, "--out", str(out), *arguments])
        run.main()
        return json.loads(out.read_text())

    return run, out, main


def test_duplicate_seeds_and_datasets_are_refused(bench):
    _, out, main = bench
    with pytest.raises(SystemExit):
        main("--seeds", "1,1")
    with pytest.raises(SystemExit):
        main("--seeds", "0", datasets="1510,1510")
    assert not out.exists()
    main("--seeds", "0")
    with pytest.raises(SystemExit):
        main("--seeds", "1,01", "--resume")


@pytest.mark.parametrize("where", ["write_text", "replace"])
def test_failed_save_keeps_the_file(bench, monkeypatch, where):
    _, out, main = bench
    main("--seeds", "0")
    before = out.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    real_write = Path.write_text

    def partial(self, text, *args, **kwargs):
        real_write(self, text[:100])
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        if where == "replace":
            patch.setattr(os, "replace", fail)
            patch.setattr(Path, "replace", fail)
        else:
            patch.setattr(Path, "write_text", partial)
        with pytest.raises(OSError, match="disk full"):
            main("--seeds", "0,1", "--resume")
    assert out.read_bytes() == before
    assert [p.name for p in out.parent.iterdir()] == ["run.json"]
    assert {r["seed"] for r in main("--seeds", "0,1", "--resume")["rows"]} == {0, 1}


def test_resume_refuses_another_version_and_an_empty_commit(bench, monkeypatch):
    run, out, main = bench
    main("--seeds", "0", "--commit", "ec1dba5")
    before = out.read_bytes()
    with pytest.raises(SystemExit):
        main("--seeds", "0", "--resume", "--commit", "")
    real = run._featranker_info
    monkeypatch.setattr(run, "_featranker_info", lambda commit: {**real(commit), "version": "9.9.9"})
    with pytest.raises(SystemExit):
        main("--seeds", "0", "--resume")
    assert out.read_bytes() == before


def test_docs():
    usage = (Path(__file__).parents[1] / "docs" / "usage.md").read_text(encoding="utf-8")
    assert "permutation seed `random_state + i`." not in usage
    assert "an xgboost model whose `missing`" not in usage


# review round 1


@pytest.mark.parametrize("alias", ["boosting_type", "boosting"])
def test_lightgbm_random_forest_is_unsupported(alias):
    pytest.importorskip("shap")
    lightgbm = pytest.importorskip("lightgbm")
    forest = lightgbm.LGBMRegressor(n_estimators=5, bagging_freq=1, bagging_fraction=0.6, verbose=-1,
                                    **{alias: "rf"})
    block, entry = evidence(forest)
    assert [u["model"] for u in block["unsupported"]] == ["m"] and entry is None
    for boosting in ("gbdt", "dart"):
        block, entry = evidence(lightgbm.LGBMRegressor(n_estimators=5, boosting_type=boosting, verbose=-1))
        assert block["unsupported"] == [] and entry


@pytest.mark.parametrize("option,value", [
    ("cv", 1), ("cv", DURATION), ("cv", "5"), ("cv", 2.5), ("cv", True), ("scoring", "nope"), ("scoring", []),
])
def test_rank_resampled_checks_cv_and_scoring_first(option, value):
    with pytest.raises((TypeError, ValueError)):
        plain().rank_resampled(Untouchable(), Untouchable(), **{"scoring": "r2", option: value})


@pytest.mark.parametrize("option,value", [
    ("max_categories", 1), ("cv", 1), ("cv", DURATION), ("auto_groups_threshold", 0), ("scoring", "nope"),
    ("n_repeats", 0), ("shadow_seed", -1), ("shadow_mode", "x"), ("missing", "x"), ("shuffle", "x"),
    ("n_jobs", 0), ("feature_groups", "x"), ("feature_groups", 3),
])
def test_selector_checks_every_option_first(option, value):
    options = {"task": "reg", "scoring": "r2", "models": {"r": Ridge()}, "k": 1, "feature_groups": "auto",
               option: value}
    with pytest.raises((TypeError, ValueError)):
        ConsensusSelector(**options).fit(Untouchable(), Untouchable())


@pytest.mark.parametrize("n_jobs", [0, DURATION, "2", 1.5, True])
def test_bad_n_jobs_is_refused_first(n_jobs):
    with pytest.raises((TypeError, ValueError), match="n_jobs"):
        FeatureRanker("reg", models={"r": Ridge()}, n_jobs=n_jobs, verbose=False).fit(
            Untouchable(), Untouchable())


@pytest.mark.parametrize("n_jobs", [None, 1, np.int64(1), -1, -2])
def test_good_n_jobs(n_jobs):
    FeatureRanker("reg", models={"r": Ridge()}, n_jobs=n_jobs, verbose=False).fit(
        X3, X3[:, 0], feature_names=list("abc"))


@pytest.mark.parametrize("value", [DURATION, 1, 2 ** 63])
def test_max_categories_assigned_later_is_checked_by_fit(value):
    ranker = plain()
    ranker.max_categories = value
    with pytest.raises(ValueError, match="max_categories"):
        ranker.fit(Untouchable(), Untouchable())


def test_duration_split_indices_are_refused():
    indices = np.arange(60)
    splits = [(indices[:40].astype("m8[ns]"), indices[40:].astype("m8[ns]"))]
    with pytest.raises((TypeError, ValueError)):
        plain().rank_resampled(X3, X3[:, 0], scoring="r2", cv=splits, n_repeats=1, feature_names=list("abc"))
    plain().rank_resampled(X3, X3[:, 0], scoring="r2", cv=[(indices[:40], indices[40:])], n_repeats=1,
                           feature_names=list("abc"))


@pytest.mark.parametrize("arguments", [
    ["--n-jobs=0"], ["--scoring=zz"], ["--feature-groups=auto", "--auto-groups-threshold=5"],
])
def test_cli_checks_every_option_before_reading_files(monkeypatch, capsys, tmp_path, arguments):
    missing = str(tmp_path / "missing.csv")
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--train-data", missing, "--eval-data", missing, "--target", "y", "--task", "reg",
        "--group", "fast", "--scoring", "r2", *arguments])
    from featranker import cli

    with pytest.raises(SystemExit) as caught:
        sys.exit(cli.main())
    assert caught.value.code == 2
    assert "No such file" not in capsys.readouterr().err


def test_symlinked_output_keeps_the_link(bench):  # local: needs symlink rights
    _, out, main = bench
    target = out.parent / "real.json"
    target.write_text("{}")
    try:
        out.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available")
    main("--seeds", "0")
    assert out.is_symlink() and json.loads(target.read_text())["rows"]
    assert {p.name for p in out.parent.iterdir()} == {"run.json", "real.json"}


# review round 2


@pytest.mark.parametrize("name", ["average_output", "my_average_output_kw"])
def test_feature_name_does_not_look_like_a_random_forest(name):
    pytest.importorskip("shap")
    lightgbm = pytest.importorskip("lightgbm")
    Z = np.random.default_rng(93).normal(size=(150, 2))
    frame = pd.DataFrame(Z, columns=[name, "b"])
    for boosting, supported in (("gbdt", True), ("rf", False)):
        model = lightgbm.LGBMRegressor(n_estimators=5, boosting_type=boosting, bagging_freq=1,
                                       bagging_fraction=0.6, verbose=-1)
        ranker = FeatureRanker("reg", models={"m": model}, evidence=True, model_input="dataframe", verbose=False)
        ranker.fit(frame[:100], Z[:100, 0])
        report = ranker.rank_features(frame[100:], Z[100:, 0], scoring="r2", n_repeats=1)
        assert (report["evidence"]["shap"]["unsupported"] == []) is supported


@pytest.mark.parametrize("arguments,message", [
    (["--drop=y"], "--drop"), (["--categorical=y"], "categorical"),
    (["--drop=a", "--sample-weight=a"], "a"), (["--categorical=a", "--sample-weight=a"], "--categorical"),
    (["--sample-weight=y"], "--sample-weight"),
])
def test_cli_name_clashes_before_reading_files(monkeypatch, capsys, tmp_path, arguments, message):
    from featranker import cli

    missing = str(tmp_path / "missing.csv")
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--train-data", missing, "--eval-data", missing, "--target", "y", "--task", "reg",
        "--group", "fast", "--scoring", "r2", *arguments])
    with pytest.raises(SystemExit) as caught:
        sys.exit(cli.main())
    error = capsys.readouterr().err.split("error: ")[-1]
    assert caught.value.code == 2 and "No such file" not in error and message in error


@pytest.mark.parametrize("option,value", [("n_jobs", 0), ("max_categories", 1)])
def test_options_assigned_later_are_checked_by_rank_resampled(option, value):
    ranker = plain()
    setattr(ranker, option, value)
    with pytest.raises((TypeError, ValueError), match=option):
        ranker.rank_resampled(Untouchable(), Untouchable(), scoring="r2")
