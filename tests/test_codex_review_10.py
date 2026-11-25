"""Acceptance check for codex-review-10 (lead-written; builder must not edit)."""
import functools
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from featranker import FeatureRanker
from test_codex_review_7 import load_run

RNG = np.random.default_rng(0)
X = RNG.normal(size=(200, 3)) * [1, 30, 1]
X[:, 2] = X[:, 0] + 0.1 * X[:, 2]
Y = X[:, 0] + 0.01 * X[:, 1]
LABELS = (Y > 0.3).astype(int)
NAMES = ["a", "b", "c"]


def fit(model, task="reg", X=X, **kwargs):
    y = Y if task == "reg" else LABELS
    return FeatureRanker(task, models={"m": model}, verbose=False, **kwargs).fit(
        X[:120], y[:120], feature_names=NAMES)


def rank(ranker, task="reg", X=X):
    y = Y if task == "reg" else LABELS
    report = ranker.rank_features(X[120:], y[120:], scoring="r2" if task == "reg" else "accuracy", n_repeats=1)
    return json.loads(json.dumps(report, sort_keys=True, allow_nan=False))


def ablate(ranker, report, task="reg", X=X, ks=(3,)):
    y = Y if task == "reg" else LABELS
    return ranker.ablate(X[:120], y[:120], X[120:], y[120:], report, ks=list(ks))["ablation"]["curve"][0]["scores"]


def pipe(order):
    steps = {"s": StandardScaler(), "p": PCA(1)}
    return Pipeline([(n, steps[n]) for n in order] + [("r", Ridge())])


def double(values):
    return 2 * values


def triple(values):
    return 3 * values


def columns(cols):
    return make_pipeline(ColumnTransformer([("s", StandardScaler(), cols)], remainder="drop"), Ridge())


PAIRS = {
    "order": ("reg", lambda: pipe("sp"), lambda: pipe("ps")),
    "class_weight": ("clf", lambda: LogisticRegression(class_weight={0: 1, 1: 1}),
                     lambda: LogisticRegression(class_weight={0: 1, 1: 20})),
    "selector list": ("reg", lambda: columns([0, 1]), lambda: columns([1, 2])),
    "selector slice": ("reg", lambda: columns(slice(0, 1)), lambda: columns(slice(1, 2))),
    "selector mask": ("reg", lambda: columns(np.array([True, False, True])),
                      lambda: columns(np.array([False, True, True]))),
    "tuple": ("reg", lambda: make_pipeline(MinMaxScaler((0, 1)), SVR()),
              lambda: make_pipeline(MinMaxScaler((0, 100)), SVR())),
    "function": ("reg", lambda: make_pipeline(FunctionTransformer(double), Ridge()),
                 lambda: make_pipeline(FunctionTransformer(triple), Ridge())),
    "partial": ("reg", lambda: make_pipeline(FunctionTransformer(functools.partial(np.clip, a_min=0, a_max=1)), Ridge()),
                lambda: make_pipeline(FunctionTransformer(functools.partial(np.clip, a_min=0, a_max=2)), Ridge())),
    "kw_args": ("reg", lambda: make_pipeline(FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 1}), Ridge()),
                lambda: make_pipeline(FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 5}), Ridge())),
    "array": ("reg", lambda: Ridge(alpha=np.array([1.0])), lambda: Ridge(alpha=np.array([1e6]))),
    "union weights": (
        "reg",
        lambda: make_pipeline(FeatureUnion([("p", PCA(1)), ("s", StandardScaler())], transformer_weights={"p": 1}), Ridge()),
        lambda: make_pipeline(FeatureUnion([("p", PCA(1)), ("s", StandardScaler())], transformer_weights={"p": 9}), Ridge())),
}


@pytest.mark.parametrize("name", PAIRS)
def test_structure_changes_are_refused(name):
    task, first, second = PAIRS[name]
    report = rank(fit(first(), task), task)
    with pytest.raises(ValueError, match="'m'"):
        ablate(fit(second(), task), report, task)
    for same in (fit(first(), task), pickle.loads(pickle.dumps(fit(first(), task)))):
        assert ablate(same, report, task)["m"] == pytest.approx(report["models"]["m"]["evaluation_score"], abs=1e-9)


def test_hashes_are_recorded_and_stable():
    entry = rank(fit(pipe("sp")))["models"]["m"]
    assert "steps" in entry["param_hashes"] and not set(entry["param_hashes"]) & set(entry["params"])
    assert all(len(v) == 64 for v in entry["param_hashes"].values())
    assert entry["param_hashes"] == rank(fit(pipe("sp")))["models"]["m"]["param_hashes"]


class Odd:
    def __repr__(self):
        raise RuntimeError("no repr")

    def __eq__(self, other):
        raise RuntimeError("no eq")

    __hash__ = None


class WithOdd(Ridge):
    def __init__(self, odd=None, loop=None, rng=None, alpha=1.0):
        super().__init__(alpha=alpha)
        self.odd, self.loop, self.rng = odd, loop, rng


def test_strange_parameters_do_not_break_reports():
    loop = [1]
    loop.append({"x": Odd()})
    deep = [1]
    for _ in range(200):
        deep = [deep]
    for value in (Odd(), loop, deep, {1: Odd(), "1": 2}, {(1, 2): 3, "a": {4, "5"}}, object(), lambda v: v,
                  np.array([Odd(), 1], dtype=object), float("nan"), b"x", np.datetime64("2020-01-01"),
                  range(3), memoryview(b"x")):
        model = WithOdd(odd=value)
        ranker = FeatureRanker("reg", models={"m": model}, verbose=False)
        try:
            ranker.fit(X[:120], Y[:120], feature_names=NAMES)
        except RuntimeError:
            continue  # scikit-learn could not clone it
        report = ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=1)
        assert report["models"]["m"]["param_hashes"]["odd"]
        ablate(ranker, json.loads(json.dumps(report)))


def test_self_referencing_parameter():
    loop = []
    loop.append(loop)
    model = WithOdd()
    ranker = FeatureRanker("reg", models={"m": model}, verbose=False).fit(X[:120], Y[:120], feature_names=NAMES)
    ranker.trained_models["m"].loop = loop
    assert rank(ranker)["models"]["m"]["param_hashes"]["loop"]


def test_generators_and_objects_do_not_depend_on_addresses():
    def hashes():
        return rank(fit(WithOdd(odd=object(), rng=np.random.RandomState(0))))["models"]["m"]["param_hashes"]

    assert hashes() == hashes()


def test_older_report_without_hashes_is_accepted():
    report = rank(fit(pipe("sp")))
    del report["models"]["m"]["param_hashes"]
    assert ablate(fit(pipe("sp")), report)


def test_hashes_from_another_sklearn_version_warn():
    report = rank(fit(pipe("sp")))
    report["provenance"]["scikit-learn"] = "0.0.1"
    with pytest.warns(UserWarning, match="'m'"):
        ablate(fit(pipe("ps")), report)


def test_report_matches_the_schema():
    jsonschema = pytest.importorskip("jsonschema")
    import featranker

    schema = json.loads((Path(featranker.__file__).parent / "report.schema.json").read_text(encoding="utf-8"))
    report = rank(fit(pipe("sp")))
    jsonschema.validate(report, schema)
    report["models"]["m"]["param_hashes"] = {"steps": 3}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(report, schema)


# categorical columns and model input


def coded():
    values = np.random.default_rng(1).integers(0, 20, (200, 3)).astype(float)
    values[:, 1] = np.clip(np.round(5 * Y) + 10, 0, 19)
    return values


def test_other_categorical_columns_are_refused():
    Z = coded()

    def make(cols):
        return fit(Ridge(), X=Z, categorical=cols)

    report = rank(make(["b"]), X=Z)
    for cols in (["a"], ["a", "b"], None):
        with pytest.raises(ValueError, match="categorical"):
            ablate(make(cols), report, X=Z)
    plain = rank(make(None), X=Z)
    with pytest.raises(ValueError, match="categorical"):
        ablate(make(["a"]), plain, X=Z)
    assert ablate(make(["b"]), report, X=Z)
    both = rank(make(["a", "b"]), X=Z)
    assert ablate(make(["b", "a"]), both, X=Z)


def test_other_model_input_is_refused():
    pytest.importorskip("pandas")
    report = rank(fit(Ridge()))
    with pytest.raises(ValueError, match="model_input"):
        ablate(fit(Ridge(), model_input="dataframe"), report)
    with pytest.raises(ValueError, match="model_input"):
        ablate(fit(Ridge()), rank(fit(Ridge(), model_input="dataframe")))
    assert ablate(fit(Ridge(), model_input="dataframe"), rank(fit(Ridge(), model_input="dataframe")))


# templates edited after a fit by an older version


@pytest.mark.parametrize("model", [lambda: Ridge(alpha=1.0), lambda: make_pipeline(StandardScaler(), Ridge())])
def test_old_pickle_refits_the_fitted_model(model):
    ranker = fit(model())
    report = rank(ranker)
    ranker.fit_models_ = {}  # as in a ranker pickled before 1.22
    last = ranker.user_models["m"]
    (last[-1] if hasattr(last, "steps") else last).alpha = 1e5
    assert ablate(ranker, report)["m"] == pytest.approx(report["models"]["m"]["evaluation_score"], abs=1e-9)


# options changed after fit


@pytest.mark.parametrize("option,value", [
    ("max_categories", 7), ("missing", "impute"), ("model_input", "dataframe"), ("shadows", True),
    ("shadow_seed", 5), ("shadow_mode", "independent"),
])
def test_rank_features_refuses_changed_options(option, value):
    ranker = fit(Ridge(), categorical=["c"], X=coded())
    setattr(ranker, option, value)
    with pytest.raises(ValueError, match=option):
        rank(ranker, X=coded())
    ranker.fit(coded()[:120], Y[:120], feature_names=NAMES)
    assert rank(ranker, X=coded())


def test_shuffle_may_change_without_shadows():
    ranker = fit(Ridge())
    ranker.shuffle = "circular"
    assert rank(ranker)
    ranker = fit(Ridge(), shadows=True, shadow_seed=0)
    ranker.shuffle = "circular"
    with pytest.raises(ValueError, match="shuffle"):
        rank(ranker)


def test_unchanged_and_unrelated_options_still_work():
    ranker = fit(Ridge(), shadows=True, shadow_seed=np.int64(3))
    ranker.verbose, ranker.n_jobs, ranker.shadow_seed = False, 1, 3
    assert rank(pickle.loads(pickle.dumps(ranker)))


# SHAP


def evidence(model, task="reg", scoring=None):
    pytest.importorskip("shap")
    ranker = fit(model, task, evidence=True)
    y = Y if task == "reg" else LABELS
    scoring = scoring or (lambda e, A, b: -float(np.mean(np.abs(np.asarray(e.predict(A)).reshape(len(b), -1)[:, 0] - b))))
    report = ranker.rank_features(X[120:], y[120:], scoring=scoring, n_repeats=1)
    return report["evidence"]["shap"]["unsupported"]


@pytest.mark.parametrize("alphas", [[0.1, 0.9], [0.1, 0.5, 0.9]])
def test_multi_quantile_regression_is_unsupported(alphas):
    try:
        from catboost import CatBoostRegressor
    except Exception:
        pytest.skip("catboost is not installed")
    text = ",".join(map(str, alphas))
    unsupported = evidence(CatBoostRegressor(loss_function=f"MultiQuantile:alpha={text}", iterations=5, verbose=0,
                                             allow_writing_files=False))
    assert [u["model"] for u in unsupported] == ["m"] and "outputs" in unsupported[0]["reason"]


def test_xgboost_multi_quantile_is_unsupported_and_classifiers_are_not():
    try:
        import xgboost
    except Exception:
        pytest.skip("xgboost is not installed")
    unsupported = evidence(xgboost.XGBRegressor(n_estimators=5, objective="reg:quantileerror",
                                                quantile_alpha=np.array([0.1, 0.9])))
    assert [u["model"] for u in unsupported] == ["m"]
    assert evidence(xgboost.XGBClassifier(n_estimators=5), "clf", "accuracy") == []


def test_classifiers_with_two_outputs_are_still_supported():
    from sklearn.ensemble import RandomForestClassifier

    assert evidence(RandomForestClassifier(n_estimators=5), "clf", "accuracy") == []


# benchmark and ks


@pytest.fixture
def bench(tmp_path, monkeypatch):
    run = load_run()
    values = np.random.default_rng(0).normal(size=(569, 30))
    y = np.r_[np.zeros(568, dtype=int), 1]
    state = {"hook": lambda: None}

    def load(*args, **kwargs):
        state["hook"]()
        return values, y, [f"f{i}" for i in range(30)]

    monkeypatch.setattr(run, "load", load)
    out = tmp_path / "run.json"

    def main(*arguments):
        monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", "1510", "--out", str(out), *arguments])
        run.main()
        return json.loads(out.read_text())

    return run, out, main, state


def test_second_writer_is_refused(bench, capsys):
    _, out, main, state = bench
    main("--seeds", "0")
    refused = []

    def second():
        state["hook"] = lambda: None
        with pytest.raises(SystemExit) as caught:
            main("--seeds", "2", "--resume")
        refused.append(caught.value.code)
        sys.argv = ["run.py", "--datasets", "1510", "--out", str(out), "--seeds", "1", "--resume"]

    state["hook"] = second
    result = main("--seeds", "1", "--resume")
    assert refused == [2] and ".lock" in capsys.readouterr().err
    assert {r["seed"] for r in result["rows"]} == {0, 1}
    assert [p.name for p in out.parent.iterdir()] == ["run.json"]
    assert {r["seed"] for r in main("--seeds", "2", "--resume")["rows"]} == {0, 1, 2}


def test_lock_is_removed_after_errors(bench):
    _, out, main, state = bench
    main("--seeds", "0", "--commit", "x")
    with pytest.raises(SystemExit):
        main("--seeds", "1", "--resume", "--commit", "y")

    def interrupt():
        raise KeyboardInterrupt

    state["hook"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        main("--seeds", "1", "--resume")
    assert [p.name for p in out.parent.iterdir()] == ["run.json"]


def test_render_needs_no_lock(bench, monkeypatch, capsys):
    run, out, main, _ = bench
    main("--seeds", "0")
    lock = out.with_name(out.name + ".lock")
    lock.touch()
    monkeypatch.setattr(sys, "argv", ["run.py", "--render", str(out)])
    run.main()
    assert lock.exists() and "| dataset |" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main("--seeds", "1", "--resume")
    assert lock.exists()  # another run's lock is never removed


def test_wheel_install_keeps_its_version(bench, tmp_path, monkeypatch):
    import featranker

    run, *_ = bench
    site = tmp_path / "site"
    shutil.copytree(Path(featranker.__file__).parent, site / "featranker",
                    ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(featranker, "__file__", str(site / "featranker" / "__init__.py"))
    monkeypatch.setattr(run.importlib.metadata, "version", lambda name: "2.0.0" if name == "featranker" else "0")
    assert run._featranker_info(None)["version"] == "2.0.0"


def test_scalar_ks_raises_the_documented_error():
    ranker = fit(Ridge())
    report = rank(ranker)
    for ks in (np.array(2), np.array([[1, 2]]), np.array([])):
        with pytest.raises(ValueError, match="ks must be"):
            ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=ks)
    assert len(ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=np.array([1, 2]))["ablation"]["curve"]) == 2


# review round 1


def hashes(model):
    from featranker.importance import _param_hashes

    return _param_hashes(model)


@pytest.mark.parametrize("first,second", [
    (np.nan_to_num, np.sort), (np.round, np.zeros_like), (np.log1p, np.exp), (np.clip, np.mean),
    (np.linalg.norm, np.linalg.det), (str.lower, str.upper), (abs, round), (dict, list),
    (np.add.reduce, np.multiply.reduce), ("x".upper, "x".lower),
])
def test_callables_are_told_apart(first, second):
    one, other = hashes(WithOdd(odd=first))["odd"], hashes(WithOdd(odd=second))["odd"]
    assert one != other and one == hashes(WithOdd(odd=first))["odd"]


def test_changed_numpy_function_is_refused():
    def make(function):
        return fit(make_pipeline(FunctionTransformer(function), Ridge()))

    report = rank(make(np.nan_to_num))
    with pytest.raises(ValueError, match="'m'"):
        ablate(make(np.sort), report)
    assert ablate(make(np.nan_to_num), report)


@pytest.mark.parametrize("first,second", [
    ({0: 1.0, 1: 2.0}, {np.int64(0): 1.0, np.int64(1): np.float64(2)}),
    ([0.1, 1.0], [np.float64(0.1), np.float64(1.0)]), (["a"], [np.str_("a")]), ([float("nan")], [np.float64("nan")]),
    ([True], [np.bool_(True)]), (float("inf"), np.float64("inf")),
    (np.arange(3, dtype="<i8"), np.arange(3, dtype=">i8")),
    (np.arange(6.0).reshape(2, 3), np.asfortranarray(np.arange(6.0).reshape(2, 3))),
])
def test_equal_values_hash_equally(first, second):
    assert hashes(WithOdd(odd=first))["odd"] == hashes(WithOdd(odd=second))["odd"]


def test_memmaps_hash_as_arrays(tmp_path):
    array = np.arange(1000.0)
    np.save(tmp_path / "a.npy", array)
    mapped = np.load(tmp_path / "a.npy", mmap_mode="r")
    assert type(mapped) is np.memmap
    assert hashes(WithOdd(odd=mapped)) == hashes(WithOdd(odd=array))


def test_array_contents_are_told_apart():
    import scipy.sparse as sp

    masked = [np.ma.array([1.0, 2.0], mask=m) for m in ([0, 1], [1, 0])]
    sparse = [sp.csr_matrix(m) for m in ([[1.0, 0]], [[0, 1.0]])]
    for one, other in (masked, sparse):
        assert hashes(WithOdd(odd=one)) != hashes(WithOdd(odd=other))
    assert hashes(WithOdd(odd=sp.csc_matrix([[1.0, 0]]))) == hashes(WithOdd(odd=sp.csc_matrix([[1.0, 0]])))


def test_parallel_ranking_does_not_change_hashes():
    model = WithOdd(odd=np.arange(400_000.0))
    ranker = FeatureRanker("reg", models={"m": model, "r": Ridge()}, n_jobs=2, verbose=False)
    ranker.fit(X[:120], Y[:120], feature_names=NAMES)
    report = rank(ranker)
    assert ablate(ranker, report)
    single = FeatureRanker("reg", models={"m": model, "r": Ridge()}, verbose=False)
    assert rank(single.fit(X[:120], Y[:120], feature_names=NAMES))["models"]["m"]["param_hashes"] == (
        report["models"]["m"]["param_hashes"])


# review round 2


def structured(flag=1, time=2.0, label="x", align=True):
    dtype = np.dtype([("flag", "?"), ("time", "f8"), ("label", object), ("when", "M8[s]"), ("sub", "f4", (2,))], align=align)
    array = np.empty(50, dtype=dtype)
    array["flag"], array["time"], array["label"] = flag, time, label
    array["when"], array["sub"] = np.datetime64("2020-01-01"), 1.5
    return array


def test_structured_arrays_hash_by_value():
    reference = hashes(WithOdd(odd=structured()))["odd"]
    for _ in range(20):
        assert hashes(WithOdd(odd=structured()))["odd"] == reference
        assert hashes(WithOdd(odd=pickle.loads(pickle.dumps(structured()))))["odd"] == reference
    for changed in ({"flag": 0}, {"time": 3.0}, {"label": "y"}):
        assert hashes(WithOdd(odd=structured(**changed)))["odd"] != reference
    later = structured()
    later["when"] = np.datetime64("2021-01-01")
    assert hashes(WithOdd(odd=later))["odd"] != reference
    padded = np.zeros(1000, dtype=np.dtype([("flag", "?"), ("time", "f8")], align=True))
    assert len({hashes(WithOdd(odd=padded))["odd"] for _ in range(20)}) == 1


def test_shapes_are_told_apart():
    arrays = [np.array(3.0), np.array([3.0]), np.array([[3.0]]), np.array(3.0, dtype=object),
              np.array([3.0], dtype=object), np.array([3.0], dtype="f4")]
    assert len({hashes(WithOdd(odd=a))["odd"] for a in arrays}) == len(arrays)
    for array in arrays:
        assert hashes(WithOdd(odd=array.copy()))["odd"] == hashes(WithOdd(odd=array))["odd"]
