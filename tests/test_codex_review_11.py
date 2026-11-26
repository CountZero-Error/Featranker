"""Acceptance check for codex-review-11 (lead-written; builder must not edit)."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from featranker import FeatureRanker
from test_codex_review_10 import NAMES, X, Y, WithOdd, ablate, bench, evidence, fit, hashes, rank  # noqa: F401

ROOT = Path(__file__).parents[1]


def odd(value):
    return hashes(WithOdd(odd=value))["odd"]


def test_spawned_process_accepts_an_unchanged_ranker(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent("""
        import json, multiprocessing, sys
        import numpy as np
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        from featranker import FeatureRanker

        X = np.random.default_rng(0).normal(size=(100, 2))
        Y = X[:, 0] + 0.1 * X[:, 1]

        def squared(values):
            return np.c_[values, values ** 2]

        class Identity(TransformerMixin, BaseEstimator):
            def __init__(self, tool=None):
                self.tool = tool
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X

        def work(ranker, report):
            ranker.ablate(X[:60], Y[:60], X[60:], Y[60:], report, ks=[2])

        if __name__ == "__main__":
            model = make_pipeline(FunctionTransformer(squared), Identity([Identity(squared), Identity]), Ridge())
            ranker = FeatureRanker("reg", models={"m": model}, verbose=False)
            ranker.fit(X[:60], Y[:60], feature_names=["a", "b"])
            report = dict(ranker.rank_features(X[60:], Y[60:], scoring="r2", n_repeats=1))
            process = multiprocessing.get_context("spawn").Process(target=work, args=(ranker, report))
            process.start(); process.join(); sys.exit(process.exitcode)
    """))
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0, done.stderr[-800:]


def test_grid_candidates_are_compared():
    def make(alpha):
        return fit(GridSearchCV(make_pipeline(Ridge()), {"ridge": [Ridge(alpha=alpha)]}, cv=2))

    report = rank(make(0.01))
    with pytest.raises(ValueError, match="'m'"):
        ablate(make(100.0), report)
    assert ablate(make(0.01), report)
    assert odd([Ridge(alpha=1)]) != odd([Ridge(alpha=2)]) and odd({"a": Ridge()}) == odd({"a": Ridge()})


def test_pandas_labels_are_compared():
    pd = pytest.importorskip("pandas")
    series = [pd.Series([10.0, 0.01], index=i) for i in (["a", "b"], ["b", "a"])]
    frames = [pd.DataFrame({"x": [1, 2]}, index=i) for i in ([0, 1], [1, 0])]
    columns = [pd.DataFrame([[1, 2]], columns=c) for c in (["a", "b"], ["b", "a"])]
    dates = [pd.Series([1], index=pd.date_range(d, periods=1)) for d in ("2020-01-01", "2021-01-01")]
    for one, other in (series, frames, columns, dates):
        assert odd(one) != odd(other) and odd(one) == odd(one.copy())
    assert odd(pd.Index(["a"])) != odd(pd.Index(["b"]))
    assert odd(pd.Categorical(["a", "b"])) == odd(pd.Categorical(["a", "b"]))


def record(names, values=(2, 4)):
    return np.array([values], dtype=[(n, "i8") for n in names])[0]


def test_record_scalars():
    assert odd(record(["a", "b"])) != odd(record(["b", "a"]))
    assert odd(record(["a", "b"])) != odd(record(["a", "b"], (4, 2)))
    vector = np.array([((1.123456, 2.0),)], dtype=[("v", "f8", (2,))])[0]
    reference = odd(vector)
    with np.printoptions(precision=2):
        assert odd(vector) == reference
    holder = np.array([(object,)], dtype=[("o", object)])[0]
    assert odd(holder) == odd(holder.copy())


def test_masked_fill_values():
    one, other = (np.ma.array([1.0, 2.0], mask=[0, 1], fill_value=f) for f in (2.0, 4.0))
    assert odd(one) != odd(other) and odd(one) == odd(one.copy())


@pytest.mark.parametrize("shape", [(3,), (2, 2, 2)])
def test_n_dimensional_sparse(shape):
    def make(value):
        dense = np.zeros(shape)
        dense.flat[1] = value
        try:
            array = sp.coo_array(dense)
        except (TypeError, ValueError):
            pytest.skip("this SciPy has no n-dimensional coo_array")
        if array.shape != shape:
            pytest.skip("this SciPy has no n-dimensional coo_array")
        return array

    assert odd(make(2.0)) != odd(make(4.0)) and odd(make(2.0)) == odd(make(2.0))


def test_parameter_moving_between_scalar_and_array_warns_across_versions():
    report = rank(fit(Ridge(alpha=1.0)))
    report["provenance"]["scikit-learn"] = "0.0.1"
    with pytest.warns(UserWarning, match="'m'"):
        ablate(fit(Ridge(alpha=np.array([100.0]))), report)
    report["provenance"]["scikit-learn"] = sklearn.__version__
    with pytest.raises(ValueError, match="'m'"):
        ablate(fit(Ridge(alpha=np.array([100.0]))), report)


def test_hash_text_is_not_a_scalar():
    digest = odd([1])
    report = rank(fit(WithOdd(odd=[1])))
    with pytest.raises(ValueError, match="'m'"):
        ablate(fit(WithOdd(odd=digest)), report)


def test_rmse_with_uncertainty_is_unsupported():
    try:
        from catboost import CatBoostRegressor
    except Exception:
        pytest.skip("catboost is not installed")
    options = {"iterations": 5, "verbose": 0, "allow_writing_files": False}
    unsupported = evidence(CatBoostRegressor(loss_function="RMSEWithUncertainty", **options))
    assert [u["model"] for u in unsupported] == ["m"] and "outputs" in unsupported[0]["reason"]
    assert evidence(CatBoostRegressor(**options), scoring="r2") == []


def test_linear_and_pls_shap_still_work():
    from sklearn.cross_decomposition import PLSRegression

    assert evidence(Ridge(), scoring="r2") == []
    before = evidence(PLSRegression(1), scoring="r2")
    assert not any("outputs" in u["reason"] for u in before)


@pytest.mark.parametrize("make", [np.int64, np.uint8])
def test_assigned_numpy_options_keep_reports_json(make):
    ranker = FeatureRanker("reg", models={"m": Ridge()}, shadows=True, shadow_seed=3, max_categories=5,
                           verbose=False)
    ranker.fit(X[:120], Y[:120], feature_names=NAMES)
    ranker.shadow_seed, ranker.max_categories = make(3), make(5)
    report = ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=1)
    json.dumps(report, allow_nan=False)
    assert report["shadow_seed"] == 3 and report["run"]["max_categories"] == 5
    ranker.shadow_seed, ranker.max_categories = make(3), make(5)
    json.dumps(ranker.rank_resampled(X, Y, scoring="r2", cv=2, n_repeats=1, feature_names=NAMES), allow_nan=False)


def test_changed_missing_needs_a_new_ranker_for_built_ins():
    Z = X.copy()
    Z[0, 0] = np.nan
    ranker = FeatureRanker("reg", "linear", verbose=False).fit(X[:120], Y[:120], feature_names=NAMES)
    ranker.missing = "impute"
    with pytest.raises(ValueError, match="missing.*new FeatureRanker"):
        ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=1)
    with pytest.raises(ValueError, match="missing.*new FeatureRanker"):
        ranker.fit(Z[:120], Y[:120], feature_names=NAMES)
    ranker.missing = "error"
    ranker.fit(X[:120], Y[:120], feature_names=NAMES)
    user = FeatureRanker("reg", models={"m": Ridge()}, verbose=False)
    user.missing = "impute"
    user.fit(X[:120], Y[:120], feature_names=NAMES)


def test_old_pickles_without_a_build_record_still_fit():
    ranker = FeatureRanker("reg", "fast", missing="impute", verbose=False)
    known = {"task", "group", "n_jobs", "verbose", "shadows", "shadow_seed", "shadow_mode", "evidence", "missing",
             "model_input", "shuffle", "families", "max_categories", "categorical"}
    state = {k: v for k, v in vars(ranker).items() if k in known or not k.startswith("_")}
    old = FeatureRanker.__new__(FeatureRanker)
    old.__setstate__(state)
    assert old.fit(X[:120], Y[:120], feature_names=NAMES).is_fitted_


# benchmark


def test_repointed_symlink_keeps_the_target(bench):  # noqa: F811
    _, out, main, state = bench
    a, b = out.parent / "a.json", out.parent / "b.json"
    main("--seeds", "0")
    a.write_bytes(out.read_bytes())
    b.write_bytes(out.read_bytes())
    out.unlink()
    try:
        out.symlink_to(a)
    except OSError:
        pytest.skip("symlinks are not available")
    before = b.read_bytes()

    def repoint():
        state["hook"] = lambda: None
        out.unlink()
        out.symlink_to(b)

    state["hook"] = repoint
    main("--seeds", "1", "--resume")
    assert b.read_bytes() == before
    assert {r["seed"] for r in json.loads(a.read_text())["rows"]} == {0, 1}
    assert {p.name for p in out.parent.iterdir()} == {"run.json", "a.json", "b.json"}


def test_lock_named_output_is_refused(bench, monkeypatch, capsys):  # noqa: F811
    run, out, *_ = bench
    lock = out.with_name(out.name + ".lock")
    monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", "1510", "--seeds", "0", "--out", str(lock)])
    with pytest.raises(SystemExit):
        run.main()
    assert ".lock" in capsys.readouterr().err and not lock.exists()


def test_docs():
    usage = (ROOT / "docs" / "guide" / "selection.md").read_text(encoding="utf-8")
    line = next(text for text in usage.splitlines() if "1.22" in text and "clone" in text)
    assert "user" in line


def test_steps_hash_does_not_repeat_the_step_parameters():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    def structure(model):
        return {k: v for k, v in hashes(model).items()}

    def make(alpha, with_mean):
        scaler = ColumnTransformer([("s", StandardScaler(with_mean=with_mean), [0, 1])])
        return make_pipeline(scaler, Ridge(alpha=alpha))

    assert structure(make(1.0, True)) == structure(make(2.0, False))  # params holds these


def test_unchanged_pipeline_does_not_warn_across_versions():
    import warnings

    report = rank(fit(make_pipeline(Ridge(alpha=2.0))))
    report["provenance"]["scikit-learn"] = "0.0.1"
    entry = report["models"]["m"]
    entry["params"] = {k: v for k, v in entry["params"].items() if k != "ridge__positive"}  # older: no such parameter
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ablate(fit(make_pipeline(Ridge(alpha=2.0))), report)


def test_pandas_values_without_an_index_are_compared():
    pd = pytest.importorskip("pandas")
    pairs = [(pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01")), (pd.array([1.0]), pd.array([9.0])),
             (pd.Categorical([1.0]), pd.Categorical([9.0])), (pd.Timedelta(1, "D"), pd.Timedelta(2, "D"))]
    for one, other in pairs:
        assert odd(one) != odd(other)
