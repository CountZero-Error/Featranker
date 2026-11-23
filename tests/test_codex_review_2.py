"""Acceptance check for codex-review-2 (lead-written; builder must not edit)."""
import copy
import hashlib
import importlib.util
import json
import re
import sys
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(31)
X = RNG.normal(size=(150, 3))
Y = 2 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * RNG.normal(size=150)
NAMES = ["a", "b", "c"]
TWO = {"r": Ridge(), "l": Lasso(alpha=0.01)}


def fitted(models=None, **kwargs):
    return FeatureRanker("reg", models=models or TWO, verbose=False, **kwargs).fit(
        X[:100], Y[:100], feature_names=NAMES)


def ranked(ranker, scoring="r2"):
    return ranker.rank_features(X[100:], Y[100:], scoring=scoring, n_repeats=2, random_state=0)


# Old reports into ablate warn; current ones do not


def old_style(report):
    old = json.loads(json.dumps(report, allow_nan=False))
    del old["run"]
    for model in old["models"].values():
        model.pop("family", None)
    old["provenance"]["featranker"] = "1.11.0"
    return old


def schema_warnings(caught):
    return [w for w in caught if "report.schema.json" in str(w.message)]


def test_old_report_into_ablate_warns_and_still_runs():
    ranker = fitted()
    old = old_style(ranked(ranker))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ranker.ablate(X[:100], Y[:100], X[100:], Y[100:], old, ks=[1])
    hits = schema_warnings(caught)
    assert hits and all(issubclass(w.category, UserWarning) for w in hits)
    assert out["ablation"]["curve"]


def test_current_report_into_ablate_does_not_warn():
    ranker = fitted()
    report = ranked(ranker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ranker.ablate(X[:100], Y[:100], X[100:], Y[100:], report, ks=[1])
    assert not schema_warnings(caught)


def test_api_doc_scopes_promise():
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    body = re.split(r"^## .*Report schema.*$", api, flags=re.M | re.I)[1]
    body = re.split(r"^## ", body, flags=re.M)[0]
    assert "ablate" in body and re.search(r"older|before 2\.0|pre-2\.0|1\.x", body)


# Scorer name is always a string

jsonschema = pytest.importorskip("jsonschema")
VALIDATOR = jsonschema.Draft202012Validator(
    json.loads((files("featranker") / "report.schema.json").read_text(encoding="utf-8")))


class NamelessScorer:
    __name__ = None

    def __call__(self, est, X_, y_):
        return r2_score(y_, est.predict(X_))


def test_nameless_callable_scorer_gives_string_scoring():
    held = ranked(fitted(), scoring=NamelessScorer())
    resampled = FeatureRanker("reg", models=TWO, verbose=False).rank_resampled(
        X, Y, scoring=NamelessScorer(), cv=3, n_repeats=2, random_state=0, feature_names=NAMES)
    for report in (held, resampled):
        assert isinstance(report["scoring"], str) and report["scoring"]
        VALIDATOR.validate(json.loads(json.dumps(report, allow_nan=False)))


# Overflow-safe ablation medians


def huge(est, X_, y_):
    model = est.steps[-1][1] if hasattr(est, "steps") else est
    return 0.0 if isinstance(model, DummyRegressor) else 1e308


def test_ablation_median_does_not_overflow():
    ranker = fitted()
    report = ranked(ranker, scoring=huge)
    out = ranker.ablate(X[:100], Y[:100], X[100:], Y[100:], report, ks=[1], scoring=huge)
    row = out["ablation"]["curve"][0]
    assert row["median_score"] == 1e308
    json.dumps(out, allow_nan=False)


# SHAP: a reused private encoder with duplicate sources is not folded

shap = pytest.importorskip("shap")
from featranker.evidence import _shap_values  # noqa: E402


def categorical_frame(seed=2, n=300):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 5, n)
    frame = pd.DataFrame({"c": codes.astype(str), "x": rng.normal(size=n), "z": rng.normal(size=n)})
    return frame, (codes % 3) + frame["x"].to_numpy() + 0.1 * rng.normal(size=n)


def reused_encoder_pipeline():
    frame, y = categorical_frame()
    builtin = FeatureRanker("reg", "linear", verbose=False, categorical=["c"]).fit(frame.iloc[:200], y[:200])
    encoder = clone(builtin.trained_models["ridge_regression"][0].transformers[0][1])
    model = make_pipeline(
        ColumnTransformer([("cat", encoder, [0, 0])], remainder=StandardScaler()), Ridge())
    codes = np.column_stack([np.random.default_rng(3).integers(0, 5, 300),
                             np.random.default_rng(4).normal(size=(300, 2))]).astype(float)
    target = codes[:, 1] + (codes[:, 0] == 2) + 0.1 * np.random.default_rng(5).normal(size=300)
    return model, codes, target


def test_reused_encoder_duplicate_sources_not_folded():
    model, codes, target = reused_encoder_pipeline()
    model.fit(codes, target)
    with pytest.raises(ValueError, match="ColumnTransformer"):
        _shap_values(model, codes[:30], codes[30:90], shap)


def test_reused_encoder_listed_unsupported():
    model, codes, target = reused_encoder_pipeline()
    ranker = FeatureRanker("reg", models={"m": model, "r": Ridge()}, verbose=False, evidence=True)
    ranker.fit(codes[:200], target[:200], feature_names=["c", "x", "z"])
    report = ranker.rank_features(codes[200:], target[200:], scoring="r2", n_repeats=2, random_state=0)
    reasons = {e["model"]: e["reason"] for e in report["evidence"]["shap"]["unsupported"]}
    assert "ColumnTransformer" in reasons.get("m", "")


# Real-data benchmark records and checks its own script hash


def load_run():
    spec = importlib.util.spec_from_file_location(
        "realdata_run_review2", ROOT / "benchmarks" / "realdata" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_SHA = hashlib.sha256((ROOT / "benchmarks" / "realdata" / "run.py").read_bytes()).hexdigest()


def stored(run, script_sha):
    first = run.DATASETS[0]
    prev = {
        "rows": [{**run._row("all", m, 0, 0.5, 1, 0.0, None), "dataset": first["name"]} for m in run.MODELS],
        "shapes": {}, "versions": run._versions(), "featranker": run._featranker_info("482b62d"),
        "prereg_sha256": hashlib.sha256(run.PREREG.read_bytes()).hexdigest(), "started": "x",
        "wall_seconds": 0.0,
    }
    if script_sha is not None:
        prev["script_sha256"] = script_sha
    return first, prev


def resume(run, tmp_path, monkeypatch, script_sha):
    first, prev = stored(run, script_sha)
    out = tmp_path / "out.json"
    out.write_text(json.dumps(prev))
    monkeypatch.setattr(sys, "argv", ["run.py", "--datasets", str(first["data_id"]), "--seeds", "0",
                                      "--out", str(out), "--resume"])
    run.main()
    return json.loads(out.read_text())


@pytest.mark.parametrize("script_sha", ["0" * 64, None])
def test_resume_refuses_other_script(tmp_path, monkeypatch, capsys, script_sha):
    run = load_run()
    with pytest.raises(SystemExit) as info:
        resume(run, tmp_path, monkeypatch, script_sha)
    assert info.value.code == 2
    assert "script" in capsys.readouterr().err


def test_resume_same_script_records_hash(tmp_path, monkeypatch):
    run = load_run()
    saved = resume(run, tmp_path, monkeypatch, SCRIPT_SHA)
    assert saved["script_sha256"] == SCRIPT_SHA


def test_deviation_names_script_revision():
    text = (ROOT / "benchmarks" / "realdata" / "DEVIATIONS.md").read_text(encoding="utf-8")
    assert "be82957" in text


# Docs


def test_no_unreleased_versions_in_user_docs():
    pattern = re.compile(r"\b(since|from|in|added in)\s+(version\s+)?1\.2[234]\b", re.I)
    for name in ("README.md", "docs/README.md", "docs/usage.md", "docs/api.md", "docs/report.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not pattern.search(text), (name, pattern.search(text))


def test_usage_documents_sparse_limits():
    usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8").lower()
    assert "same problem" in usage or "same optimisation problem" in usage or "same optimization problem" in usage
    assert "density" in usage


def test_recipe_decides_on_validation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    recipe = re.split(r"^#{1,4} .*10-minute recipe.*$", readme, flags=re.M | re.I)[1]
    recipe = re.split(r"^#{1,4} ", recipe, flags=re.M)[0].lower()
    assert "validation" in recipe and "once" in recipe
