"""Acceptance check for report-schema / 2.0 (lead-written; builder must not edit)."""
import copy
import json
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from featranker import FeatureRanker

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((files("featranker") / "report.schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
RNG = np.random.default_rng(7)
X = RNG.normal(size=(240, 4))
Y = 2 * X[:, 0] + X[:, 1] + 0.2 * RNG.normal(size=240)
NAMES = ["a", "b", "c", "d"]
REG = {"r": Ridge(), "l": Lasso(alpha=0.01)}


def roundtrip(report):
    return json.loads(json.dumps(report, allow_nan=False))


def valid(report):
    VALIDATOR.validate(roundtrip(report))


def held_out(models=None, fit_kwargs=None, rank_kwargs=None, X_=X, y_=Y, task="reg", **kwargs):
    ranker = FeatureRanker(task, models=models or REG, verbose=False, **kwargs)
    fit_args = {} if isinstance(X_, pd.DataFrame) else {"feature_names": NAMES}
    ranker.fit(X_[:160], y_[:160], **fit_args, **(fit_kwargs or {}))
    rank_kwargs = {"scoring": "r2", **(rank_kwargs or {})}
    return ranker, ranker.rank_features(X_[160:], y_[160:], n_repeats=2, random_state=0, **rank_kwargs)


# Shipped schema


def test_schema_is_draft_2020_12_and_versioned():
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    assert "featranker" in SCHEMA["$id"].lower() and "2" in SCHEMA["$id"]


def test_schema_in_package_data():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    line = re.search(r"^\[tool\.setuptools\.package-data\]\nfeatranker = \[(.*)\]$", text, flags=re.M)
    assert line and '"report.schema.json"' in line.group(1)


# Real reports validate


def test_plain_held_out():
    valid(held_out()[1])


def test_shadows_and_evidence():
    valid(held_out(shadows=True, shadow_seed=0, evidence=True)[1])


def test_missing_impute():
    Xm = X.copy()
    Xm[::9, 2] = np.nan
    ranker = FeatureRanker("reg", "linear", verbose=False, missing="impute")
    ranker.fit(Xm[:160], Y[:160], feature_names=NAMES)
    valid(ranker.rank_features(Xm[160:], Y[160:], scoring="r2", n_repeats=2, random_state=0))


def test_categorical_frame():
    frame = pd.DataFrame(X, columns=NAMES)
    frame["c"] = (X[:, 2] > 0).astype(int).astype(str)
    valid(held_out(X_=frame, categorical=["c"])[1])


def test_several_scorers():
    valid(held_out(rank_kwargs={"scoring": ["r2", "neg_mean_absolute_error"]})[1])


def test_sample_weights():
    w = np.ones(240)
    valid(held_out(fit_kwargs={"sample_weight": w[:160]}, rank_kwargs={"sample_weight": w[160:]})[1])


def test_classification():
    yc = (Y > 0).astype(int)
    models = {"lr": LogisticRegression(), "dt": DecisionTreeClassifier(max_depth=3, random_state=0)}
    valid(held_out(models=models, y_=yc, task="clf", rank_kwargs={"scoring": "accuracy"})[1])


def test_with_failure_entry():
    class Broken(Ridge):
        def fit(self, X, y, sample_weight=None):
            raise ValueError("boom")

    report = held_out(models={**REG, "bad": Broken()})[1]
    assert report["failures"]
    valid(report)


def test_ablation():
    ranker, report = held_out()
    out = ranker.ablate(X[:160], Y[:160], X[160:], Y[160:], report, ks=[1, 2])
    valid(out)


def test_resampled():
    ranker = FeatureRanker("reg", models=REG, verbose=False)
    valid(ranker.rank_resampled(X, Y, scoring="r2", cv=3, n_repeats=2, random_state=0,
                                feature_names=NAMES))


def test_resampled_with_shadows():
    ranker = FeatureRanker("reg", models=REG, verbose=False, shadows=True, shadow_seed=0)
    valid(ranker.rank_resampled(X, Y, scoring="r2", cv=3, n_repeats=2, random_state=0,
                                feature_names=NAMES))


# Broken reports fail


@pytest.fixture(scope="module")
def base():
    _, report = held_out()
    return roundtrip(report)


@pytest.fixture(scope="module")
def base_resampled():
    ranker = FeatureRanker("reg", models=REG, verbose=False)
    return roundtrip(ranker.rank_resampled(X, Y, scoring="r2", cv=3, n_repeats=2,
                                           random_state=0, feature_names=NAMES))


def broken(report, edit):
    report = copy.deepcopy(report)
    edit(report)
    return not VALIDATOR.is_valid(report)


@pytest.mark.parametrize("key", ["evaluation_mode", "scoring", "provenance", "run", "models",
                                 "consensus", "failures"])
def test_held_out_required(base, key):
    assert broken(base, lambda r: r.pop(key))


@pytest.mark.parametrize("key", ["evaluation_mode", "scoring", "provenance", "run", "splits",
                                 "stability", "failures"])
def test_resampled_required(base_resampled, key):
    assert broken(base_resampled, lambda r: r.pop(key))


def test_bad_types_fail(base):
    assert broken(base, lambda r: r.__setitem__("evaluation_mode", "other"))
    assert broken(base, lambda r: r["provenance"].__setitem__("featranker", 2))
    assert broken(base, lambda r: r["consensus"][0].pop("feature_group"))
    assert broken(base, lambda r: r["consensus"][0].__setitem__("median_rank", "1"))
    assert broken(base, lambda r: r["models"]["r"].__setitem__("evaluation_score", "0.5"))
    assert broken(base, lambda r: r["models"]["r"].pop("family"))
    assert broken(base, lambda r: r["models"]["r"].pop("importance"))


def test_null_score_allowed(base):
    report = copy.deepcopy(base)
    report["models"]["r"]["evaluation_score"] = None
    assert VALIDATOR.is_valid(report)


def test_ablation_requires_curve(base):
    ranker, report = held_out()
    out = roundtrip(ranker.ablate(X[:160], Y[:160], X[160:], Y[160:], report, ks=[1]))
    assert broken(out, lambda r: r["ablation"].pop("curve"))


def test_unknown_keys_allowed(base):
    report = copy.deepcopy(base)
    report["x_future"] = {"any": 1}
    report["models"]["r"]["x_future"] = True
    report["consensus"][0]["x_future"] = "y"
    report["run"]["x_future"] = 1
    assert VALIDATOR.is_valid(report)


# Docs and dependencies


def test_docs_and_dependencies():
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    section = re.split(r"^## .*Report schema.*$", api, flags=re.M | re.I)
    assert len(section) == 2, "docs/api.md needs one 'Report schema' section"
    body = re.split(r"^## ", section[1], flags=re.M)[0]
    assert "report.schema.json" in body and "jsonschema" in body and "2.x" in body
    assert "report.schema.json" in (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "jsonschema" not in runtime
    test_extra = re.search(r"^test = \[(.*)\]$", text, flags=re.M).group(1)
    assert "jsonschema" in test_extra
