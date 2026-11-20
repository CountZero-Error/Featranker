"""Acceptance check for 1.22 clear-workflow (lead-written; builder must not edit)."""
import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Lasso, Ridge

import featranker.cli as cli_module
from featranker import FeatureRanker, summarize
from featranker.ablation import ablate as module_ablate

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(22)
X = RNG.normal(size=(180, 3))
Y = 3 * X[:, 0] + 2 * X[:, 1] + 0.1 * RNG.normal(size=180)
NAMES = ["a", "b", "c"]


def fit_ranker(models, **kwargs):
    return FeatureRanker("reg", models=models, verbose=False, **kwargs).fit(
        X[:120], Y[:120], feature_names=NAMES
    )


def rank(ranker):
    return ranker.rank_features(X[120:], Y[120:], scoring="r2", n_repeats=2, random_state=0)


# Findings state basis and scorer


def test_held_out_findings_are_descriptive_with_scorer():
    ranker = fit_ranker({"r": Ridge(), "l": Lasso(alpha=0.01)})
    findings = summarize(rank(ranker))
    assert findings, "expected at least one finding (e.g. few_families)"
    for finding in findings:
        assert finding["basis"] == "descriptive", finding["code"]
        assert finding["scoring"] == "r2"


def test_shadow_q_finding_is_experimental():
    report = {
        "evaluation_mode": "resampled",
        "scoring": "r2",
        "n_splits": 3,
        "splits": [],
        "stability": [{"feature_group": "a", "median_rank": 1.0}],
        "failures": [],
        "shadow_test": {"n_trials": 3, "min_p_value": 0.125},
    }
    findings = {f["code"]: f for f in summarize(report)}
    assert findings["insufficient_splits"]["basis"] == "experimental"
    assert findings["insufficient_splits"]["scoring"] == "r2"


def test_scoring_is_null_when_report_has_none():
    report = {
        "evaluation_mode": "resampled",
        "n_splits": 3,
        "splits": [],
        "stability": [{"feature_group": "a", "median_rank": 1.0}],
        "failures": [],
        "shadow_test": {"n_trials": 3, "min_p_value": 0.125},
    }
    for finding in summarize(report):
        assert finding["scoring"] is None


# Report records model class; ablation refuses a swapped class


def test_report_records_model_class():
    report = rank(fit_ranker({"r": Ridge()}))
    assert report["models"]["r"]["class"] == "Ridge"
    json.dumps(report, allow_nan=False)


def test_ablate_rejects_swapped_class_same_version():
    ranker = fit_ranker({"m": Ridge()})
    report = rank(ranker)
    report["models"]["m"]["class"] = "Lasso"
    with pytest.raises(ValueError, match="'m'"):
        ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[2])


def test_ablate_rejects_swapped_class_other_version():
    ranker = fit_ranker({"m": Ridge()})
    report = rank(ranker)
    report["models"]["m"]["class"] = "Lasso"
    report["provenance"]["scikit-learn"] = "0.0.1"
    with pytest.raises(ValueError, match="'m'"):
        ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[2])


def test_ablate_accepts_report_without_class():
    ranker = fit_ranker({"m": Ridge()})
    report = rank(ranker)
    del report["models"]["m"]["class"]
    out = ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[2])
    assert out["ablation"]["curve"][0]["scores"]["m"] is not None


# Pickled seeds normalised


def test_unpickled_numpy_seed_becomes_int():
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False, shadows=True,
                           shadow_seed=7)
    ranker.shadow_seed = np.int64(7)  # as a pre-1.21.2 ranker could hold it
    loaded = pickle.loads(pickle.dumps(ranker))
    assert type(loaded.shadow_seed) is int and loaded.shadow_seed == 7
    loaded.fit(X[:120], Y[:120], feature_names=NAMES)
    json.dumps(rank(loaded), allow_nan=False)


# CLI rejects --output -


class ExplodingRanker:
    def __init__(self, *args, **kwargs):
        raise AssertionError("ranking must not start")


def test_cli_rejects_dash_output(monkeypatch, tmp_path, capsys):
    (tmp_path / "train.csv").write_text("a,target\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8")
    (tmp_path / "eval.csv").write_text("a,target\n3,0\n4,1\n5,0\n6,1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "FeatureRanker", ExplodingRanker)
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--task", "reg", "--scoring", "r2",
        "--train-data", "train.csv", "--eval-data", "eval.csv",
        "--target", "target", "--output", "-",
    ])
    with pytest.raises(SystemExit) as info:
        cli_module.main()
    assert info.value.code == 2
    assert "--output" in capsys.readouterr().err
    assert not (tmp_path / "-.json").exists() and not (tmp_path / "-").exists()


# Ablation warning points at the caller


def _version_mismatch(ranker):
    report = rank(ranker)
    report["provenance"]["scikit-learn"] = "0.0.1"
    report["models"]["m"]["params"] = {**report["models"]["m"]["params"], "alpha": 5}
    return report


def test_warning_points_at_caller_via_method():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = _version_mismatch(ranker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ranker.ablate(X[:120], Y[:120], X[120:], Y[120:], report, ks=[2])
    hits = [w for w in caught if "0.0.1" in str(w.message)]
    assert hits and Path(hits[0].filename).resolve() == Path(__file__).resolve()


def test_warning_points_at_caller_via_module_function():
    ranker = fit_ranker({"m": Ridge(alpha=1)})
    report = _version_mismatch(ranker)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module_ablate(ranker, X[:120], Y[:120], X[120:], Y[120:], report, ks=[2])
    hits = [w for w in caught if "0.0.1" in str(w.message)]
    assert hits and Path(hits[0].filename).resolve() == Path(__file__).resolve()


# Docs


def _headings(text):
    return [h.lower() for h in re.findall(r"^#{1,4} (.+)$", text, flags=re.M)]


def test_readme_has_recipe_and_when_to_use():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    heads = _headings(readme)
    assert any("10-minute recipe" in h for h in heads)
    assert any("when to use" in h for h in heads)
    recipe = re.split(r"^#{1,4} .*10-minute recipe.*$", readme, flags=re.M | re.I)[1]
    recipe = re.split(r"^#{1,4} ", recipe, flags=re.M)[0]
    assert "```python" in recipe and "```text" in recipe
    for word in ("ablate", "test"):
        assert word in recipe


def test_docs_index_and_report_doc():
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8").lower()
    assert "recipe" in index and "when to use" in index
    report_doc = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    assert '"class"' in report_doc or "`class`" in report_doc
    assert "basis" in report_doc and "experimental" in report_doc


def test_version_is_at_least_1_22():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'^version = "(\d+)\.(\d+)\.(\d+)"$', text, flags=re.M)
    assert found and tuple(map(int, found.groups())) >= (1, 22, 0)
