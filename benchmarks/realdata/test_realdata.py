"""Acceptance check for the real-data study script (lead-written; builder must not edit)."""
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("realdata_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

ARMS = {"all", "fr_share", "fr_half", "rf_median"}


def test_datasets_match_prereg():
    table = (HERE / "PREREG.md").read_text(encoding="utf-8")
    ids = {int(m) for m in re.findall(r"^\| (\d+) \|", table, flags=re.M)}
    assert len(ids) == 13
    assert {d["data_id"] for d in run.DATASETS} == ids
    tasks = {int(m[0]): m[1] for m in re.findall(r"^\| (\d+) \| [^|]+ \| (clf|reg) \|", table, flags=re.M)}
    assert {d["data_id"]: d["task"] for d in run.DATASETS} == tasks


def _check_rows(rows, n_features):
    assert {r["arm"] for r in rows} == ARMS
    assert len({r["model"] for r in rows}) == 3
    for r in rows:
        assert set(r) >= {"arm", "model", "seed", "test_score", "n_kept", "selector_seconds", "error"}
        if r["error"] is None:
            assert math.isfinite(r["test_score"])
            assert 1 <= r["n_kept"] <= n_features
        else:
            assert r["test_score"] is None
    all_rows = [r for r in rows if r["arm"] == "all"]
    assert all(r["n_kept"] == n_features and r["error"] is None for r in all_rows)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("task", ["clf", "reg"])
def test_run_dataset_smoke(task):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 6))
    X[rng.random(X.shape) < 0.02] = np.nan  # missing cells are imputed, not fatal
    signal = X[:, 0] + X[:, 1]
    signal = np.nan_to_num(signal)
    y = (signal > 0).astype(int) if task == "clf" else signal + 0.1 * rng.normal(size=240)
    names = [f"x{i}" for i in range(6)]
    rows = run.run_dataset(X, y, names, task=task, seed=0, smoke=True)
    _check_rows(rows, 6)
    assert all(r["seed"] == 0 for r in rows)


def test_summary_counts_within_and_worse():
    rows = [
        {"dataset": "d", "seed": s, "arm": arm, "model": "m", "test_score": score,
         "n_kept": 1, "selector_seconds": 0.0, "error": None}
        for s in (0, 1, 2)
        for arm, score in (("all", 0.80), ("fr_share", 0.795), ("fr_half", 0.70), ("rf_median", 0.85))
    ]
    summary = run.summarize_results(rows)
    entry = {(e["dataset"], e["model"], e["arm"]): e for e in summary["per_dataset"]}
    assert entry[("d", "m", "fr_share")]["mean_delta"] == pytest.approx(-0.005)
    counts = summary["counts"]
    assert counts["fr_share"] == {"within": 1, "better": 0, "worse": 0}
    assert counts["fr_half"] == {"within": 0, "better": 0, "worse": 1}
    assert counts["rf_median"] == {"within": 0, "better": 1, "worse": 0}
