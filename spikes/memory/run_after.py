"""The 1.21 before/after measurements (see FINDINGS.md, "After 1.21").

usage: MEM_PYTHONPATH=<copy of featranker> python run_after.py <label> <results.jsonl> [case ...]
Each case runs in a fresh subprocess (measure.py), one at a time. <label> only names
the output rows. Cases: base, shadows, evidence, autogroups, cap1000, train, train_both.
"""
import json
import sys

import run_grid
from run_grid import mixed, numeric, run

run_grid.TIMEOUT = run_grid.TIMEOUT_EVIDENCE = 300

CASES = {
    # framework only (GaussianNB is the only model), 100k x 500 float64, as in the study
    "base": [dict(numeric(100_000, 500), probe=True, nb_only=True)],
    "shadows": [dict(numeric(100_000, 500, shadows=True), probe=True, nb_only=True)],
    "evidence": [dict(numeric(100_000, 500, evidence=True), probe=True, nb_only=True)],
    "autogroups": [numeric(100_000, 500, kind="autogroups")],
    # 50 numeric + 10 categorical columns, real "fast" models, one group x one repeat
    "cap1000": [
        dict(mixed(4_000, 1000), probe=True),
        dict(mixed(4_000, 1000, max_categories=20), probe=True),
        dict(mixed(10_000, 1000, max_categories=20), probe=True),
        dict(mixed(10_000, 1000, shadows=True, max_categories=20), probe=True),
    ],
    # real "fast" models on 100k x 500 with the training rows capped
    "train": [dict(numeric(100_000, 500, max_train_samples=5000), probe=True)],
    "train_both": [dict(numeric(100_000, 500, max_train_samples=5000, max_samples=5000), probe=True)],
}

if __name__ == "__main__":
    label, out, *wanted = sys.argv[1:]
    run_grid.OUT = run_grid.Path(out)
    for case in wanted or list(CASES):
        for cfg in CASES[case]:
            record = dict(run(cfg), case=case, label=label)
            with run_grid.OUT.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(label, record["name"], record["status"], record.get("peak_rss_mb"),
                  record.get("fit_seconds"), record.get("rss_before_fit_mb"), flush=True)
