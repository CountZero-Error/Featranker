"""Run the grid, each config in a fresh subprocess, appending to results.jsonl.

usage: python run_grid.py [name_substring ...]   (no args = every config)
"""
import itertools
import json
import os
import subprocess
import sys
import time

import psutil
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "results.jsonl"
SCRATCH = Path(os.environ.get("TMPDIR", "/tmp")) / "memscratch"
SCRATCH.mkdir(exist_ok=True)
TIMEOUT, TIMEOUT_EVIDENCE = 120, 180
GUARD_BYTES = 8 * 1024**3
ONE_HOT_SKIP_GB = 4  # dense one-hot estimate above which the config is not attempted


def numeric(rows, cols, **opt):
    return {"kind": "numeric", "rows": rows, "cols": cols, "shadows": False,
            "evidence": False, "max_samples": None, **opt}


def mixed(rows, card, **opt):
    return {"kind": "mixed", "rows": rows, "cols": 50, "ncat": 10, "card": card,
            "shadows": False, "evidence": False, "max_samples": None, **opt}


def grid():
    out = []
    for rows, cols in itertools.product((10_000, 100_000), (20, 100, 500)):
        out.append(numeric(rows, cols))
    out.append(numeric(500_000, 20))
    for rows, cols in ((10_000, 20), (10_000, 100), (10_000, 500), (100_000, 20), (100_000, 100)):
        out.append(numeric(rows, cols, shadows=True))
    for rows, cols in ((10_000, 20), (10_000, 100), (10_000, 500)):
        out.append(numeric(rows, cols, evidence=True))
    for rows, cols in ((100_000, 20), (100_000, 100), (100_000, 500), (500_000, 20)):
        out.append(numeric(rows, cols, max_samples=5000))
    for rows, cols in ((100_000, 20), (100_000, 100), (100_000, 500)):
        out.append(numeric(rows, cols, max_samples=5000, shadows=True, evidence=True))
    out.append(numeric(10_000, 500, kind="autogroups"))
    out.append(numeric(100_000, 500, kind="autogroups"))
    for rows, card in itertools.product((10_000, 100_000), (10, 1000, 10_000)):
        out.append(mixed(rows, card))
        out.append(mixed(rows, card, shadows=True))
        out.append(mixed(rows, card, max_samples=5000))
    out.append(mixed(2_000, 10_000))  # scaled-down probe: 10k cardinality is skipped at 10k rows
    return out


def probes():
    """Memory probes: one group, one repeat (see measure.py). Fit time and RSS are real."""
    a = [numeric(10_000, 500), numeric(10_000, 500, shadows=True), numeric(10_000, 100, shadows=True),
         numeric(10_000, 100, evidence=True), numeric(10_000, 500, evidence=True),
         numeric(100_000, 20), numeric(100_000, 20, shadows=True), numeric(100_000, 20, max_samples=5000),
         numeric(100_000, 20, max_samples=5000, shadows=True, evidence=True),
         numeric(100_000, 100), numeric(100_000, 100, shadows=True), numeric(100_000, 100, max_samples=5000),
         numeric(100_000, 500), numeric(100_000, 500, max_samples=5000),
         numeric(100_000, 500, max_samples=5000, shadows=True, evidence=True),
         numeric(500_000, 20), numeric(500_000, 20, max_samples=5000)]
    b = [mixed(10_000, 10, shadows=True), mixed(10_000, 1000), mixed(10_000, 1000, shadows=True),
         mixed(10_000, 1000, max_samples=5000), mixed(100_000, 10), mixed(100_000, 10, shadows=True),
         mixed(100_000, 10, max_samples=5000), mixed(100_000, 1000), mixed(10_000, 10_000),
         mixed(2_000, 10_000), mixed(2_000, 10_000, shadows=True)]
    return [dict(c, probe=True) for c in (a, b)[LANE]]


def overhead():
    """Framework-only memory: GaussianNB as the sole model, probe-style rank (see measure.py)."""
    cfgs = [numeric(r, c, **o) for r, c, o in (
        (100_000, 100, {}), (100_000, 100, {"shadows": True}), (100_000, 500, {}),
        (100_000, 500, {"shadows": True}), (100_000, 500, {"max_samples": 5000}),
        (100_000, 500, {"shadows": True, "max_samples": 5000}),
        (100_000, 500, {"shadows": True, "evidence": True, "max_samples": 5000}),
        (100_000, 500, {"evidence": True}), (500_000, 20, {}), (500_000, 20, {"shadows": True}))]
    cfgs += [mixed(100_000, 10), mixed(100_000, 10, shadows=True), mixed(10_000, 1000),
             mixed(10_000, 1000, shadows=True)]
    return [dict(c, probe=True, nb_only=True) for c in cfgs]


def name(cfg):
    parts = [cfg["kind"], f"{cfg['rows']//1000}k"]
    if cfg.get("nb_only"):
        parts.insert(0, "overhead")
    elif cfg.get("probe"):
        parts.insert(0, "probe")
    parts.append(f"c{cfg['cols']}" if cfg["kind"] != "mixed" else f"card{cfg['card']}")
    for key, tag in (("shadows", "shadows"), ("evidence", "evidence")):
        if cfg[key]:
            parts.append(tag)
    if cfg["max_samples"]:
        parts.append(f"ms{cfg['max_samples']}")
    return "-".join(parts)


def one_hot_gb(cfg):
    if cfg["kind"] != "mixed":
        return 0
    cats = cfg["ncat"] * (2 if cfg["shadows"] else 1)
    return cfg["rows"] * cats * min(cfg["card"], cfg["rows"]) * 8 / 1024**3


def run(cfg):
    label = name(cfg)
    record = {"name": label, "cfg": cfg}
    if one_hot_gb(cfg) > ONE_HOT_SKIP_GB:
        record.update(status="skipped", note=f"dense one-hot estimate {one_hot_gb(cfg):.1f} GB per copy")
        return record
    progress, result = SCRATCH / f"{label}.progress.json", SCRATCH / f"{label}.result.json"
    for path in (progress, result):
        path.unlink(missing_ok=True)
    timeout = TIMEOUT_EVIDENCE if cfg["evidence"] else TIMEOUT
    began = time.perf_counter()
    child = subprocess.Popen(
        [sys.executable, str(HERE / "measure.py"), json.dumps(cfg), str(progress), str(result)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": os.environ["MEM_PYTHONPATH"]})
    watcher, polled_peak, status = psutil.Process(child.pid), 0, None
    while child.poll() is None:
        try:  # parent-side RSS polling: reliable even if the child is stuck in C code
            polled_peak = max(polled_peak, watcher.memory_info().rss)
        except psutil.Error:
            pass
        if polled_peak > GUARD_BYTES:
            status = "rss_guard_8GB"
        elif time.perf_counter() - began > timeout:
            status = f"timeout_{timeout}s"
        if status:
            child.kill()
            break
        time.sleep(0.2)
    stderr = child.communicate()[1]
    if status is None:
        status = "ok" if child.returncode == 0 else "error"
        if status == "error":
            record["stderr"] = stderr[-600:]
    record["polled_peak_mb"] = round(polled_peak / 1024**2)
    record["status"] = status
    record["wall_seconds"] = round(time.perf_counter() - began, 1)
    if result.exists():
        record.update(json.loads(result.read_text()))
    if progress.exists():
        last = json.loads(progress.read_text())
        record["last_progress"] = {k: last[k] for k in ("t", "phase", "rss_mb", "maxrss_mb")}
        record.setdefault("events", last["events"])
    return record


if __name__ == "__main__":
    filters = sys.argv[1:]
    LANE = None
    if filters and filters[0] == "overhead":
        LANE = 2
    elif filters and filters[0] in ("probeA", "probeB"):
        LANE = 0 if filters[0] == "probeA" else 1
    for cfg in (overhead() if LANE == 2 else probes() if LANE is not None else grid()):
        if LANE is None and filters and not any(f in name(cfg) for f in filters):
            continue
        record = run(cfg)
        with OUT.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(record["name"], record["status"], record.get("seconds"), record.get("peak_rss_mb") or record.get("last_progress"), flush=True)
