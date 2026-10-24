"""One configuration, one process: python measure.py '<config json>' <progress_path> <result_path>.

Config keys: kind (numeric|mixed|autogroups), rows, cols, ncat, card, shadows,
evidence, max_samples (and, for 1.21+, max_categories, max_train_samples). Package code is untouched; a few internals are wrapped
only to log time and current RSS around them.
"""
import json
import os
import resource
import sys
import threading
import time

import numpy as np
import psutil

MB = 1024 * 1024
GUARD_BYTES = 8 * 1024**3  # abort a config whose RSS passes this

cfg = json.loads(sys.argv[1])
progress_path, result_path = sys.argv[2], sys.argv[3]
proc = psutil.Process()
start = time.perf_counter()
state = {"phase": "import", "events": []}


def maxrss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


def snapshot():
    return {"t": round(time.perf_counter() - start, 2), "phase": state["phase"],
            "rss_mb": round(proc.memory_info().rss / MB), "maxrss_mb": round(maxrss() / MB)}


def write_progress():
    with open(progress_path, "w") as handle:
        json.dump({**snapshot(), "events": state["events"]}, handle)


def sampler():
    while True:
        if proc.memory_info().rss > GUARD_BYTES:
            state["phase"] += "+GUARD"
            write_progress()
            os._exit(87)
        write_progress()
        time.sleep(0.5)


def log(event):
    state["events"].append({"event": event, **snapshot()})


threading.Thread(target=sampler, daemon=True).start()

import featranker  # noqa: E402
import featranker.evidence as evidence_mod  # noqa: E402
import featranker.importance as imp  # noqa: E402

assert "memhead" in featranker.__file__, featranker.__file__  # must be the frozen HEAD copy
state["events"].append({"featranker_file": featranker.__file__})


def wrap(module, name):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        before = snapshot()
        began = time.perf_counter()
        result = original(*args, **kwargs)
        state["events"].append({
            "event": name, "seconds": round(time.perf_counter() - began, 2),
            "rss_before_mb": before["rss_mb"], "rss_after_mb": snapshot()["rss_mb"],
            "maxrss_after_mb": round(maxrss() / MB)})
        return result

    setattr(module, name, wrapper)


for fn in ("_validate_X", "_encode_categorical", "_with_shadows", "_fingerprint",
           "shap_evidence", "correlation_evidence"):
    wrap(imp, fn)  # importance imported these names, so patch its namespace

if cfg.get("probe"):
    # Probe mode (harness only, package untouched): score ONE group with ONE repeat so
    # memory and fit time are real while the O(groups x repeats) rank loop is skipped.
    imp._resolve_feature_groups = lambda feature_names, feature_groups: {"probe": [0]}

# --- data -------------------------------------------------------------
state["phase"] = "gen"
rng = np.random.default_rng(0)
n = cfg["rows"]
kind = cfg["kind"]
CAT_MIN_LEVEL_ROWS = 1


def numeric_block(rows, cols):
    return rng.standard_normal((rows, cols))


def labels(num, cat_codes=None):
    score = num[:, :5].sum(axis=1) + rng.standard_normal(len(num))
    if cat_codes is not None:
        score = score + (cat_codes[0] % 2)
    return (score > np.median(score)).astype(int)


def make(rows):
    if kind == "mixed":
        import pandas as pd
        cols, ncat, card = cfg["cols"], cfg["ncat"], cfg["card"]
        num = numeric_block(rows, cols)
        frame = pd.DataFrame(num, columns=[f"n{i}" for i in range(cols)])
        pool = np.array([f"lvl{i}" for i in range(card)], dtype=object)  # shared str objects
        codes = []
        for j in range(ncat):
            code = rng.integers(0, card, rows)
            codes.append(code)
            frame[f"c{j}"] = pool[code]
        # numeric columns' bytes + one pointer per categorical cell (strings are shared)
        nbytes = num.nbytes + rows * ncat * 8
        return frame, labels(num, codes), nbytes
    num = numeric_block(rows, cfg["cols"])
    return num, labels(num), num.nbytes


X, y, x_bytes = make(n)
X_eval, y_eval, xe_bytes = make(n)
PROBE = bool(cfg.get("probe"))
names = list(X.columns) if kind == "mixed" else [f"f{i}" for i in range(cfg["cols"])]
categorical = [c for c in names if c.startswith("c")] if kind == "mixed" else None
state["phase"] = "gen_done"
rss_before = round(proc.memory_info().rss / MB)
gen_peak = round(maxrss() / MB)

result = {"cfg": cfg, "input_train_mb": round(x_bytes / MB, 1),
          "input_total_mb": round((x_bytes + xe_bytes) / MB, 1),
          "rss_before_fit_mb": rss_before, "gen_peak_mb": gen_peak}

# --- run --------------------------------------------------------------
if kind == "autogroups":
    state["phase"] = "auto_groups"
    began = time.perf_counter()
    groups = featranker.auto_groups(X, feature_names=names)
    result["seconds"] = round(time.perf_counter() - began, 2)
    result["n_groups"] = len(groups)
else:
    extra = {}
    if cfg.get("nb_only"):  # framework overhead: one cheap estimator, so fit/score time is negligible
        from sklearn.naive_bayes import GaussianNB
        extra = {"models": {"naive_bayes": GaussianNB()}}
    # 1.21 options, passed only when set so the same script measures older copies.
    cap = {"max_categories": cfg["max_categories"]} if cfg.get("max_categories") else {}
    train_sample = (
        {"max_samples": cfg["max_train_samples"], "random_state": 0}
        if cfg.get("max_train_samples") else {}
    )
    ranker = featranker.FeatureRanker(
        "clf", **({"group": "fast"} if not extra else {}), **extra, verbose=False, shadows=cfg["shadows"], shadow_seed=0,
        evidence=cfg["evidence"], categorical=categorical, **cap)
    state["phase"] = "fit"
    log("fit_start")
    began = time.perf_counter()
    ranker.fit(X, y, feature_names=None if kind == "mixed" else names, **train_sample)
    result["fit_seconds"] = round(time.perf_counter() - began, 2)
    result["rss_after_fit_mb"] = round(proc.memory_info().rss / MB)
    result["peak_after_fit_mb"] = round(maxrss() / MB)
    state["phase"] = "rank"
    began = time.perf_counter()
    report = ranker.rank_features(
        X_eval, y_eval, scoring="roc_auc", n_repeats=1 if PROBE else 3, random_state=0,
        max_samples=cfg.get("max_samples"))
    result["rank_seconds"] = round(time.perf_counter() - began, 2)
    result["seconds"] = round(result["fit_seconds"] + result["rank_seconds"], 2)
    result["models_ranked"] = len(report["models"])
    result["failures"] = [f"{f['model']}:{f['stage']}" for f in report["failures"]][:12]

result["peak_rss_mb"] = round(maxrss() / MB)
result["events"] = state["events"]
state["phase"] = "done"
write_progress()
with open(result_path, "w") as handle:
    json.dump(result, handle)
