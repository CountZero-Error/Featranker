"""Markdown tables from results.jsonl (last record per config name wins).

Full runs (status ok) are measured end to end. Probe runs (name starts with
"probe-") score one group with one repeat: peak RSS and fit seconds are
measured; "est. rank s" is an extrapolation of the probe's rank time to the
full groups x repeats x shadows loop (evidence time, measured, is kept apart).
"""
import json
from pathlib import Path

MODELS = 7  # group="fast", classification
REPEATS = 3
rows = {}
for line in (Path(__file__).parent / "results.jsonl").read_text().splitlines():
    record = json.loads(line)
    rows[record["name"]] = record


def event_seconds(record, name):
    return sum(e.get("seconds", 0) for e in record.get("events", []) if e.get("event") == name)


def n_features(cfg):
    return cfg["cols"] + (cfg["ncat"] if cfg["kind"] == "mixed" else 0)


def peak_of(r):
    if r.get("peak_rss_mb"):
        return r["peak_rss_mb"], ""
    parent = r.get("polled_peak_mb") or 0
    child = (r.get("last_progress") or {}).get("maxrss_mb") or 0
    return max(parent, child), f" (>=, in {(r.get('last_progress') or {}).get('phase')})"


print("| config | status | input MB (train+eval) | RSS before fit MB | peak RSS MB | peak/input | (peak-before)/input | fit s | rank s (measured) | est. full rank s |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, r in rows.items():
    status = r["status"]
    if status == "skipped":
        print(f"| {name} | skipped: {r['note']} | | | | | | | | |")
        continue
    if "timeout" in status and not name.startswith("probe-") and r.get("polled_peak_mb") is None:
        continue  # first-pass timeouts, superseded by the probe rows (sampler-thread RSS was unreliable)
    peak, flag = peak_of(r)
    inp, before = r.get("input_total_mb"), r.get("rss_before_fit_mb")
    ratio = f"{peak / inp:.1f}" if peak and inp else ""
    extra = f"{(peak - before) / inp:.1f}" if peak and inp and before else ""
    rank = r.get("rank_seconds", "")
    est = ""
    cfg = r["cfg"]
    if status == "ok" and cfg.get("probe") and not cfg.get("nb_only") and rank != "":
        shadow = 2 if cfg["shadows"] else 1
        rank_core = rank - event_seconds(r, "shap_evidence") - event_seconds(r, "correlation_evidence")
        probe_units, full_units = 1 + shadow, 1 + n_features(cfg) * REPEATS * shadow
        est = f"{rank_core * full_units / probe_units:.0f}"
    print(f"| {name} | {status} | {inp or ''} | {before or ''} | {peak}{flag} | {ratio} | {extra} | {r.get('fit_seconds', '')} | {rank} | {est} |")
