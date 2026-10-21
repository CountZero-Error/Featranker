"""
Calibration study for FeatRanker's shadow rules (design: openspec/changes/calibration-study).

Null study (target independent of the features) measures how often each rule selects
noise and how the shadow hits test p-values are distributed; signal study (the
benchmark generator) measures recall. Every rule is applied to the stability table of
ONE report per configuration. Decisions Q, F, S are computed here, not by hand.

Not part of the package and not run by CI. Freeze: the script writes its own SHA-256
into the result file; docs/calibration.md records the hash before the full run.

    PYTHONPATH=<frozen featranker copy> python benchmarks/calibration.py --null-seeds 100 \
        --signal-seeds 100 --n-rows 300 --out <scratch>.json      # smoke run
    PYTHONPATH=<frozen featranker copy> python benchmarks/calibration.py   # full run
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import run  # the benchmark generator (benchmarks/run.py)

HERE = Path(__file__).resolve().parent
N_ROWS = 1000
N_FEATURES = 10
TASKS = ("reg", "clf")
GROUPS = ("fast", "tree")
MODES = ("joint", "independent")
NULL_SEEDS = list(range(100, 120))
SIGNAL_SEEDS = list(range(100, 110))
CV_SPLITS, CV_REPEATS, PERMUTATION_REPEATS = 5, 2, 10
FROZEN_COMMIT = "a1c7380"

# D3 constants
P_LEVEL = 0.05  # a "hit" of the shadow test: shadow_p_value <= 0.05
Q_MAX_FRACTION, Q_MAX_UPPER = 0.05, 0.075
F_MAX_NULL, F_MIN_RECALL = 0.05, 0.8
S_NULL_LIMIT = 0.05
TOLERANCE = 1e-9

# Rule name -> predicate on one stability row, exactly as ConsensusSelector.fit applies them.
RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "share": lambda r: r["above_shadow_share"] >= 0.5,
    "q": lambda r: r["shadow_q_value"] <= 0.05,
    "fam1": lambda r: r["shadow_families"] >= 1,
    "fam2": lambda r: r["shadow_families"] >= 2,
    "fam3": lambda r: r["shadow_families"] >= 3,
}
FAMILY_RULES = {1: "fam1", 2: "fam2", 3: "fam3"}


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# D1: data

def make_null(task: str, seed: int, n_rows: int = N_ROWS) -> run.Data:
    """Standard-normal features; the target is drawn independently of them."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, N_FEATURES))
    y = rng.standard_normal(n_rows) if task == "reg" else rng.integers(0, 2, n_rows)
    names = [f"f{i}" for i in range(N_FEATURES)]
    return run.Data(X, y, names, {"signal": [], "neutral": [], "noise": list(names)})


def make_study_data(study: str, task: str, seed: int, n_rows: int) -> run.Data:
    return make_null(task, seed, n_rows) if study == "null" else run.make_data(task, seed, False, n_rows)


def featranker_report(task: str, X: np.ndarray, y: np.ndarray, names: List[str], group: str,
                      mode: str, seed: int) -> Any:
    from featranker import FeatureRanker
    from sklearn.model_selection import RepeatedKFold

    ranker = FeatureRanker(task, group, n_jobs=-1, verbose=False, shadows=True,
                           shadow_seed=seed, shadow_mode=mode)
    return ranker.rank_resampled(
        X, y, scoring="r2" if task == "reg" else "roc_auc",
        cv=RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=seed),
        feature_names=names, n_repeats=PERMUTATION_REPEATS, random_state=seed,
    )


def apply_rules(stability: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    return {rule: [r["feature_group"] for r in stability if test(r)] for rule, test in RULES.items()}


STABILITY_KEYS = ("feature_group", "above_shadow_share", "shadow_p_value", "shadow_q_value", "shadow_families")


def run_study(study: str, tasks: List[str], groups: List[str], modes: List[str], seeds: List[int],
              ranker: Callable[..., Any] = featranker_report, n_rows: int = N_ROWS,
              on_run: Optional[Callable[[Dict[str, Any]], None]] = None,
              skip: Optional[set] = None) -> List[Dict[str, Any]]:
    """One ranker call per (task, group, mode, seed); all rules read that call's stability table."""
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for group in groups:
            for mode in modes:
                for seed in seeds:
                    if skip and (study, task, group, mode, seed) in skip:
                        continue
                    data = make_study_data(study, task, seed, n_rows)
                    row: Dict[str, Any] = {"study": study, "task": task, "group": group, "mode": mode,
                                           "seed": seed, "roles": data.roles}
                    start = time.perf_counter()
                    try:
                        report = ranker(task, data.X, data.y, data.names, group, mode, seed)
                        stability = [{k: r[k] for k in STABILITY_KEYS} for r in report["stability"]]
                        row.update(status="ok", stability=stability, selected=apply_rules(stability))
                    except Exception as error:  # a failing configuration is a result, not a crash
                        row.update(status="error", error=f"{type(error).__name__}: {error}")
                    row["seconds"] = time.perf_counter() - start
                    rows.append(row)
                    if on_run:
                        on_run(row)
    return rows


# D2: summaries

def _ok(runs: List[Dict[str, Any]], study: str) -> Dict[tuple, List[Dict[str, Any]]]:
    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        if r["study"] == study and r["status"] == "ok":
            cells.setdefault((r["task"], r["group"], r["mode"]), []).append(r)
    return cells


def summarize_null(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from scipy.stats import binomtest

    out = []
    for (task, group, mode), items in _ok(runs, "null").items():
        n_groups = sum(len(r["stability"]) for r in items)
        hits = sum(s["shadow_p_value"] <= P_LEVEL for r in items for s in r["stability"])
        ci = binomtest(hits, n_groups).proportion_ci(confidence_level=0.95, method="exact")
        rules = {
            rule: {
                "frac_groups": sum(len(r["selected"][rule]) for r in items) / n_groups,
                "frac_runs_any": sum(bool(r["selected"][rule]) for r in items) / len(items),
            }
            for rule in RULES
        }
        out.append({"task": task, "group": group, "mode": mode, "n_runs": len(items), "n_groups": n_groups,
                    "hits": hits, "fraction": hits / n_groups, "ci_low": ci.low, "ci_high": ci.high,
                    "rules": rules})
    return out


def _role_of(roles: Dict[str, List[str]]) -> Dict[str, str]:
    return dict(zip(roles["signal"] + roles["neutral"], run.ROLE_KEYS))


def summarize_signal(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for (task, group, mode), items in _ok(runs, "signal").items():
        for rule in RULES:
            per_run = []
            for r in items:
                role = _role_of(r["roles"])
                chosen = {role.get(n, "noise") for n in r["selected"][rule]}
                noise = sum(role.get(n) is None for n in r["selected"][rule])
                per_run.append({
                    "recall": len(chosen & set("abcde")) / 5, "noise": noise, "copy": float("f" in chosen),
                    "interaction_recall": len(chosen & set("cd")) / 2, "e_recall": float("e" in chosen),
                })
            out.append({"task": task, "group": group, "mode": mode, "rule": rule, "n_runs": len(items),
                        **{k: float(np.mean([p[k] for p in per_run])) for k in per_run[0]}})
    return out


# D3: preregistered decisions

def decide_q(null_summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Q: per configuration, fraction of null groups with p <= 0.05 at most 0.05 and 95% upper bound at most 0.075."""
    out = []
    for s in null_summary:
        allowed = s["fraction"] <= Q_MAX_FRACTION + TOLERANCE and s["ci_high"] <= Q_MAX_UPPER + TOLERANCE
        where = f"`{s['group']}` models, {s['task']}, shadow_mode={s['mode']}"
        measured = f"{s['fraction']:.1%} (95% CI {s['ci_low']:.1%} to {s['ci_high']:.1%}, {s['hits']} of {s['n_groups']})"
        if allowed:
            text = (f"on this study's null data, the binomial reference values were conservative for built-in "
                    f"`{s['group']}` models")
        elif s["fraction"] > Q_MAX_FRACTION + TOLERANCE:
            text = f"measured excess: {measured} of null groups had shadow_p_value <= 0.05, against 5% nominal"
        else:
            text = (f"no conservativeness claim: {measured} of null groups had shadow_p_value <= 0.05, "
                    f"but the 95% upper bound is above {Q_MAX_UPPER}")
        out.append({"task": s["task"], "group": s["group"], "mode": s["mode"], "fraction": s["fraction"],
                    "ci_high": s["ci_high"], "allowed": allowed, "where": where, "measured": measured, "text": text})
    return out


def decide_f(null_summary: List[Dict[str, Any]], signal_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """F: per task, the smallest N in {1, 2, 3} with null group-selection <= 0.05 in every null configuration
    and interaction recall >= 0.8 in every signal configuration with group="fast" (both shadow modes)."""
    out: Dict[str, Any] = {}
    for task in dict.fromkeys(s["task"] for s in null_summary + signal_summary):
        table = {}
        for n, rule in FAMILY_RULES.items():
            nulls = [s["rules"][rule]["frac_groups"] for s in null_summary if s["task"] == task]
            recalls = [s["interaction_recall"] for s in signal_summary
                       if s["task"] == task and s["group"] == "fast" and s["rule"] == rule]
            null_max = max(nulls) if nulls else None
            recall_min = min(recalls) if recalls else None
            passes = (null_max is not None and recall_min is not None
                      and null_max <= F_MAX_NULL + TOLERANCE and recall_min >= F_MIN_RECALL - TOLERANCE)
            table[n] = {"null_max": null_max, "interaction_recall_min": recall_min, "passes": passes}
        passing = [n for n, t in table.items() if t["passes"]]
        out[task] = {"recommended": passing[0] if passing else None, "table": table}
    return out


def decide_s(null_summary: List[Dict[str, Any]], signal_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    """S: recommend independent shadows for collinear data iff (pooled over tasks, groups and rules) it has a higher
    mean recall of e OR a higher copy-kept rate than joint, AND no (task, group, rule) null selection fraction is
    pushed above 0.05 (independent > 0.05 and > joint)."""
    def pooled(mode: str, key: str) -> float:
        return float(np.mean([s[key] for s in signal_summary if s["mode"] == mode]))

    gains = {key: pooled("independent", key) - pooled("joint", key) for key in ("e_recall", "copy")}
    joint = {(s["task"], s["group"], rule): v["frac_groups"]
             for s in null_summary if s["mode"] == "joint" for rule, v in s["rules"].items()}
    raised = [{"task": s["task"], "group": s["group"], "rule": rule, "joint": joint.get((s["task"], s["group"], rule)),
               "independent": v["frac_groups"]}
              for s in null_summary if s["mode"] == "independent" for rule, v in s["rules"].items()
              if v["frac_groups"] > S_NULL_LIMIT + TOLERANCE
              and v["frac_groups"] > joint.get((s["task"], s["group"], rule), 0.0) + TOLERANCE]
    gain = any(g > TOLERANCE for g in gains.values())
    return {"recommend": bool(gain and not raised), "e_recall_gain": gains["e_recall"], "copy_gain": gains["copy"],
            "null_raised_above_limit": raised}


def decisions(null_summary: List[Dict[str, Any]], signal_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"Q": decide_q(null_summary), "F": decide_f(null_summary, signal_summary),
            "S": decide_s(null_summary, signal_summary)}


# Rendering

def render_markdown(results: Dict[str, Any]) -> str:
    rule_names = list(RULES)
    lines = ["**Null study: shadow_p_value hits and noise groups selected (fraction of all null groups / fraction of runs with at least one)**", "",
             "| Task | Group | Mode | Runs | p <= 0.05 | 95% CI | " + " | ".join(rule_names) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(rule_names)]
    for s in results["null_summary"]:
        cells = [f"{s['rules'][r]['frac_groups']:.3f} / {s['rules'][r]['frac_runs_any']:.2f}" for r in rule_names]
        lines.append(f"| {s['task']} | {s['group']} | {s['mode']} | {s['n_runs']} | {s['fraction']:.3f} "
                     f"({s['hits']}/{s['n_groups']}) | {s['ci_low']:.3f} to {s['ci_high']:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "**Signal study: mean over seeds**", "",
              "| Task | Group | Mode | Rule | Recall | Noise selected | Copy kept | c,d recall | e recall | Runs |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in results["signal_summary"]:
        lines.append(f"| {s['task']} | {s['group']} | {s['mode']} | {s['rule']} | {s['recall']:.2f} | {s['noise']:.2f} | "
                     f"{s['copy']:.2f} | {s['interaction_recall']:.2f} | {s['e_recall']:.2f} | {s['n_runs']} |")
    return "\n".join(lines) + "\n"


def render_decisions(decided: Dict[str, Any]) -> str:
    lines = ["**Decision Q (hits test), per configuration**", ""]
    lines += [f"- {q['where']}: {q['measured']} -> {'ALLOWED' if q['allowed'] else 'NOT allowed'}: {q['text']}"
              for q in decided["Q"]]
    lines += ["", "**Decision F (min_shadow_families), per task**", ""]
    for task, f in decided["F"].items():
        lines.append(f"- {task}: recommended N = {f['recommended']}")
        for n, t in f["table"].items():
            nm = "n/a" if t["null_max"] is None else f"{t['null_max']:.3f}"
            rm = "n/a" if t["interaction_recall_min"] is None else f"{t['interaction_recall_min']:.2f}"
            lines.append(f"  - N={n}: worst null group-selection fraction {nm}, worst fast-group c,d recall {rm}, "
                         f"{'passes' if t['passes'] else 'fails'}")
    s = decided["S"]
    lines += ["", "**Decision S (shadow_mode)**", "",
              f"- independent recommended for collinear data: {s['recommend']} (pooled gain in e recall "
              f"{s['e_recall_gain']:+.3f}, in copy kept {s['copy_gain']:+.3f}; null fractions raised above "
              f"{S_NULL_LIMIT}: {len(s['null_raised_above_limit'])})"]
    lines += [f"  - {r['task']} {r['group']} {r['rule']}: joint {r['joint']:.3f} -> independent {r['independent']:.3f}"
              for r in s["null_raised_above_limit"]]
    return "\n".join(lines) + "\n"


def summarize_all(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    null, signal = summarize_null(runs), summarize_signal(runs)
    return {"null_summary": null, "signal_summary": signal, "decisions": decisions(null, signal)}


# Environment

def _frozen_version(commit: str) -> Optional[str]:
    """Version from pyproject.toml at the frozen commit (installed metadata can be stale)."""
    root = HERE.parent
    text = subprocess.run(["git", "show", f"{commit}:pyproject.toml"], cwd=root, capture_output=True, text=True).stdout
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    return match.group(1) if match else None


def environment(commit: str) -> Dict[str, Any]:
    import featranker

    package_dir = Path(featranker.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(path.relative_to(package_dir).as_posix().encode() + path.read_bytes())
    versions = {}
    for package in run.PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "featranker": {"commit": commit, "version": _frozen_version(commit), "imported_from": str(package_dir),
                       "package_sha256": digest.hexdigest()},
        "run_py": {"imported_from": str(Path(run.__file__).resolve()),
                   "sha256": hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest()},
        "packages": versions, "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
    }


def main() -> None:
    ints = lambda text: [int(s) for s in text.split(",")]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--null-seeds", default=",".join(map(str, NULL_SEEDS)))
    parser.add_argument("--signal-seeds", default=",".join(map(str, SIGNAL_SEEDS)))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--groups", default=",".join(GROUPS), help="groups for the signal study")
    parser.add_argument("--null-groups", default=",".join(GROUPS), help="groups for the null study (D4: drop tree first)")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--commit", default=FROZEN_COMMIT, help="commit the imported featranker copy was taken from")
    parser.add_argument("--note", default="")
    parser.add_argument("--drop", action="append", default=[], metavar="CONFIG=REASON")
    parser.add_argument("--resume", action="store_true", help="continue an existing --out file (same script hash)")
    parser.add_argument("--out", type=Path, default=HERE / "results" / f"{datetime.date.today():%Y%m%d}-calibration.json")
    args = parser.parse_args()

    tasks, modes = args.tasks.split(","), args.modes.split(",")
    sha = script_sha256()
    results: Dict[str, Any] = {"runs": []}
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text())
        if results["calibration_sha256"] != sha:
            parser.error("the result file was written by a different calibration.py; re-run from scratch")
    else:
        results.update(
            calibration_sha256=sha, environment=environment(args.commit), note=args.note,
            settings={"n_rows": args.n_rows, "n_features": N_FEATURES, "tasks": tasks, "modes": modes,
                      "signal_groups": args.groups.split(","), "null_groups": args.null_groups.split(","),
                      "null_seeds": ints(args.null_seeds), "signal_seeds": ints(args.signal_seeds),
                      "cv": f"RepeatedKFold({CV_SPLITS}, {CV_REPEATS})", "permutation_repeats": PERMUTATION_REPEATS},
            dropped=[dict(zip(("config", "reason"), d.split("=", 1))) for d in args.drop],
            started=datetime.datetime.now().isoformat(timespec="seconds"))
    done = {(r["study"], r["task"], r["group"], r["mode"], r["seed"]) for r in results["runs"] if r["status"] == "ok"}
    results["runs"] = [r for r in results["runs"] if r["status"] == "ok"]
    args.out.parent.mkdir(exist_ok=True)
    started = time.perf_counter()

    def on_run(row: Dict[str, Any]) -> None:  # rewrite after every run so a crash keeps what finished
        results["runs"].append(row)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"[*] {row['study']} {row['task']} {row['group']} {row['mode']} seed {row['seed']} "
              f"{row['status']} {row['seconds']:.1f}s {row.get('error', '')[:80]}", flush=True)

    plan = [("signal", g, ints(args.signal_seeds)) for g in args.groups.split(",")] + \
           [("null", g, ints(args.null_seeds)) for g in args.null_groups.split(",")]
    for study, group, seeds in plan:
        run_study(study, tasks, [group], modes, seeds, n_rows=args.n_rows, on_run=on_run, skip=done)
    results["total_seconds"] = time.perf_counter() - started + results.get("total_seconds", 0.0)
    results.update(summarize_all(results["runs"]))
    args.out.write_text(json.dumps(results, indent=2))
    text = render_markdown(results) + "\n" + render_decisions(results["decisions"])
    args.out.with_suffix(".md").write_text(text)
    print("\n" + text)
    print(f"[*] results: {args.out}  total {results['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
