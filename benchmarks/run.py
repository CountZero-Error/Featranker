"""
Benchmark FeatRanker's selector against common feature selectors on synthetic
data with known signal, noise, and one correlated copy.

Not part of the package and not run by CI. Optional competitors (boruta,
feature-engine) are imported lazily; a missing one is recorded, not fatal.

    PYTHONPATH=. python benchmarks/run.py --seeds 0 --n-rows 300   # smoke run
    PYTHONPATH=. python benchmarks/run.py                          # full run
    python benchmarks/run.py --render benchmarks/results/<date>.json
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent


def _shown_path(path: Path) -> str:
    """A path for the results file: repo-relative, never the machine's absolute path."""
    try:
        return path.resolve().relative_to(HERE.parent).as_posix()
    except ValueError:
        return f"<outside repo>/{path.parent.name}/{path.name}"
N_ROWS = 1000
N_FEATURES = 10
N_TREES = 100
TASKS = ("reg", "clf")
VARIANTS = ("standard", "hard")
ROLE_KEYS = ("a", "b", "c", "d", "e", "f")  # signal roles in order, then the copy
NOISE_SD = {False: 1.0, True: 2.0}  # the hard variant doubles the noise sd
PERMUTATION_RULE_STD = 2.0
PACKAGES = (
    "numpy", "scipy", "scikit-learn", "pandas", "joblib",
    "xgboost", "catboost", "boruta", "feature-engine",
)


class Data(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    names: List[str]
    roles: Dict[str, List[str]]  # signal / neutral / noise -> column names


# D1: data generator

def make_data(task: str, seed: int, hard: bool, n_rows: int = N_ROWS) -> Data:
    """Role order a, b, c, d, e, f(copy of e), four noise columns; placed at seeded columns."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_rows, N_FEATURES))
    a, b, c, d, e = (z[:, i] for i in range(5))
    z[:, 5] = e + 0.1 * rng.standard_normal(n_rows)
    eps = rng.standard_normal(n_rows)
    draws = rng.random(n_rows)
    place = rng.permutation(N_FEATURES)  # role i lives in column place[i]
    X = np.empty_like(z)
    X[:, place] = z
    names = [f"f{i}" for i in range(N_FEATURES)]
    by_role = [names[p] for p in place]
    signal = 2 * a + 3 * (b > 0) + 2 * c * d + 1.5 * e
    sd = NOISE_SD[hard]
    if task == "reg":
        y = signal + sd * eps
    else:
        logit = signal - 1.5 + sd * eps  # centred: the threshold term has mean 1.5
        y = (draws < 1 / (1 + np.exp(-logit))).astype(int)
    roles = {"signal": by_role[:5], "neutral": [by_role[5]], "noise": by_role[6:]}
    return Data(X, y, names, roles)


# D2: scoring

def score(data: Data, selected: List[int]) -> Dict[str, Any]:
    chosen = {data.names[i] for i in selected}
    signal = data.roles["signal"]
    return {
        "recall": len(chosen & set(signal)) / len(signal),
        "noise_selected": len(chosen & set(data.roles["noise"])),
        "neutral_selected": len(chosen & set(data.roles["neutral"])),
        "n_selected": len(chosen),
    }


# D3: selectors. Each is fn(task, X, y, seed) -> selected column indices.

def _forest(task: str, seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    cls = RandomForestRegressor if task == "reg" else RandomForestClassifier
    return cls(n_estimators=N_TREES, n_jobs=-1, random_state=seed)


def _shadow_diagnostics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Shadow shares per feature and per family, keyed by the selector's x0, x1, ... names.

    Per family: the fraction of its voting members above the shadows, per split, averaged over splits.
    """
    groups = {row["feature_group"]: row["above_shadow_share"] for row in report["stability"]}
    per_family: Dict[str, Dict[str, List[float]]] = {}
    for split in report["splits"]:
        flags: Dict[str, Dict[str, List[bool]]] = {}
        for model in split["report"]["models"].values():
            if not model["in_consensus"]:
                continue
            for group, entry in model["importance"].items():
                flags.setdefault(model["family"], {}).setdefault(group, []).append(entry["above_shadow"])
        for family, by_group in flags.items():
            for group, values in by_group.items():
                per_family.setdefault(family, {}).setdefault(group, []).append(float(np.mean(values)))
    families = {f: {g: float(np.mean(v)) for g, v in by_group.items()} for f, by_group in per_family.items()}
    return {"above_shadow_share": groups, "family_shares": families}


def _featranker(rule: str, group: str) -> Callable[..., List[int]]:
    def select(task: str, X: np.ndarray, y: np.ndarray, seed: int) -> List[int]:
        from sklearn.model_selection import RepeatedKFold

        from featranker import ConsensusSelector

        common: Dict[str, Any] = dict(
            task=task, scoring="r2" if task == "reg" else "roc_auc", group=group,
            n_jobs=-1, random_state=seed, shadow_seed=seed,
        )
        if rule == "share":
            sel = ConsensusSelector(min_shadow_share=0.5, cv=5, **common)
        else:
            sel = ConsensusSelector(
                max_shadow_q=0.05, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed), **common
            )
        sel.fit(X, y)
        selected = [int(i) for i in np.flatnonzero(sel.support_)]
        return (selected, _shadow_diagnostics(sel.report_)) if rule == "share" else selected

    return select


def boruta(task: str, X: np.ndarray, y: np.ndarray, seed: int) -> List[int]:
    from boruta import BorutaPy

    # Confirmed features only (support_); tentative ones are not selected.
    return list(np.flatnonzero(BorutaPy(_forest(task, seed), n_estimators=N_TREES, random_state=seed).fit(X, y).support_))


def permutation_rule(task: str, X: np.ndarray, y: np.ndarray, seed: int) -> List[int]:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y if task == "clf" else None
    )
    forest = _forest(task, seed).fit(X_tr, y_tr)
    imp = permutation_importance(forest, X_te, y_te, random_state=seed, n_jobs=-1)
    return list(np.flatnonzero(imp.importances_mean - PERMUTATION_RULE_STD * imp.importances_std > 0))


def shuffling(task: str, X: np.ndarray, y: np.ndarray, seed: int) -> List[int]:
    import pandas as pd
    from feature_engine.selection import SelectByShuffling

    sel = SelectByShuffling(
        estimator=_forest(task, seed), scoring="r2" if task == "reg" else "roc_auc", random_state=seed
    )
    frame = pd.DataFrame(X, columns=range(X.shape[1]))  # feature-engine renames bare arrays to x0, x1, ...
    sel.fit(frame, pd.Series(y))
    dropped = set(sel.features_to_drop_)
    return [i for i in range(X.shape[1]) if i not in dropped]


SELECTORS: Dict[str, Callable[..., List[int]]] = {
    "featranker_share_fast": _featranker("share", "fast"),
    "featranker_share_all": _featranker("share", "all"),
    "featranker_q_fast": _featranker("q", "fast"),
    "featranker_q_all": _featranker("q", "all"),
    "boruta": boruta,
    "permutation_rule": permutation_rule,
    "select_by_shuffling": shuffling,
    # Post-hoc (design D6), not part of the preregistered set.
    "featranker_share_tree": _featranker("share", "tree"),
}
PREREGISTERED = tuple(name for name in SELECTORS if name != "featranker_share_tree")


# Running

def run_one(fn: Callable[..., List[int]], data: Data, task: str, seed: int) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        selected = fn(task, data.X, data.y, seed)
        diagnostics = None
        if isinstance(selected, tuple):
            selected, diagnostics = selected
    except ImportError as error:
        return {"status": "missing", "error": f"{type(error).__name__}: {error}"}
    except Exception as error:  # a failing competitor is a result, not a crash
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    seconds = time.perf_counter() - start
    selected = [int(i) for i in selected]
    row = {
        "status": "ok", "seconds": seconds, "selected": [data.names[i] for i in selected],
        **score(data, selected),
    }
    if diagnostics is not None:
        row["diagnostics"] = _label_diagnostics(data, diagnostics)
    return row


def role_labels(data: Data) -> Dict[str, str]:
    """Column name -> role label (a, b, c, d, e, f) for the six planted roles."""
    return dict(zip(data.roles["signal"] + data.roles["neutral"], ROLE_KEYS))


def _label_diagnostics(data: Data, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    label = {f"x{i}": role_labels(data).get(name) for i, name in enumerate(data.names)}
    keep = lambda shares: {label[g]: v for g, v in shares.items() if label.get(g)}
    return {
        "above_shadow_share": keep(diagnostics["above_shadow_share"]),
        "family_shares": {f: keep(v) for f, v in diagnostics["family_shares"].items()},
    }


def run_benchmark(
    tasks: List[str], variants: List[str], seeds: List[int],
    selectors: Optional[Dict[str, Callable[..., List[int]]]] = None,
    n_rows: int = N_ROWS, on_run: Optional[Callable[[Dict[str, Any]], None]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selectors = {n: SELECTORS[n] for n in PREREGISTERED} if selectors is None else selectors
    results: Dict[str, Any] = {**(extra or {}), "runs": []}
    for task in tasks:
        for variant in variants:
            for seed in seeds:
                data = make_data(task, seed, variant == "hard", n_rows)
                for name, fn in selectors.items():
                    row = {"task": task, "variant": variant, "seed": seed, "selector": name,
                           "roles": data.roles, **run_one(fn, data, task, seed)}
                    results["runs"].append(row)
                    if on_run:
                        on_run(row)
    return results


# Summary and table

def _mean_std(values: List[float]) -> List[float]:
    return [float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0]


def summarize(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per task/variant/selector: mean and sample std (ddof=1) over seeds."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["task"], r["variant"], r["selector"]), []).append(r)
    rows = []
    for (task, variant, selector), items in groups.items():
        ok = [r for r in items if r["status"] == "ok"]
        row: Dict[str, Any] = {"task": task, "variant": variant, "selector": selector, "n_runs": len(ok)}
        if not ok:
            row["status"] = items[0]["status"]
            row["error"] = items[0].get("error", "")
        else:
            row["status"] = "ok"
            for key, field in (("recall", "recall"), ("noise", "noise_selected"),
                               ("copy", "neutral_selected"), ("seconds", "seconds")):
                row[f"{key}_mean"], row[f"{key}_std"] = _mean_std([r[field] for r in ok])
        rows.append(row)
    return rows


def render_table(results: Dict[str, Any]) -> str:
    lines: List[str] = []
    rows = summarize(results["runs"])
    for task, variant in dict.fromkeys((r["task"], r["variant"]) for r in rows):
        title = {"reg": "Regression", "clf": "Classification"}[task]
        lines += [f"**{title}, {variant}**", "",
                  "| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |",
                  "|---|---|---|---|---|---|"]
        for r in (r for r in rows if (r["task"], r["variant"]) == (task, variant)):
            if r["status"] != "ok":
                lines.append(f"| {r['selector']} | {r['status']}: {r['error'][:60]} | | | | 0 |")
                continue
            cells = [f"{r[k + '_mean']:.2f} ± {r[k + '_std']:.2f}" for k in ("recall", "noise", "copy")]
            cells.append(f"{r['seconds_mean']:.1f} ± {r['seconds_std']:.1f}")
            lines.append(f"| {r['selector']} | " + " | ".join(cells) + f" | {r['n_runs']} |")
        lines.append("")
    return "\n".join(lines)


ROLE_LABELS = ("a (linear)", "b (threshold)", "c (interaction)", "d (interaction)", "e (linear)", "f (copy of e)")


def role_table(results: Dict[str, Any]) -> str:
    """Share of seeds in which each planted role was selected (roles from each run; re-derived from the seed only for files that predate them)."""
    n_rows = results.get("settings", {}).get("n_rows", N_ROWS)
    hits: Dict[tuple, List[List[int]]] = {}
    for r in results["runs"]:
        if r["status"] != "ok":
            continue
        # Files from before roles were stored (20261019.json) re-derive them from the seed.
        roles = r.get("roles") or make_data(r["task"], r["seed"], r["variant"] == "hard", n_rows).roles
        order = roles["signal"] + roles["neutral"]
        hits.setdefault((r["task"], r["variant"], r["selector"]), []).append(
            [int(name in r["selected"]) for name in order])
    lines: List[str] = []
    for task, variant in dict.fromkeys(k[:2] for k in hits):
        lines += [f"**{ {'reg': 'Regression', 'clf': 'Classification'}[task]}, {variant}: share of seeds in which each role was selected**", "",
                  "| Selector | " + " | ".join(ROLE_LABELS) + " |", "|---" * (len(ROLE_LABELS) + 1) + "|"]
        for (t, v, name), rows in hits.items():
            if (t, v) == (task, variant):
                lines.append(f"| {name} | " + " | ".join(f"{x:.1f}" for x in np.mean(rows, axis=0)) + " |")
        lines.append("")
    return "\n".join(lines)


def family_table(results: Dict[str, Any], roles: Tuple[str, ...] = ("c", "d")) -> str:
    """Per task/variant/selector: each family's above-shadow share for the given roles, mean over seeds."""
    cells: Dict[tuple, Dict[str, Dict[str, List[float]]]] = {}
    for r in results["runs"]:
        for family, by_role in r.get("diagnostics", {}).get("family_shares", {}).items():
            for role in roles:
                cells.setdefault((r["task"], r["variant"], r["selector"]), {}).setdefault(
                    family, {}).setdefault(role, []).append(by_role.get(role, 0.0))
    lines: List[str] = []
    for key, by_family in cells.items():
        title = {"reg": "Regression", "clf": "Classification"}[key[0]]
        lines += [f"**{title}, {key[1]}, {key[2]}: above-shadow share by family, mean over seeds**", "",
                  "| Family | " + " | ".join(roles) + " |", "|---" * (len(roles) + 1) + "|"]
        for family, by_role in sorted(by_family.items()):
            lines.append(f"| {family} | " + " | ".join(f"{np.mean(by_role[x]):.2f}" for x in roles) + " |")
        overall = results_shares(results, key, roles)
        lines.append("| **share used by the rule (average over families)** | " + " | ".join(f"{v:.2f}" for v in overall) + " |")
        lines.append("")
    return "\n".join(lines)


def results_shares(results: Dict[str, Any], key: tuple, roles: Tuple[str, ...]) -> List[float]:
    rows = [r["diagnostics"]["above_shadow_share"] for r in results["runs"]
            if (r["task"], r["variant"], r["selector"]) == key and "diagnostics" in r]
    return [float(np.mean([row.get(x, 0.0) for row in rows])) for x in roles]


def environment() -> Dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Installed metadata can be stale (PYTHONPATH=. uses the working tree), so read the tree itself.
    import featranker

    root = HERE.parent
    pyproject = (root / "pyproject.toml").read_text()
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    tree = {
        "version": re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1),
        "commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "featranker")),
        "imported_from": _shown_path(Path(featranker.__file__).parent),
    }
    return {
        "featranker": tree, "packages": versions, "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--selectors", default=",".join(PREREGISTERED),
                        help="default: the preregistered set; featranker_share_tree is post-hoc (design D6)")
    parser.add_argument("--note", default="", help="free text stored in the result file")
    parser.add_argument("--n-rows", type=int, default=N_ROWS)
    parser.add_argument("--out", type=Path, default=HERE / "results" / f"{datetime.date.today():%Y%m%d}.json")
    parser.add_argument("--drop", action="append", default=[], metavar="SELECTOR=REASON",
                        help="record a configuration left out of the run, with the reason")
    parser.add_argument("--render", type=Path, help="print the table of an existing result file and exit")
    args = parser.parse_args()

    if args.render:
        loaded = json.loads(args.render.read_text())
        print(render_table(loaded) + "\n" + role_table(loaded) + "\n" + family_table(loaded))
        return
    unknown = [s for s in args.selectors.split(",") if s not in SELECTORS]
    if unknown:
        parser.error(f"unknown selectors: {unknown}; choose from {list(SELECTORS)}")
    chosen = {s: SELECTORS[s] for s in args.selectors.split(",")}
    extra = {
        "environment": environment(),
        "settings": {
            "n_rows": args.n_rows, "n_features": N_FEATURES, "n_trees": N_TREES,
            "seeds": [int(s) for s in args.seeds.split(",")], "tasks": args.tasks.split(","),
            "variants": args.variants.split(","), "selectors": list(chosen),
            "noise_sd": {"standard": NOISE_SD[False], "hard": NOISE_SD[True]},
        },
        "note": args.note,
        "warm_up": True,
        "dropped": [dict(zip(("selector", "reason"), d.split("=", 1))) for d in args.drop],
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(exist_ok=True)
    # Warm-up: untimed calls on tiny data so imports and worker start-up are not charged to the first run.
    for task in extra["settings"]["tasks"]:
        warm = make_data(task, 0, False, 200)
        for fn in chosen.values():
            run_one(fn, warm, task, 0)
    start = time.perf_counter()
    results: Dict[str, Any] = {**extra, "runs": []}

    def on_run(row: Dict[str, Any]) -> None:  # rewrite after every run so a crash keeps what finished
        results["runs"].append(row)
        args.out.write_text(json.dumps(results, indent=2))
        detail = f"recall {row['recall']:.1f} noise {row['noise_selected']} {row['seconds']:.1f}s" \
            if row["status"] == "ok" else row["error"][:80]
        print(f"[*] {row['task']} {row['variant']} seed {row['seed']} {row['selector']:<24} {row['status']} {detail}", flush=True)

    run_benchmark(extra["settings"]["tasks"], extra["settings"]["variants"], extra["settings"]["seeds"],
                  chosen, args.n_rows, on_run)
    results["total_seconds"] = time.perf_counter() - start
    results["summary"] = summarize(results["runs"])
    args.out.write_text(json.dumps(results, indent=2))
    table = render_table(results)
    args.out.with_suffix(".md").write_text(table + "\n" + role_table(results) + "\n" + family_table(results))
    print("\n" + table)
    print(f"[*] results: {args.out}  total {results['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
