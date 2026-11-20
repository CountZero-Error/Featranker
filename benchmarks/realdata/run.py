"""
Real-data study for FeatRanker, as fixed in benchmarks/realdata/PREREG.md.

Not part of the package and not run by CI.

    PYTHONPATH=. python benchmarks/realdata/run.py --data-home ~/openml --out out.json
    PYTHONPATH=. python benchmarks/realdata/run.py --data-home ~/openml --out out.json --resume
    python benchmarks/realdata/run.py --render out.json
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import re
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG.md"

DATASETS: List[Dict[str, Any]] = [
    {"data_id": 6, "name": "letter", "task": "clf", "rows": 20000, "features": 16},
    {"data_id": 22, "name": "mfeat-zernike", "task": "clf", "rows": 2000, "features": 47},
    {"data_id": 44, "name": "spambase", "task": "clf", "rows": 4601, "features": 57},
    {"data_id": 458, "name": "analcatdata_authorship", "task": "clf", "rows": 841, "features": 70},
    {"data_id": 1053, "name": "jm1", "task": "clf", "rows": 10885, "features": 21},
    {"data_id": 1067, "name": "kc1", "task": "clf", "rows": 2109, "features": 21},
    {"data_id": 1487, "name": "ozone-level-8hr", "task": "clf", "rows": 2534, "features": 72},
    {"data_id": 1510, "name": "wdbc", "task": "clf", "rows": 569, "features": 30},
    {"data_id": 40499, "name": "texture", "task": "clf", "rows": 5500, "features": 40},
    {"data_id": 201, "name": "pol", "task": "reg", "rows": 15000, "features": 48},
    {"data_id": 216, "name": "elevators", "task": "reg", "rows": 16599, "features": 18},
    {"data_id": 287, "name": "wine_quality", "task": "reg", "rows": 6497, "features": 11},
    {"data_id": 42730, "name": "us_crime", "task": "reg", "rows": 1994, "features": 126},
]
ARMS = ("all", "fr_share", "fr_half", "rf_median")
MODELS = ("linear", "random_forest", "hist_gb")
TEST_SIZE = 0.3
MAX_SAMPLES = 5000
WITHIN = 0.01
EPS = 1e-9  # makes the 0.01 boundary inclusive despite float error
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "joblib")


class _Empty(Exception):
    """Raised when a selector keeps no features; reported as EmptySelection."""


def _scoring(task: str) -> str:
    return "balanced_accuracy" if task == "clf" else "r2"


def _downstream(model: str, task: str, seed: int, n_trees: int) -> Any:
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = task == "clf"
    imputer = SimpleImputer(strategy="median")
    if model == "linear":
        est = LogisticRegression(max_iter=2000, random_state=seed) if clf else Ridge(random_state=seed)
        return make_pipeline(imputer, StandardScaler(), est)
    if model == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return make_pipeline(imputer, cls(n_estimators=n_trees, random_state=seed, n_jobs=-1))
    cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
    return make_pipeline(imputer, cls(random_state=seed))


def _consensus(task: str, seed: int, smoke: bool, **extra: Any) -> Any:
    from featranker import ConsensusSelector

    return ConsensusSelector(
        task=task, scoring=_scoring(task), group="fast", cv=2 if smoke else 5,
        n_repeats=2 if smoke else 5, random_state=seed, shadow_seed=seed,
        max_samples=MAX_SAMPLES, max_train_samples=MAX_SAMPLES, missing="impute", n_jobs=-1, **extra,
    )


def _select(arm: str, X: np.ndarray, y: np.ndarray, task: str, seed: int, smoke: bool,
            n_trees: int) -> List[int]:
    """Column indices kept by the arm, fit on the training rows only."""
    if arm == "rf_median":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        from sklearn.impute import SimpleImputer

        cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
        sel = SelectFromModel(cls(n_estimators=n_trees, random_state=seed, n_jobs=-1), threshold="median")
        sel.fit(SimpleImputer(strategy="median").fit_transform(X), y)
        return [int(i) for i in np.flatnonzero(sel.get_support())]
    if arm == "fr_share":
        sel = _consensus(task, seed, smoke, min_shadow_share=0.5)
    else:  # fr_half: each feature is its own group, so n_groups = n_features
        sel = _consensus(task, seed, smoke, k=math.ceil(X.shape[1] / 2))
    sel.fit(X, y)
    return [int(i) for i in np.flatnonzero(sel.support_)]


def _row(arm: str, model: str, seed: int, score: Optional[float], n_kept: int, secs: float,
         error: Optional[str], dummy_score: Optional[float] = None) -> Dict[str, Any]:
    # dummy_score: set only on EmptySelection rows (test_score stays None); the summary scores
    # those rows with it (prior / mean predictor), so an empty selection counts as a result.
    return {"dataset": None, "seed": seed, "arm": arm, "model": model, "test_score": score,
            "n_kept": n_kept, "selector_seconds": secs, "error": error, "dummy_score": dummy_score}


def _dummy_score(task: str, scorer: Any, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                 y_te: np.ndarray) -> Optional[float]:
    from sklearn.dummy import DummyClassifier, DummyRegressor

    try:
        est = DummyClassifier(strategy="prior") if task == "clf" else DummyRegressor(strategy="mean")
        return float(scorer(est.fit(X_tr[:, :1], y_tr), X_te[:, :1], y_te))
    except Exception:
        return None


def run_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str], *, task: str, seed: int,
                smoke: bool = False) -> List[Dict[str, Any]]:
    """One 70/30 outer split; every arm x downstream model. Never raises for an arm failure."""
    from sklearn.metrics import get_scorer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    X = np.asarray(X, dtype=float)
    y = LabelEncoder().fit_transform(y) if task == "clf" else np.asarray(y, dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y if task == "clf" else None)
    scorer = get_scorer(_scoring(task))
    n_trees = 20 if smoke else 200
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        start = time.perf_counter()
        try:
            kept = list(range(X.shape[1])) if arm == "all" else _select(arm, X_tr, y_tr, task, seed, smoke, n_trees)
            if not kept:
                raise _Empty("selector kept no features")
        except Exception as exc:  # a failed arm is data, not a crash
            secs = time.perf_counter() - start
            msg = "EmptySelection: " + str(exc) if isinstance(exc, _Empty) else f"{type(exc).__name__}: {exc}"
            dummy = _dummy_score(task, scorer, X_tr, y_tr, X_te, y_te) if isinstance(exc, _Empty) else None
            rows += [_row(arm, m, seed, None, 0, secs, msg, dummy) for m in MODELS]
            continue
        secs = time.perf_counter() - start
        for model in MODELS:
            try:
                est = _downstream(model, task, seed, n_trees).fit(X_tr[:, kept], y_tr)
                score = float(scorer(est, X_te[:, kept], y_te))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score {score}")
                rows.append(_row(arm, model, seed, score, len(kept), secs, None))
            except Exception as exc:
                rows.append(_row(arm, model, seed, None, len(kept), secs, f"{type(exc).__name__}: {exc}"))
    return rows


def load(data_id: int, data_home: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=data_home)
    return bunch.data.to_numpy(dtype=float), bunch.target.to_numpy(), [str(c) for c in bunch.data.columns]


def _effective(r: Dict[str, Any]) -> Optional[float]:
    """Score used for deltas: the test score, or the dummy score of an EmptySelection row."""
    return r["test_score"] if r["error"] is None else r.get("dummy_score")


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Paired deltas (arm - all) per dataset x model x arm, within/better/worse counts, and failures.

    Rows with error EmptySelection are scored with their dummy_score and share kept 0; other
    errored pairs are excluded from deltas and reported under failures / excluded.
    """
    scored = [r for r in rows if _effective(r) is not None]
    base = {(r["dataset"], r["seed"], r["model"]): r["test_score"]
            for r in scored if r["arm"] == "all" and r["error"] is None}
    n_features = {(r["dataset"], r["seed"]): r["n_kept"] for r in scored if r["arm"] == "all" and r["error"] is None}
    deltas: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)
    shares: Dict[Tuple[Any, str, str], List[float]] = defaultdict(list)
    for r in scored:
        key = (r["dataset"], r["seed"], r["model"])
        if r["arm"] == "all" or key not in base:
            continue
        group = (r["dataset"], r["model"], r["arm"])
        deltas[group].append(_effective(r) - base[key])
        shares[group].append(r["n_kept"] / n_features[(r["dataset"], r["seed"])])
    failed: Dict[Tuple[Any, str, str], List[str]] = defaultdict(list)
    for r in rows:
        if r["error"] is not None:
            failed[(r["dataset"], r["model"], r["arm"])].append(r["error"])
    per_dataset = [
        {"dataset": d, "model": m, "arm": a, "mean_delta": float(np.mean(v)), "min_delta": float(min(v)),
         "max_delta": float(max(v)), "mean_share_kept": float(np.mean(shares[(d, m, a)])), "n_seeds": len(v),
         "n_failed": len(failed.get((d, m, a), [])), "first_error": (failed.get((d, m, a)) or [None])[0]}
        for (d, m, a), v in deltas.items()
    ]
    counts = {a: {"within": 0, "better": 0, "worse": 0} for a in ARMS if a != "all"}
    for e in per_dataset:
        d = e["mean_delta"]
        counts[e["arm"]]["within" if abs(d) <= WITHIN + EPS else "better" if d > 0 else "worse"] += 1
    pairs = {(r["dataset"], r["model"]) for r in rows}
    have = {(e["dataset"], e["model"], e["arm"]) for e in per_dataset}
    excluded = {a: sum((d, m, a) not in have for d, m in pairs) for a in counts}
    return {
        "per_dataset": per_dataset, "counts": counts, "excluded": excluded,
        "failures": [{"dataset": d, "model": m, "arm": a, "n_failed": len(v), "first_error": v[0]}
                     for (d, m, a), v in failed.items()],
        "failed_datasets": [{"dataset": d, "n_failed": len(v), "first_error": v[0]}
                            for d, v in _all_failures(rows).items()],
        "note": "EmptySelection rows keep test_score null and are scored with dummy_score in the deltas",
    }


def _all_failures(rows: List[Dict[str, Any]]) -> Dict[Any, List[str]]:
    out: Dict[Any, List[str]] = defaultdict(list)
    for r in rows:
        if r["arm"] == "all" and r["error"] is not None:
            out[r["dataset"]].append(r["error"])
    return out


def render(results: Dict[str, Any]) -> str:
    summary = results["summary"]
    lines = ["| dataset | model | arm | mean delta [min, max] | share kept | seeds | failed |",
             "|---|---|---|---|---:|---:|---:|"]
    for e in summary["per_dataset"]:
        lines.append(f"| {e['dataset']} | {e['model']} | {e['arm']} | {e['mean_delta']:+.4f} "
                     f"[{e['min_delta']:+.4f}, {e['max_delta']:+.4f}] | {e['mean_share_kept']:.2f} | "
                     f"{e['n_seeds']} | {e['n_failed']} |")
    lines += ["", "| arm | within 0.01 | better | worse | pairs excluded |", "|---|---:|---:|---:|---:|"]
    for arm, c in summary["counts"].items():
        lines.append(f"| {arm} | {c['within']} | {c['better']} | {c['worse']} | {summary['excluded'][arm]} |")
    lines += ["", "Failed datasets (`all` arm errored):"]
    lines += [f"- {f['dataset']}: {f['n_failed']} failed rows; {f['first_error'][:150]}"
              for f in summary["failed_datasets"]] or ["- none"]
    lines += ["", "| failures: dataset | model | arm | n_failed | first error |", "|---|---|---|---:|---|"]
    lines += [f"| {f['dataset']} | {f['model']} | {f['arm']} | {f['n_failed']} | {f['first_error'][:100]} |"
              for f in summary["failures"]]
    shapes = results.get("shapes", {})
    lines += ["", "Shapes (rows x features; shape_ok vs PREREG):"]
    lines += [f"- {n}: {v['rows']} x {v['features']} ok={v['shape_ok']}" for n, v in shapes.items()]
    return "\n".join(lines)


def _versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _featranker_info(commit: str) -> Dict[str, Any]:
    """The imported tree's own identity (not importlib.metadata): path, source hash, pyproject version."""
    import featranker

    pkg = Path(featranker.__file__).resolve().parent
    digest = hashlib.sha256()
    for f in sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and (p.suffix in (".py", ".yaml", ".yml", ".json", ".typed") or p.name == "py.typed")):
        digest.update(f.relative_to(pkg).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    pyproject = pkg.parent / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M) if pyproject.exists() else None
    return {"file": featranker.__file__, "version": match.group(1) if match else None,
            "source_sha256": digest.hexdigest(), "commit": commit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--datasets", default=",".join(str(d["data_id"]) for d in DATASETS),
                        help="comma-separated OpenML data ids")
    parser.add_argument("--data-home", default=None, help="OpenML cache directory")
    parser.add_argument("--out", type=Path, default=HERE / f"results_{datetime.date.today():%Y%m%d}.json")
    parser.add_argument("--commit", default="482b62d", help="FeatRanker commit, stored as given")
    parser.add_argument("--resume", action="store_true", help="skip dataset x seed already in --out")
    parser.add_argument("--render", type=Path, help="print the tables of an existing result file and exit")
    args = parser.parse_args()

    if args.render:
        print(render(json.loads(args.render.read_text())))
        return
    seeds = [int(s) for s in args.seeds.split(",")]
    by_id = {d["data_id"]: d for d in DATASETS}
    chosen = [int(i) for i in args.datasets.split(",")]
    unknown = [i for i in chosen if i not in by_id]
    if unknown:
        parser.error(f"unknown data ids: {unknown}; choose from {list(by_id)}")

    started = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    prior_wall = 0.0
    results: Dict[str, Any] = {
        "rows": [], "shapes": {}, "versions": _versions(), "featranker": _featranker_info(args.commit),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "started": started,
    }
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        for key, now in (("featranker source", (prev["featranker"]["source_sha256"], results["featranker"]["source_sha256"])),
                         ("prereg sha256", (prev["prereg_sha256"], results["prereg_sha256"])),
                         ("versions", (prev["versions"], results["versions"]))):
            if now[0] != now[1]:
                parser.error(f"--resume refused: {key} differs from the stored run ({args.out})")
        results["rows"], results["started"] = prev["rows"], prev.get("started", started)
        results["shapes"], prior_wall = prev.get("shapes", {}), prev.get("wall_seconds", 0.0)
    done = {(r["dataset"], r["seed"]) for r in results["rows"]}

    def save() -> None:
        results["summary"] = summarize_results(results["rows"])
        results["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
        results["wall_seconds"] = prior_wall + time.perf_counter() - t0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, allow_nan=False))

    for data_id in chosen:
        spec = by_id[data_id]
        todo = [s for s in seeds if (spec["name"], s) not in done]
        if not todo:
            continue
        try:
            X, y, names = load(data_id, args.data_home)
        except Exception as exc:  # reported as failed, never replaced
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[!] {spec['name']} failed to load: {msg[:120]}", file=sys.stderr, flush=True)
            for s in todo:
                results["rows"] += [{**_row(a, m, s, None, 0, 0.0, f"load: {msg}"), "dataset": spec["name"]}
                                    for a in ARMS for m in MODELS]
                save()
            continue
        shape = {"rows": int(X.shape[0]), "features": int(X.shape[1]), "expected_rows": spec["rows"],
                 "expected_features": spec["features"]}
        shape["shape_ok"] = (shape["rows"], shape["features"]) == (spec["rows"], spec["features"])
        results["shapes"][spec["name"]] = shape
        if not shape["shape_ok"]:
            print(f"[!] {spec['name']} shape {X.shape} differs from PREREG", file=sys.stderr, flush=True)
        for s in todo:
            rows = run_dataset(X, y, names, task=spec["task"], seed=s)
            results["rows"] += [{**r, "dataset": spec["name"]} for r in rows]
            save()
            failed = sum(r["error"] is not None for r in rows)
            print(f"[*] {spec['name']} seed {s}: {len(rows)} rows, {failed} errors", file=sys.stderr, flush=True)
    save()
    print(f"[*] results: {args.out}  wall {results['wall_seconds'] / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
