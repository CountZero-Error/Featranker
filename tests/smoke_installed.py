"""Smoke test for an INSTALLED featranker (wheel), not the source tree.

Run it from outside the checkout with the interpreter of the environment the
wheel was installed in:

    python tests/smoke_installed.py --expect-version X.Y.Z [--extras]

--expect-version is the version in the checkout's pyproject.toml; the installed
metadata and featranker.__version__ must both equal it. --extras also requires
xgboost, catboost, SHAP evidence and report.plot(); without it the environment
must be core-only and the SHAP and plot paths must degrade as documented. Plain
asserts, no pytest: any failure exits non-zero. pytest does not collect it
(the name does not match test_*.py).
"""

import argparse
import importlib.metadata
import importlib.resources
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

SUBPROCESS_TIMEOUT_SECONDS = 600
OPTIONAL_PACKAGES = ("xgboost", "catboost", "shap", "matplotlib")


def check_environment(extras: bool) -> None:
    present = [name for name in OPTIONAL_PACKAGES if importlib.util.find_spec(name) is not None]
    if extras:
        missing = sorted(set(OPTIONAL_PACKAGES) - set(present))
        assert not missing, f"--extras but the environment lacks {missing}"
    else:
        assert not present, f"core-only run, but the environment has {present}"


def check_installed_copy(expected_version: str) -> None:
    import featranker

    here = Path(featranker.__file__).resolve().parent
    pyproject = here.parent / "pyproject.toml"  # flat layout: the checkout root
    if pyproject.is_file() and re.search(
        r'^name\s*=\s*"featranker"', pyproject.read_text(encoding="utf-8"), re.M
    ):
        sys.exit(f"FAIL: featranker imported from the source tree {here.parent}, not an installed copy")
    installed = importlib.metadata.version("featranker")
    assert installed == expected_version, f"installed metadata {installed}, expected {expected_version}"
    assert featranker.__version__ == expected_version, (featranker.__version__, expected_version)
    files = importlib.resources.files("featranker")
    for name in ("importance_config.yaml", "py.typed"):
        assert (files / name).is_file(), f"package data missing from the wheel: {name}"
    print(f"featranker {installed} at {here}")


def make_data(task: str, n_rows: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 4))
    signal = 2.0 * X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n_rows)
    y = signal if task == "reg" else (signal > 0).astype(int)
    return X, y, ["strong", "weak", "noise_1", "noise_2"]


def check_api(extras: bool) -> None:
    from featranker import ConsensusSelector, FeatureRanker, summarize

    X, y, names = make_data("reg")
    train, val = slice(0, 80), slice(80, None)
    ranker = FeatureRanker(
        task="reg", group="tree" if extras else "fast", evidence=True, verbose=False
    )
    ranker.fit(X[train], y[train], feature_names=names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # core-only: the documented "shap is not installed" warning
        report = ranker.rank_features(X[val], y[val], scoring="r2", n_repeats=2, random_state=0)
    assert report["consensus"], f"empty consensus; failures: {report['failures']}"
    assert report["consensus"][0]["feature_group"] == "strong", report["consensus"][0]
    assert isinstance(summarize(report), list)
    ablation = ranker.ablate(X[train], y[train], X[val], y[val], report, ks=[1, 2])["ablation"]
    curve = ablation["curve"]
    assert len(curve) == 2, curve
    for row in curve:
        score = row["paired_median_score"]
        assert row["n_models_scored"] > 0 and score is not None and math.isfinite(score), (
            f"ablation row k={row['k']} has no scored model; failures: {ablation['failures']}"
        )

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("select", ConsensusSelector("reg", "r2", k=2, group="fast", cv=3, n_repeats=2, random_state=0)),
        ("model", Ridge()),
    ])
    assert pipeline.fit(X, y).predict(X).shape == (len(y),)
    if extras:
        for model in ("xgboost_regressor", "catboost_regressor"):
            entry = report["models"].get(model)
            assert entry and entry.get("evaluation_score") is not None, (
                f"{model} did not rank: {report['failures']}"
            )
        assert report["evidence"]["shap"]["available"], "SHAP evidence missing"
        assert any("shap" in m for m in report["models"].values()), "no model has SHAP values"
        import matplotlib

        matplotlib.use("Agg")
        assert report.plot() is not None, "report.plot() returned nothing"
    else:
        assert report["evidence"]["shap"] == {"available": False}, report["evidence"]["shap"]
        try:
            report.plot()
        except ImportError as error:
            assert "featranker[plot]" in str(error), error
        else:
            raise AssertionError("report.plot() should raise ImportError without matplotlib")
    print("api ok" + (" (xgboost, catboost, shap, plot)" if extras else " (core-only; SHAP and plot degrade)"))


def console_script() -> str:
    script = Path(sys.executable).parent / "featranker"
    assert script.is_file(), f"featranker console script not found next to {sys.executable}"
    return str(script)


def check_cli() -> None:
    X, y, names = make_data("clf")
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for label, rows in (("train", slice(0, 80)), ("eval", slice(80, None))):
            paths[label] = Path(tmp) / f"{label}.csv"
            data = np.column_stack([X[rows], y[rows]])
            np.savetxt(paths[label], data, delimiter=",", header=",".join([*names, "target"]), comments="")
        out = Path(tmp) / "report.json"
        run = subprocess.run(
            [console_script(), "--task", "clf", "--train-data", str(paths["train"]),
             "--eval-data", str(paths["eval"]), "--target", "target", "--scoring", "accuracy",
             "--group", "fast", "--n-repeats", "2", "--random-state", "0", "--quiet",
             "--output", str(out)],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        assert run.returncode == 0, f"featranker CLI failed ({run.returncode}):\n{run.stderr}"
        assert json.loads(out.read_text())["consensus"], "CLI report has an empty consensus"
    print("cli ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test an installed featranker (wheel), not the source tree."
    )
    parser.add_argument("--expect-version", required=True, help="version from the checkout's pyproject.toml")
    parser.add_argument("--extras", action="store_true", help="also require xgboost, catboost, SHAP, matplotlib")
    args = parser.parse_args()
    check_installed_copy(args.expect_version)
    check_environment(args.extras)
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)  # catboost writes catboost_info into the cwd
        try:
            check_api(args.extras)
            check_cli()
        finally:
            os.chdir(previous)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
