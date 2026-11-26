"""Acceptance check for codex-review-12 (lead-written; builder must not edit)."""
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from featranker import FeatureRanker
from test_codex_review_10 import bench  # noqa: F401

ROOT = Path(__file__).parents[1]
X = np.random.default_rng(0).random((150, 4))
Y = X[:, 0] + 2 * X[:, 1]


def test_xgboost_with_a_random_state_refits_with_the_same_seed():
    xgboost = pytest.importorskip("xgboost")
    model = xgboost.XGBRegressor(n_estimators=15, subsample=0.7, n_jobs=1,
                                 random_state=np.random.RandomState(23))
    ranker = FeatureRanker("reg", models={"m": model}, verbose=False)
    ranker.fit(X[:100], Y[:100], feature_names=list("abcd"))
    report = ranker.rank_features(X[100:], Y[100:], scoring="r2", n_repeats=1)
    ablated = ranker.ablate(X[:100], Y[:100], X[100:], Y[100:], report, ks=[4])
    full = ablated["ablation"]["curve"][-1]["scores"]["m"]
    assert full == pytest.approx(report["models"]["m"]["evaluation_score"], abs=1e-12)


def test_seeded_user_models_are_unchanged():
    from sklearn.ensemble import RandomForestRegressor

    def score():
        ranker = FeatureRanker("reg", models={"m": RandomForestRegressor(10, random_state=0)}, verbose=False)
        ranker.fit(X[:100], Y[:100], feature_names=list("abcd"))
        return ranker.rank_features(X[100:], Y[100:], scoring="r2", n_repeats=1)["models"]["m"]["evaluation_score"]

    assert score() == score()


def test_one_missing_value_keeps_the_shap_direction():
    pytest.importorskip("shap")
    rng = np.random.default_rng(0)
    signal = np.linspace(-2, 2, 100)
    data = np.c_[signal, rng.normal(size=100)]
    target = 4 * signal
    order = rng.permutation(100)
    data, target = data[order], target[order]
    holdout = data[80:].copy()
    holdout[0, 0] = np.nan
    ranker = FeatureRanker("reg", models={"m": make_pipeline(SimpleImputer(), Ridge())}, missing="impute",
                           evidence=True, verbose=False)
    ranker.fit(data[:80], target[:80], feature_names=["signal", "noise"])
    report = ranker.rank_features(holdout, target[80:], scoring="r2", n_repeats=1)
    direction = report["models"]["m"]["shap"]["groups"]["signal"]["direction"]
    assert direction == pytest.approx(1.0)


@pytest.mark.skipif(sys.platform == "win32", reason="needs resource.RLIMIT_FSIZE")
def test_failed_cli_write_keeps_the_previous_report(tmp_path):
    target = tmp_path / "current.json"
    target.write_text(json.dumps({"old": "x" * 5000}), encoding="utf-8")
    before = target.read_bytes()
    script = tmp_path / "write.py"
    script.write_text(textwrap.dedent(f"""
        import resource, sys
        from pathlib import Path
        from featranker import cli
        resource.setrlimit(resource.RLIMIT_FSIZE, (4096, resource.RLIM_INFINITY))
        try:
            cli._write_report({{"new": "y" * 20000}}, Path({str(target)!r}))
        except OSError:
            sys.exit(3)
        sys.exit(0)
    """))
    done = subprocess.run([sys.executable, str(script)], env={**os.environ, "PYTHONPATH": str(ROOT)},
                          capture_output=True, text=True)
    assert done.returncode == 3, done.stderr[-500:]
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["current.json", "write.py"]


def test_cli_write_replaces_and_keeps_a_symlink(tmp_path):
    from featranker import cli

    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks are not available")
    cli._write_report({"a": 1}, link)
    assert link.is_symlink() and json.loads(real.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link.json", "real.json"]


def test_render_survives_a_narrow_console(bench, monkeypatch):  # noqa: F811
    run, out, main, _ = bench
    main("--seeds", "0")
    results = json.loads(out.read_text())
    results["summary"]["failed_datasets"] = [
        {"dataset": "x", "n_failed": 1, "first_error": "FileNotFoundError: C:/Users/東京/OpenML data/1510"}]
    out.write_text(json.dumps(results), encoding="utf-8")
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "argv", ["run.py", "--render", str(out)])
    run.main()
    console.flush()
    text = buffer.getvalue().decode("cp1252")
    assert "| dataset |" in text and "OpenML data/1510" in text


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs POSIX folder permissions and a non-root user")
def test_writable_report_in_a_read_only_folder_is_still_written(tmp_path):
    from featranker import cli

    folder = tmp_path / "locked"
    folder.mkdir()
    target = folder / "r.json"
    target.write_text("{}", encoding="utf-8")
    folder.chmod(0o555)
    try:
        cli._write_report({"a": 1}, target)
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        assert sorted(p.name for p in folder.iterdir()) == ["r.json"]
    finally:
        folder.chmod(0o755)
