"""Acceptance check for the release-candidate fixes (lead-written; builder must not edit)."""
import ctypes
import errno
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker
from featranker.cli import _write_report

ROOT = Path(__file__).parents[1]
NAMES = ["a", "b"]


def _shadow_failures(report, name):
    return [f for f in report["failures"] if f["model"] == name and f["stage"] == "shadows"]


def _rank(model):
    X = np.random.default_rng(8).normal(size=(160, 2))
    y = X.sum(axis=1)
    ranker = FeatureRanker("reg", models={"m": model}, shadows=True, shadow_seed=1, verbose=False)
    ranker.fit(X[:120], y[:120], feature_names=NAMES)
    return ranker, ranker.rank_features(X[120:], y[120:], scoring="r2", n_repeats=3, random_state=0)


def test_bagging_with_selector_chosen_by_inner_search_is_unverified():
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"standardscaler": [SelectKBest(f_regression, k=2)]},
        cv=2,
    )
    ranker, report = _rank(BaggingRegressor(search, n_estimators=2, random_state=0))
    members = ranker.trained_models["m"].estimators_
    assert all(s.best_estimator_[0].get_support(indices=True).tolist() == [0, 1] for s in members)
    assert _shadow_failures(report, "m"), report["failures"]
    assert report["models"]["m"].get("shadow_max") is None


def test_plain_bagging_still_gives_shadow_evidence():
    _, report = _rank(BaggingRegressor(Ridge(), n_estimators=2, random_state=0))
    assert not _shadow_failures(report, "m")
    assert report["models"]["m"].get("shadow_max") is not None


def _fitted_ridge():
    X = np.random.default_rng(8).normal(size=(80, 2))
    y = X.sum(axis=1)
    ranker = FeatureRanker("reg", models={"m": Ridge()}, verbose=False)
    ranker.fit(X[:60], y[:60], feature_names=NAMES)
    return ranker, X, y


@pytest.mark.parametrize("bad", [0, -1, 1.5, "all", True])
def test_refit_refused_for_max_samples_leaves_ranker_unfitted(bad):
    ranker, X, y = _fitted_ridge()
    with pytest.raises((ValueError, TypeError)):
        ranker.fit(X[:60], -y[:60], feature_names=NAMES, max_samples=bad)
    assert ranker.is_fitted_ is False
    with pytest.raises(RuntimeError):
        ranker.rank_features(X[60:], y[60:], scoring="r2", n_repeats=1, random_state=0)


def test_refit_refused_for_seed_still_leaves_ranker_fitted():
    ranker, X, y = _fitted_ridge()
    with pytest.raises(ValueError):
        ranker.fit(X[:60], -y[:60], feature_names=NAMES, random_state=-1)
    assert ranker.is_fitted_ is True


def test_cli_type_checks_on_this_platform():
    if shutil.which("mypy") is None and subprocess.run(
        [sys.executable, "-m", "mypy", "--version"], capture_output=True
    ).returncode:
        pytest.skip("mypy is not installed")
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "featranker/cli.py", "--disable-error-code", "import-untyped"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("umask, expected", [(0o022, 0o644), (0o077, 0o600), (0o002, 0o664)])
def test_new_report_follows_the_umask(tmp_path, umask, expected):
    target = tmp_path / "report.json"
    old = os.umask(umask)
    try:
        _write_report({"new": True}, target)
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


class _NoAclLibc:
    """A libc whose ACL calls fail as on FAT, exFAT or some network volumes."""

    def __init__(self, *args, **kwargs):
        self.acl_init = self._init
        self.acl_set_file = self._unsupported
        self.acl_free = lambda acl: 0
        self.fcopyfile = self._unsupported

    @staticmethod
    def _init(_):
        return 1

    @staticmethod
    def _unsupported(*_):
        ctypes.set_errno(errno.ENOTSUP)
        return -1


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL calls")
@pytest.mark.parametrize("existing", [False, True])
def test_volume_without_acls_still_gets_the_report(tmp_path, monkeypatch, existing):
    target = tmp_path / "report.json"
    if existing:
        target.write_text("old", encoding="utf-8")
        target.chmod(0o640)
    monkeypatch.setattr(ctypes, "CDLL", _NoAclLibc)
    _write_report({"new": True}, target)
    assert target.read_text(encoding="utf-8").startswith("{")
    if existing:
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux xattrs")
def test_volume_without_xattrs_still_gets_the_report(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    def unsupported(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))

    for name in ("listxattr", "getxattr", "setxattr", "removexattr"):
        monkeypatch.setattr(os, name, unsupported)
    _write_report({"new": True}, target)
    assert target.read_text(encoding="utf-8").startswith("{")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
