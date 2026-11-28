"""Acceptance check for the 2.0.1 polish change (lead-written; builders must not edit)."""
import errno
import json
import os
import stat
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from featranker import ConsensusSelector, cli
from featranker.importance import _check_index, _is_column_aware

posix = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


# CLI --output: a new report gets its mode from the kernel, without reading the umask

@posix
def test_new_report_follows_umask_without_reading_it(tmp_path, monkeypatch):
    old = os.umask(0o027)
    try:
        def no_umask(*_):
            raise AssertionError("the umask must not be read or changed")

        monkeypatch.setattr(cli.os, "umask", no_umask)
        target = tmp_path / "report.json"
        cli._write_report({"a": 1}, target)
    finally:
        monkeypatch.undo()  # cli.os is the os module: restore umask() before using it
        os.umask(old)
    assert json.loads(target.read_text()) == {"a": 1}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


# CLI --output: hard links keep pointing at the report

@posix
def test_hard_linked_report_is_updated_in_place(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    os.link(target, link)
    inode = target.stat().st_ino
    cli._write_report({"b": 2}, target)
    assert target.stat().st_ino == inode
    assert json.loads(link.read_text()) == {"b": 2}


# CLI --output: a writable report owned by someone else is still updated

@posix
def test_writable_report_whose_owner_cannot_be_copied_is_updated(tmp_path, monkeypatch):
    others = [g for g in os.getgroups() if g != os.getgid()]
    if not others:
        pytest.skip("needs a secondary group")
    target = tmp_path / "report.json"
    target.write_text("{}")
    os.chown(target, -1, others[0])  # the staged copy would need chown to match
    target.chmod(0o664)
    link_free_inode = target.stat().st_ino

    def refuse(*_):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(cli.os, "chown", refuse)
    cli._write_report({"c": 3}, target)
    assert json.loads(target.read_text()) == {"c": 3}
    kept = target.stat()
    assert kept.st_gid == others[0] and stat.S_IMODE(kept.st_mode) == 0o664
    assert kept.st_ino == link_free_inode


# CLI --output on Windows: a report held open gives a clear error

@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing rules")
def test_windows_report_held_open_gives_a_clear_error(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{}")
    with open(target, "r", encoding="utf-8"):
        with pytest.raises(PermissionError, match="(?i)open"):
            cli._write_report({"d": 4}, target)
    assert target.read_text() == "{}"


# Shadow evidence: a non-iterable estimators_ is one member, not a crash

class OneMember(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.estimators_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.estimators_.predict(X)


def test_non_iterable_estimators_attribute_is_one_member():
    model = OneMember().fit(np.arange(10.0).reshape(-1, 1), np.arange(10.0))
    assert _is_column_aware(model) is False


# Index check: unhashable labels warn instead of crashing

def test_unhashable_index_labels_warn():
    X = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([(1, [0]), (2, [1])], dtype=object, tupleize_cols=False))
    y = pd.Series([1.0, 2.0], index=pd.Index([(1, [0]), (3, [1])], dtype=object, tupleize_cols=False))
    with pytest.warns(UserWarning, match="index"):
        _check_index(X, y, "y")


# Docs: ConsensusSelector.fit lists the dummy raise

def test_selector_fit_docstring_names_the_dummy_raise():
    raises = ConsensusSelector.fit.__doc__.split("Raises", 1)[1]
    assert "dummy" in raises


def test_selector_uses_the_baseline_name_constant():
    import inspect

    from featranker import selection

    assert '"dummy_baseline"' not in inspect.getsource(selection)
