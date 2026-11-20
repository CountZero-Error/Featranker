"""Acceptance check for ci-matrix (lead-written; builder must not edit)."""
import re
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"
SHA = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _text(job):
    return yaml.safe_dump(job)


def _runs(job):
    return " ".join(step.get("run", "") for step in job["steps"])


def test_macos_job():
    hits = [job for job in _jobs().values() if "macos-" in _text(job["runs-on"])]
    assert hits, "no macOS job"
    job = hits[0]
    assert "3.13" in _text(job)
    assert "python -m pytest -q" in _runs(job)
    assert "mypy" in _runs(job)


def test_oldest_numpy_job():
    hits = [job for job in _jobs().values() if "numpy>=1.24,<1.25" in _text(job)]
    assert hits, "no oldest-NumPy job"
    job = hits[0]
    text = _text(job)
    assert "3.10" in text and "scikit-learn>=1.5,<1.6" in text
    assert "python -m pytest -q" in _runs(job)


def test_actions_pinned_and_no_persisted_token():
    for name, job in _jobs().items():
        for step in job["steps"]:
            if "uses" in step:
                assert SHA.match(step["uses"].split()[0]), (name, step["uses"])
            if "actions/checkout" in step.get("uses", ""):
                assert step.get("with", {}).get("persist-credentials") is False, name
