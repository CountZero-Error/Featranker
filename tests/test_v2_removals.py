"""Acceptance check for remove-interpret / 2.0 (lead-written; builder must not edit)."""
import importlib
import re
from pathlib import Path

import pytest
import yaml

import featranker

ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$")


def test_llm_helpers_are_gone():
    for name in ("to_prompt", "explain"):
        assert not hasattr(featranker, name), name
        with pytest.raises(ImportError):
            exec(f"from featranker import {name}", {})
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("featranker.interpret")
    assert not (ROOT / "featranker" / "interpret.py").exists()


def test_public_names_exact():
    assert set(featranker.__all__) == {
        "ConsensusSelector", "FeatureRanker", "Report", "auto_groups", "summarize",
    }


def test_docs_do_not_mention_helpers_outside_removal_record():
    for path in (ROOT / "README.md", ROOT / "docs" / "usage.md", ROOT / "docs" / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "to_prompt" not in text, path
        assert "explain(" not in text, path
    api = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    lines = [line for line in api.splitlines() if "to_prompt" in line]
    assert lines and all("2.0" in line for line in lines)


def test_windows_job():
    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))["jobs"]
    hits = [job for job in jobs.values() if "windows-" in str(job["runs-on"])]
    assert hits, "no Windows job"
    job = hits[0]
    assert "3.12" in yaml.safe_dump(job)
    assert "python -m pytest -q" in " ".join(step.get("run", "") for step in job["steps"])
    for step in job["steps"]:
        if "uses" in step:
            assert SHA.match(step["uses"].split()[0])
        if "actions/checkout" in step.get("uses", ""):
            assert step["with"]["persist-credentials"] is False


def test_version_is_2_x():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "2\.\d+\.\d+"$', text, flags=re.M)
