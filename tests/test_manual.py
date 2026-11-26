"""Acceptance check for the 2.0 manual (lead-written; builder must not edit)."""
import importlib.util
import inspect
import re
import time
from pathlib import Path

import pytest

import featranker
from featranker import ConsensusSelector, FeatureRanker, Report, auto_groups, summarize

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
PAGES = [
    "README.md", "getting-started.md", "concepts.md", "guide/models-and-scoring.md", "guide/custom-models.md",
    "guide/data.md", "guide/validation.md", "guide/noise-and-evidence.md", "guide/selection.md",
    "guide/performance.md", "cli.md", "reference.md", "troubleshooting.md", "report.md", "models.md", "api.md",
    "usage.md",
]
RUNNABLE = [p for p in PAGES if p not in {"usage.md"}] + ["../README.md"]
OPTIONAL = ("xgboost", "lightgbm", "catboost", "shap", "matplotlib", "pandas", "jsonschema")
CALLABLES = {
    "FeatureRanker": FeatureRanker.__init__,
    "FeatureRanker.fit": FeatureRanker.fit,
    "FeatureRanker.rank_features": FeatureRanker.rank_features,
    "FeatureRanker.rank_resampled": FeatureRanker.rank_resampled,
    "FeatureRanker.ablate": FeatureRanker.ablate,
    "ConsensusSelector": ConsensusSelector.__init__,
    "ConsensusSelector.fit": ConsensusSelector.fit,
    "ConsensusSelector.transform": ConsensusSelector.transform,
    "Report.summary": Report.summary,
    "Report.to_frame": Report.to_frame,
    "Report.plot": Report.plot,
    "auto_groups": auto_groups,
    "summarize": summarize,
}
DOCSTRING_OWNER = {"FeatureRanker": FeatureRanker, "ConsensusSelector": ConsensusSelector}


def params(function):
    return [n for n, p in inspect.signature(function).parameters.items()
            if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def text(page):
    return (DOCS / page).read_text(encoding="utf-8")


def blocks(page):
    return re.findall(r"^```python\n(.*?)^```", text(page), flags=re.S | re.M)


def test_pages_exist():
    missing = [p for p in PAGES if not (DOCS / p).is_file()]
    assert not missing


def test_usage_is_a_short_landing_page():
    body = text("usage.md")
    assert len(body.splitlines()) <= 120
    for page in ("getting-started.md", "reference.md", "guide/"):
        assert page in body


@pytest.mark.parametrize("page", [p for p in PAGES if p != "README.md"])
def test_one_plain_title(page):
    lines = re.sub(r"^```.*?^```", "", text(page), flags=re.S | re.M).splitlines()
    titles = [line for line in lines if line.startswith("# ")]
    assert len(titles) == 1 and "▍" not in titles[0]
    assert not [line for line in lines if line.startswith("#") and "▍" in line]


@pytest.mark.parametrize("page", RUNNABLE)
def test_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("sklearn")
    namespace = {"__name__": "__main__"}
    started = time.perf_counter()
    for code in blocks(page):
        if code.startswith("# not run"):
            continue
        needed = [m for m in OPTIONAL if re.search(rf"^\s*(import|from) {m}\b", code, flags=re.M)]
        if any(importlib.util.find_spec(m) is None for m in needed):
            continue
        exec(compile(code, f"{page}", "exec"), namespace)
    assert time.perf_counter() - started < 120, f"{page} examples are too slow"


def test_reference_names_every_parameter():
    body = text("reference.md")
    sections = dict(re.findall(r"^### `([^`]+)`\n(.*?)(?=^### |\Z)", body, flags=re.S | re.M))
    for name, function in CALLABLES.items():
        assert name in sections, name
        absent = [p for p in params(function) if f"`{p}`" not in sections[name]]
        assert not absent, (name, absent)


def test_reference_descriptions_are_short():
    rows = [line for line in text("reference.md").splitlines() if line.startswith("| `")]
    long = [row[:80] for row in rows if len(row.split("|")[-2].strip()) > 300]
    assert rows and not long
    assert "Input rules" in text("reference.md")


@pytest.mark.parametrize("name", list(CALLABLES))
def test_docstrings_describe_every_parameter(name):
    owner = DOCSTRING_OWNER.get(name, CALLABLES[name])
    doc = inspect.getdoc(owner) or ""
    assert "Parameters" in doc or not params(CALLABLES[name]), name
    absent = [p for p in params(CALLABLES[name]) if not re.search(rf"^{p}\s*:", doc, flags=re.M)]
    assert not absent, (name, absent)


def test_public_names_have_docstrings():
    for name in featranker.__all__:
        assert len(inspect.getdoc(getattr(featranker, name)) or "") > 80, name
    assert "Examples" in (inspect.getdoc(FeatureRanker) or "")


def test_cli_page_names_every_flag():
    from featranker.cli import _build_parser

    flags = {o for a in _build_parser()._actions for o in a.option_strings if o.startswith("--")} - {"--help"}
    body = text("cli.md")
    absent = sorted(f for f in flags if f"`{f}" not in body)
    assert not absent


def slug(heading):
    heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def anchors(path):
    body = path.read_text(encoding="utf-8")
    found, seen = set(re.findall(r'<a id="([^"]+)"', body)), {}
    for heading in re.findall(r"^#+ (.+)$", body, flags=re.M):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if not count else f"{base}-{count}")
    return found


def links(path):
    body = re.sub(r"^```.*?^```", "", path.read_text(encoding="utf-8"), flags=re.S | re.M)
    for target in re.findall(r"\]\(([^)\s]+)\)", body) + re.findall(r'href="([^"]+)"', body):
        prefix = "https://github.com/CountZero-Error/Featranker/blob/main/"
        if target.startswith(prefix):
            yield ROOT, target[len(prefix):]
        elif not re.match(r"[a-z]+:", target):
            yield path.parent, target


def test_links_resolve():
    broken = []
    for path in [ROOT / "README.md"] + [DOCS / p for p in PAGES]:
        for base, target in links(path):
            file, _, anchor = target.partition("#")
            resolved = (base / file).resolve() if file else path
            if not resolved.exists():
                broken.append((path.name, target))
            elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                broken.append((path.name, target))
    assert not broken
