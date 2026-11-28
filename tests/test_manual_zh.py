"""Acceptance check for the zh-CN docs and README visuals (lead-written; builders must not edit)."""
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from featranker import ConsensusSelector, FeatureRanker, Report, auto_groups, summarize

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
ZH = DOCS / "zh-CN"
MANUAL = [
    "README.md", "getting-started.md", "concepts.md", "guide/models-and-scoring.md", "guide/custom-models.md",
    "guide/data.md", "guide/validation.md", "guide/noise-and-evidence.md", "guide/selection.md",
    "guide/performance.md", "cli.md", "reference.md", "troubleshooting.md",
]
# (English file, Chinese file)
PAIRS = [(ROOT / "README.md", ROOT / "README.zh-CN.md")] + [(DOCS / p, ZH / p) for p in MANUAL]
RAW = "https://raw.githubusercontent.com/CountZero-Error/Featranker/main/docs/assets/"
BLOB = "https://github.com/CountZero-Error/Featranker/blob/main/"
CJK = re.compile(r"[一-鿿]")


def read(path):
    return path.read_text(encoding="utf-8")


def fenced(body, lang):
    return re.findall(rf"^```{lang}\n(.*?)^```", body, flags=re.S | re.M)


def outputs(body):
    """Each ```text block that directly follows a ```python block (the shown output)."""
    return re.findall(r"^```python\n.*?^```\s*\n```text\n(.*?)^```", body, flags=re.S | re.M)


def prose(body):
    body = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
    body = re.sub(r"`[^`]*`|\([^)]*\)|<[^>]+>", "", body)
    return body


def ids(pair):
    return pair[1].relative_to(ROOT).as_posix()


@pytest.mark.parametrize("pair", PAIRS, ids=ids)
def test_page_exists_with_one_title(pair):
    body = re.sub(r"^```.*?^```", "", read(pair[1]), flags=re.S | re.M)
    if pair[1].name != "README.zh-CN.md":
        assert len([line for line in body.splitlines() if line.startswith("# ")]) == 1


@pytest.mark.parametrize("pair", PAIRS, ids=ids)
def test_code_and_output_are_identical(pair):
    english, chinese = read(pair[0]), read(pair[1])
    for lang in ("python", "bash", "json"):
        assert fenced(chinese, lang) == fenced(english, lang), lang
    assert outputs(chinese) == outputs(english)


@pytest.mark.parametrize("pair", PAIRS, ids=ids)
def test_prose_is_translated(pair):
    text = prose(read(pair[1]))
    letters = re.findall(r"[A-Za-z一-鿿]", text)
    assert letters and len(CJK.findall(text)) / len(letters) > 0.25


@pytest.mark.parametrize("pair", PAIRS, ids=ids)
def test_language_switch_both_ways(pair):
    import os

    english, chinese = pair
    en_body, zh_body = read(english), read(chinese)
    if english.parent == ROOT:
        assert BLOB + "README.zh-CN.md" in en_body and "简体中文" in en_body
        assert BLOB + "README.md" in zh_body and "English" in zh_body
        return
    forward = Path(os.path.relpath(chinese, english.parent)).as_posix()
    back = Path(os.path.relpath(english, chinese.parent)).as_posix()
    assert f"]({forward})" in en_body and "简体中文" in en_body
    assert f"]({back})" in zh_body and "English" in zh_body


def test_reference_names_every_parameter():
    import inspect

    callables = {
        "FeatureRanker": FeatureRanker.__init__, "FeatureRanker.fit": FeatureRanker.fit,
        "FeatureRanker.rank_features": FeatureRanker.rank_features,
        "FeatureRanker.rank_resampled": FeatureRanker.rank_resampled, "FeatureRanker.ablate": FeatureRanker.ablate,
        "ConsensusSelector": ConsensusSelector.__init__, "ConsensusSelector.fit": ConsensusSelector.fit,
        "ConsensusSelector.transform": ConsensusSelector.transform, "Report.summary": Report.summary,
        "Report.to_frame": Report.to_frame, "Report.plot": Report.plot, "auto_groups": auto_groups,
        "summarize": summarize,
    }
    body = read(ZH / "reference.md")
    sections = dict(re.findall(r"^### `([^`]+)`\n(.*?)(?=^### |\Z)", body, flags=re.S | re.M))
    for name, function in callables.items():
        assert name in sections, name
        params = [n for n, p in inspect.signature(function).parameters.items()
                  if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        assert not [p for p in params if f"`{p}`" not in sections[name]], name


def test_cli_page_names_every_flag():
    from featranker.cli import _build_parser

    flags = {o for a in _build_parser()._actions for o in a.option_strings if o.startswith("--")} - {"--help"}
    body = read(ZH / "cli.md")
    assert not sorted(f for f in flags if f"`{f}" not in body)


def slug(heading):
    heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def anchors(path):
    body = re.sub(r"^```.*?^```", "", read(path), flags=re.S | re.M)
    found, seen = set(re.findall(r'<a id="([^"]+)"', body)), {}
    for heading in re.findall(r"^#+ (.+)$", body, flags=re.M):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if not count else f"{base}-{count}")
    return found


def test_links_resolve():
    broken = []
    for path in [p for pair in PAIRS for p in pair]:
        body = re.sub(r"^```.*?^```", "", read(path), flags=re.S | re.M)
        for target in re.findall(r"\]\(([^)\s]+)\)", body) + re.findall(r'(?:href|src)="([^"]+)"', body):
            if target.startswith(BLOB):
                base, target = ROOT, target[len(BLOB):]
            elif target.startswith(RAW):
                base, target = DOCS / "assets", target[len(RAW):]
            elif target.startswith("https://github.com/CountZero-Error/Featranker#"):
                base, target = ROOT, "README.md#" + target.split("#", 1)[1]
            elif re.match(r"[a-z]+:", target):
                continue
            else:
                base = path.parent
            file, _, anchor = target.partition("#")
            resolved = (base / file).resolve() if file else path
            if not resolved.exists():
                broken.append((path.relative_to(ROOT).as_posix(), target))
            elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
                broken.append((path.relative_to(ROOT).as_posix(), target))
    assert not broken


def test_banner_is_safe_svg():
    banner = DOCS / "assets" / "featranker-banner.svg"
    root = ET.parse(banner).getroot()
    assert root.tag.endswith("svg")
    text = read(banner)
    assert "<script" not in text and "href=\"http" not in text and "@import" not in text
    assert "FEATRANKER" in text


def test_chart_is_generated_png():
    chart = DOCS / "assets" / "consensus.png"
    head = chart.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", head[16:24])
    assert 400 <= width <= 1600 and height >= 200
    script = read(DOCS / "assets" / "make_consensus.py")
    assert ".plot(" in script and "savefig" in script and "load_diabetes" in script


@pytest.mark.parametrize("readme", ["README.md", "README.zh-CN.md"])
def test_readmes_show_both_images(readme):
    body = read(ROOT / readme)
    assert RAW + "featranker-banner.svg" in body and RAW + "consensus.png" in body
