import importlib.metadata
from pathlib import Path

import featranker

ROOT = Path(__file__).parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_public_names():
    assert sorted(featranker.__all__) == sorted(
        [
            "ConsensusSelector", "FeatureRanker", "Report", "auto_groups", "explain",
            "summarize", "to_prompt",
        ]
    )
    assert all(hasattr(featranker, name) for name in featranker.__all__)


def test_version_matches_metadata():
    try:
        expected = importlib.metadata.version("featranker")
    except importlib.metadata.PackageNotFoundError:
        expected = None
    assert featranker.__version__ == expected


def test_py_typed_is_shipped():
    assert (ROOT / "featranker" / "py.typed").is_file()
    assert '"py.typed"' in PYPROJECT.split("[tool.setuptools.package-data]", 1)[1]


def test_extras_match_the_registry():
    extras = PYPROJECT.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]

    assert "lightgbm" not in extras
    assert 'all = ["xgboost", "catboost"]' in extras
    assert 'plot = ["matplotlib"]' in extras
    assert 'shap = ["shap"]' in extras
    for name in ("xgboost", "catboost", "test"):
        assert f"\n{name} = [" in extras


def test_test_extra_installs_shap_but_all_does_not():
    extras = PYPROJECT.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    test_line = next(line for line in extras.splitlines() if line.startswith("test = "))

    assert '"shap"' in test_line
    assert 'all = ["xgboost", "catboost"]' in extras
