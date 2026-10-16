## Why

The 1.0 thesis is now implemented: which features matter (0.5), how sure (0.6–0.7), and a leakage-safe way to use the answer (0.8), with a guided reading (0.9). 1.0 is the promise that this API will stay stable. Before making it, the package needs:
- a friendlier result object;
- type information for users' tools;
- releases that don't depend on a hand-held PyPI token;
- a written statement of what is public and how it may change.

## What Changes

- **Report object.** `rank_features()` and `rank_resampled()` return a `featranker.Report`. It is a `dict` subclass, so every existing use keeps working: indexing, `json.dumps`, equality with dicts, and pickling. It adds three methods:
  - `summary()` returns the `summarize()` findings, for single-split reports;
  - `to_frame()` returns a pandas DataFrame of the consensus (single-split) or the stability table (resampled);
  - `plot(ax=None)` returns a matplotlib `Axes`. For single-split reports it plots family ranks per group with the median marked; for resampled reports, each group's place range with its median.

  pandas and matplotlib stay optional, with a clear `ImportError` that names the extra.
- **Typing.** Ship `py.typed`, make `mypy featranker` pass, and run mypy in CI. Expose `featranker.__version__`.
- **Trusted publishing.** A new `.github/workflows/release.yml` builds and checks the package and publishes it to PyPI on a `v*` tag push, via PyPI trusted publishing (OIDC, no stored token). The maintainer configures the trusted publisher on pypi.org once.
- **Public API and stability policy.**
  - The public API is the names in `featranker.__all__`, the CLI flags, and the report keys documented in `docs/report.md`. Everything else, including underscore names, is internal.
  - From 1.0, breaking changes need a major version. Report keys may be added in minor versions, and deprecations warn for at least one minor release before removal.
  - Documented in a new `docs/api.md`.
- **Packaging cleanup before the freeze.**
  - **BREAKING:** remove the `lightgbm` extra. It installs a library that no built-in model uses, and `all` now means XGBoost and CatBoost.
  - Add a `plot` extra (matplotlib).
  - Add trove classifiers, project URLs, and keywords.
- **Planning.** Archive the five completed OpenSpec changes (0.5–0.9), so `openspec/specs/` holds the full baseline.
- **Version:** 1.0.0.

**Report schema:** unchanged. The report is a `dict` subclass with the same keys. Code that checks `type(report) is dict` (rather than `isinstance`) would see a different type; this is called out in the changelog.

## Non-goals

- New ranking methods or report fields.
- Interactive or HTML plots, or multiple plot kinds.
- A typed-dict schema for the report, or switching the report to a dataclass (which would break dict access).
- Fixing CatBoost on scikit-learn 1.8 and later; that is an upstream issue, documented as a known issue.
- Publishing the 1.0 release itself. Tag, push, and the PyPI trusted-publisher setup stay with the maintainer.

## Capabilities

### New Capabilities

- `report-object`: the `Report` type, its dict compatibility, and `summary`/`to_frame`/`plot`.
- `public-api`: what is public, `__version__`, typing, and the stability policy.

### Modified Capabilities

(none; specs for 0.5–0.9 are archived as-is first)

## Impact

- **Code:**
  - `featranker/report.py` (new);
  - `featranker/importance.py`: return `Report` from both methods;
  - `featranker/__init__.py`: `Report` and `__version__`;
  - `featranker/selection.py`: type fixes;
  - `featranker/py.typed` (new).
- **Packaging and CI:** `pyproject.toml` (extras, classifiers, URLs, package data, mypy config); `.github/workflows/tests.yml` (mypy, matplotlib); `.github/workflows/release.yml` (new).
- **Docs:**
  - `docs/api.md` (new);
  - `docs/usage.md`: the report object section;
  - `docs/report.md`;
  - `docs/models.md`: remove the LightGBM row;
  - `README.md` and `docs/README.md`.
- **Tests:** a new `tests/test_report.py`; an updated extras test.
