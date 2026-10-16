## Context

- Reports are plain dicts built at the end of `rank_features()` and `rank_resampled()`.
- `summarize`, `to_prompt`, `explain`, and the CLI accept any dict.
- Packaging is setuptools with explicit `package-data`. CI runs pytest on Python 3.10, 3.12, and 3.13, each with scikit-learn 1.5.x and the latest release.
- PyPI uploads are manual today.
- Requirements are in specs/. Motivation is in proposal.md.

## Goals / Non-Goals

**Goals:**
- Zero breakage for dict users: `Report` adds methods and nothing else.
- Optional dependencies stay optional, imported lazily inside the methods.

**Non-Goals:**
- A schema type for report contents (proposal non-goal).

## Decisions

**1. `Report(dict)`, not a dataclass or a `Mapping` wrapper.**

Subclassing `dict` keeps indexing, mutation, JSON, and `isinstance(..., dict)` exactly as before, and the methods read `self`. A dataclass or wrapper would break `report["consensus"]` or `json.dumps(report)`.

Trade-off: `type(report) is dict` checks change, which is called out in the changelog.

**2. Wrapping happens where reports are built.**

`rank_features()` returns `Report(report)`. `rank_resampled()` gets each split's `Report` from `rank_features`, and wraps its own result too. `json.dumps` and pickle work on dict subclasses without extra code.

**3. Lazy optional imports with actionable errors.**

`to_frame()` imports pandas, and `plot()` imports matplotlib, inside the method. Each raises `ImportError` naming the package and extra. Importing FeatRanker never pulls in either.

**4. One plot per report kind, returning `Axes`.**

- Single-split: a dot plot of family ranks per group, with the median marker.
- Resampled: a min–max range line per group, with the median marker.

Groups run from top to bottom in consensus/stability order. Lower rank is better, so the x axis is labeled "rank (1 = most important)". Returning `Axes` lets users restyle the plot or save it.

Alternatives considered: several plot kinds, or seaborn. Rejected as YAGNI and an extra dependency.

**5. Typing fixes are minimal.**

- Add `types-PyYAML` to the test extra.
- Annotate `ConsensusSelector.task`/`group` with the same `Literal` types as `FeatureRanker`.
- Convert column indices with `np.asarray(..., dtype=np.intp)` where NumPy's stubs require it.
- A `[tool.mypy]` section sets `ignore_missing_imports` only for libraries without stubs (sklearn, joblib, tqdm, xgboost, catboost, pandas, matplotlib).

**6. Trusted publishing workflow.**

On push of a `v*` tag:
1. check out the code, set up Python, and build the sdist and wheel;
2. run `twine check`;
3. check that the tag equals `v` plus the `pyproject.toml` version;
4. publish with `pypa/gh-action-pypi-publish@release/v1` under a GitHub environment named `pypi`, with `permissions: id-token: write`.

No secrets are stored. It only works after the maintainer adds the trusted publisher on pypi.org; the docs give the exact fields.

Alternatives considered:
- an API-token secret. Rejected: it is a stored credential;
- publishing on every push. Rejected: releases must be deliberate.

**7. Archive before the freeze.**

The five completed changes are archived in version order, so their specs land in `openspec/specs/`. Later 1.x changes can then modify them.

## Risks / Trade-offs

- [A 1.0 API freeze before real-world use of the 0.5–0.9 features] → The maintainer decided to go straight to 1.0. The policy still allows additive minor releases, and the changelog states that 0.5–0.9 were never released separately.
- [Removing the `lightgbm` extra breaks `pip install featranker[lightgbm]`; pip only warns about an unknown extra] → Call it out in the changelog. Better before 1.0 than after.
- [The trusted publisher is not configured, so the first tag push fails at publish] → Docs give the setup steps. The build and check steps still validate the release.

## Migration Plan

- Release as 1.0.0 via the new workflow, or a manual upload as a fallback.
- Rollback means yanking 1.0.0 on PyPI and fixing forward in 1.0.1.
