## Why

CI and the release job test an editable install of the checkout. Nothing installs the built wheel, so a missing package-data file (the model registry YAML, `py.typed`), a broken console script, or an import that only works from the source tree would ship unnoticed. Optional extras (xgboost, catboost, shap, plot) are only exercised in the full test environment, never in a core-only install (external review, 2026-10-23).

## What Changes

- New `tests/smoke_installed.py`: a short script, runnable from any directory, that checks the installed package (not the checkout): import path is outside the source tree, installed metadata and `featranker.__version__` both equal the version passed with `--expect-version` (the checkout's `pyproject.toml` version), registry YAML and `py.typed` are present, a small `FeatureRanker` fit/rank/summarize/ablate run, `ConsensusSelector` in a pipeline, and the `featranker` console script on two small CSVs. Without `--extras` it requires a core-only environment and checks that SHAP evidence reports unavailable and `report.plot()` raises the documented `ImportError`; with `--extras` it requires xgboost/catboost models to rank, SHAP evidence and a plot.
- `tests.yml`: a `wheel` job builds the wheel and sdist, installs the wheel in a fresh virtualenv outside the checkout, and runs the smoke script for core-only and for `[all,shap,plot]`, on Python 3.10 with the oldest scikit-learn (`>=1.5,<1.6`) and Python 3.13 with the newest, and passes the `pyproject.toml` version to the script.
- `release.yml`: after `build` and `twine check`, install the built wheel in a fresh virtualenv outside the checkout and run the smoke script before the artifact is uploaded.
- Version 1.20.0.

## Non-goals

- Testing on Windows/macOS runners.
- Running the full pytest suite against the installed wheel.
- Changing the publish job.

## Capabilities

### New Capabilities
- `release-ci`: installed-artifact smoke test in CI and release.

### Modified Capabilities

## Impact

- Files: `tests/smoke_installed.py`, `.github/workflows/tests.yml`, `.github/workflows/release.yml`, docs/api.md ("Releasing"), CHANGELOG (local), roadmap 1.20.
