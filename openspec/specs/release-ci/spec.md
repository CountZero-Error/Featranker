# release-ci Specification

## Purpose
TBD - created by archiving change release-ci. Update Purpose after archive.
## Requirements
### Requirement: Installed-artifact smoke test
`tests/smoke_installed.py` SHALL take `--expect-version` and exit non-zero unless the imported `featranker` is installed outside the source tree, both the installed metadata version and `featranker.__version__` equal the expected version, the packaged registry and `py.typed` exist, a small fit/rank/summarize/ablate run and a `ConsensusSelector` pipeline succeed, and the `featranker` console script writes a report for two small CSVs. With `--extras` it SHALL also require an xgboost and a catboost model to rank, SHAP evidence to be present and `report.plot()` to work. Without `--extras` it SHALL require that xgboost, catboost, shap and matplotlib are not importable, that the report's SHAP evidence is unavailable, and that `report.plot()` raises an `ImportError` naming `featranker[plot]`.

#### Scenario: Run from the checkout
- **WHEN** the script imports `featranker` from the source tree
- **THEN** it exits non-zero saying it is not testing an installed copy

#### Scenario: Version mismatch
- **WHEN** the installed version differs from `--expect-version`
- **THEN** it exits non-zero

#### Scenario: Installed wheel
- **WHEN** the built wheel is installed in a fresh environment and the script runs outside the checkout
- **THEN** it exits zero

### Requirement: CI and release run the smoke test
The tests workflow SHALL build the wheel and run the smoke test from a fresh virtualenv outside the checkout, core-only and with `[all,shap,plot]`, on the oldest supported Python with the oldest supported scikit-learn and on the newest Python with the newest scikit-learn, passing the `pyproject.toml` version as the expected version. The release workflow SHALL run it on the built wheel, with the same expected version, before uploading the artifact.

#### Scenario: Release blocks on a broken wheel
- **WHEN** the smoke test fails in the release build job
- **THEN** no artifact is uploaded and publish does not run

