## Why

A cross-cutting review of 1.1–1.4 (Signal, 2026-10-20) found two bugs and several claims the code does not support. The most serious: the shadow hits test is presented as a significance test, but its 0.5 null is not established for arbitrary estimators or overlapping splits (a custom-model counterexample gives noise q ≈ 2e-5). 1.1–1.4 are unreleased, so these can be fixed before any user sees them.

## What Changes

- Shadow hits test relabelled as uncalibrated: `shadow_test` gains `calibrated: false` and a `note`; docs call the p/q values binomial reference values and `max_shadow_q` a heuristic. Calibration is studied in 1.6.
- Bug: `fit()` and `rank_features()` copy the features and target they receive, so models and kept evidence data never alias the caller's arrays; changing them afterwards cannot change later reports.
- Bug: `ConsensusSelector` objects pickled before `max_shadow_q` existed load, clone, and refit (`max_shadow_q` defaults to `None`).
- Docs corrections: SHAP values are on the explainer's output scale (may be a link scale); linear `direction` is the coefficient's sign when defined, else null; a train/eval gap can come from distribution shift, not only memorization; `group="fast"` represents the same candidate families, but which ones vote can differ; CLI `--group linear` help lists what the group contains; benchmark reproduction installs `.[all]` with pinned versions; the selection-by-shuffling runtime sentence; a time-series and grouped-CV recipe.
- Small items from earlier reviews: the `fast` tag must be exactly `true`; the scaler test covers every fast linear model; a CLI test for an unknown `--group`; `"passthrough"`/`None` pipeline steps treated as identity in the SHAP summary.
- Roadmap: limitation list corrected (L2, L3, L5) and extended with Signal's additions.
- Version 1.5.0.

Report schema: additive only (`shadow_test.calibrated`, `shadow_test.note`).

## Non-goals

- Calibrating the hits test or changing any selection rule or threshold (1.6).
- New features from the additions list.

## Capabilities

### New Capabilities

### Modified Capabilities
- `shadow-significance`: `shadow_test` states that the values are uncalibrated.
- `mechanism-evidence`: the ranker owns copies of its inputs; identity pipeline steps are explained.
- `fast-preset`: wording on which families vote.
- `consensus-selector`: older pickles load and refit.

## Impact

- Code: `featranker/importance.py`, `featranker/selection.py`, `featranker/evidence.py` (reason text), `featranker/cli.py` (help), `featranker/importance_config.yaml` loader check.
- Docs: docs/usage.md, docs/report.md, docs/models.md, docs/benchmark.md, README; roadmap in openspec/config.yaml; CHANGELOG (local).
