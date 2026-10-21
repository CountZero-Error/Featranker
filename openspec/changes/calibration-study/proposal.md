## Why

1.5 labels the shadow hits test as uncalibrated, and 1.6 adds `min_shadow_families`, whose false-selection risk is unknown. Users need measured numbers: how often each rule selects pure noise, how often it finds planted signal (including interactions), and whether independent shadow shuffling matters for collinear features. Fresh seeds and preregistered decision rules keep the study honest, as in the investigator spike.

## What Changes

- New `benchmarks/calibration.py` (not shipped), frozen by SHA-256 before the run, with its own tests:
  - **Null study:** the target is independent of the features; regression and classification; seeds 100–119; `group` in {`fast`, `tree`}; `shadow_mode` in {`joint`, `independent`}; `RepeatedKFold(5, 2)` (10 splits). Measures the distribution of `shadow_p_value` (fraction ≤ 0.05) and how many noise groups each rule selects.
  - **Signal study:** the existing benchmark generator (linear, threshold, interaction, correlated copy, noise) on seeds 100–109, same grid. Measures recall and noise selected per rule.
  - Rules: `min_shadow_share=0.5`; `max_shadow_q=0.05`; `min_shadow_families` = 1, 2, 3.
- Preregistered decisions (design D3) that set what the docs may claim and which `min_shadow_families` value to recommend.
- Results in `benchmarks/results/<date>-calibration.json`; `docs/calibration.md` with method, numbers, and the decisions; links from docs/usage.md and docs/benchmark.md. The `shadow_test.note` text and docs claims are updated only as the decisions allow.

## Non-goals

- Changing defaults or thresholds in the package.
- Calibrating for custom estimators (out of reach of a benchmark; the report keeps `calibrated: false`).

## Capabilities

### New Capabilities

### Modified Capabilities

None: no package behaviour changes (`skip_specs: true`). Possible text-only changes to docs and to the `shadow_test.note` string are allowed by the 1.5 spec (it only requires the note to say the values are not calibrated for arbitrary estimators).

## Impact

- New: `benchmarks/calibration.py`, `benchmarks/test_calibration.py`, results files, `docs/calibration.md`.
- Runs in the scratch virtual environment; no new dependencies.
