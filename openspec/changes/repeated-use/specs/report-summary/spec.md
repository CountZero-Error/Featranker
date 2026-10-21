## ADDED Requirements

### Requirement: Resampled reports are summarised
`summarize` SHALL accept a report with `evaluation_mode == "resampled"` and return these findings, in this order, each only when its condition holds:
- `split_failures` (warning): splits that failed outright, with their indices and error types.
- `failures`: model failures aggregated over splits, one finding per stage, each model listed with the splits it failed in; severity as for single-split reports.
- `gate_off` (warning): splits whose `baseline_score` is `None`.
- `empty_consensus` (warning): splits whose consensus is empty.
- `below_shadow` (info): stability rows with `above_shadow_share < 0.5`, with their shares.
- `insufficient_splits` (info): shadows are on and `shadow_test.min_p_value > 0.05`, stating that no group can reach a q-value of 0.05 with this many splits.
Findings SHALL have the same shape as single-split findings, and the report SHALL not be modified.

#### Scenario: Failed split
- **WHEN** split 2 of a resampled report failed outright
- **THEN** a `split_failures` warning lists split 2

#### Scenario: Few splits
- **WHEN** a resampled report with shadows has 3 splits
- **THEN** an `insufficient_splits` finding is emitted

#### Scenario: Clean resampled report
- **WHEN** every split ranked with a gate and a consensus, without shadows
- **THEN** `summarize` returns no warnings

## REMOVED Requirements

### Requirement: Resampled reports are rejected with guidance
**Reason**: Resampled reports now have their own findings.
**Migration**: Call `summarize(resampled_report)` directly; per-split summaries still work on `report["splits"][i]["report"]`.
