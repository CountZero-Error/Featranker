# report-summary Specification

## Purpose
Turn the report reading guide into a deterministic, offline function that lists what a user should notice in a report, so that interpretation does not depend on reading raw numbers by hand.
## Requirements
### Requirement: summarize returns structured findings
`featranker.summarize(report)` SHALL accept a report dict produced by `rank_features()` and return a list of findings.
- Each finding SHALL be a dict with `code` (str), `severity` (`"info"` or `"warning"`), `message` (str), and `evidence` (a JSON-serializable dict).
- The function SHALL NOT modify the report.
- The function SHALL NOT perform I/O.
- The same report SHALL always produce the same list, in the same order.

#### Scenario: Pure and deterministic
- **WHEN** `summarize(report)` is called twice on the same report
- **THEN** both results are equal, the report is unchanged, and `json.dumps(result, allow_nan=False)` succeeds

#### Scenario: Clean report yields no warnings
- **WHEN** the report has an active gate, no failures, a non-empty consensus with at least three families, no ties, and no groups within shuffle noise
- **THEN** `summarize(report)` contains no finding with severity `"warning"`

### Requirement: Finding rules
`summarize` SHALL emit findings in the order listed below:

| Code | Severity | Emitted when | Evidence |
|---|---|---|---|
| `gate_off` | warning | `baseline_score` is `None` | the dummy failure, if present |
| `empty_consensus` | warning | `consensus` is empty | number of ranked models |
| `failures` | warning for any `fit`, `ranking`, or `shadows` failure, otherwise info | one finding per failure stage present | model names for that stage |
| `few_families` | warning | consensus is non-empty and `n_families` < 3 | `n_families`, `n_models` |
| `tied_groups` | info | two or more consensus groups share the same `median_rank` | the groups and their shared `median_rank`; one finding per tie set |
| `within_noise` | info | a consensus group's importance `mean <= std` in more than half of its voting models | the groups and their counts |

#### Scenario: Gate off
- **WHEN** `report["baseline_score"] is None`
- **THEN** the first finding has code `gate_off` and severity `"warning"`

#### Scenario: Optional dependency missing only
- **WHEN** the only failures have stage `initialization`
- **THEN** the `failures` finding has severity `"info"`

#### Scenario: Shadow-incompatible model
- **WHEN** `failures` contains a failure with stage `"shadows"`
- **THEN** the `failures` finding for that stage has severity `"warning"`

#### Scenario: Ties
- **WHEN** consensus groups `a` and `b` both have `median_rank == 2.0`
- **THEN** a `tied_groups` finding lists `a` and `b` with `median_rank == 2.0`

#### Scenario: Within noise
- **WHEN** group `age` has `mean <= std` in 3 of 4 voting models
- **THEN** a `within_noise` finding includes `age`

#### Scenario: Few families
- **WHEN** the consensus rows have `n_families == 2`
- **THEN** a `few_families` warning is emitted

### Requirement: Invalid input
`summarize` SHALL raise `TypeError` when given a non-dict. It SHALL raise `ValueError` naming the missing key when a required report key is absent. The required keys are `baseline_score`, `models`, `consensus`, and `failures`; for a report with `evaluation_mode == "resampled"` they are `splits`, `stability`, and `failures`.

#### Scenario: Missing key
- **WHEN** `summarize({"models": {}})` is called
- **THEN** a `ValueError` mentioning `baseline_score` is raised

### Requirement: Groups below shadow are reported
When consensus rows include `above_shadow_share`, `summarize` SHALL emit one `below_shadow` finding, with severity `info`, listing every consensus group whose `above_shadow_share < 0.5` along with its share. The finding SHALL come after `within_noise`. Without `above_shadow_share`, no `below_shadow` finding SHALL be emitted.

#### Scenario: Below shadow
- **WHEN** consensus groups `a` and `b` have `above_shadow_share` 1.0 and 0.25
- **THEN** one `below_shadow` finding lists `b` with share 0.25

#### Scenario: No shadows
- **WHEN** the report has no `above_shadow_share`
- **THEN** no `below_shadow` finding is emitted

### Requirement: Overfitting models are reported
When model reports include `train_score` and `baseline_score` is not `None`, `summarize` SHALL emit one `overfit` finding, with severity `info`, listing every model whose `train_score - baseline_score > 0` and whose `(train_score - evaluation_score) / (train_score - baseline_score) >= 0.5`, that is, a model that lost at least half of its training advantage over the dummy on the evaluation data. Each listed model SHALL carry its `train_score`, `evaluation_score`, and that share. Models with a `null` `train_score` SHALL be skipped. The finding SHALL come after `below_shadow`. Without `train_score`, no `overfit` finding SHALL be emitted.

#### Scenario: Overfit tree
- **WHEN** a model has `train_score` 1.0, `evaluation_score` 0.2, and `baseline_score` is 0.0
- **THEN** an `overfit` finding lists it with share 0.8

#### Scenario: Healthy model
- **WHEN** a model has `train_score` 0.56, `evaluation_score` 0.64, and `baseline_score` is 0.0
- **THEN** it is not listed

### Requirement: Correlated groups are reported
When the report has `evidence.correlation.pairs` and it is non-empty, `summarize` SHALL emit one `correlated_groups` finding, with severity `info`, listing the pairs and their `rho`, and saying that permutation importance may be split between correlated groups. The finding SHALL come after `overfit`. Otherwise no `correlated_groups` finding SHALL be emitted.

#### Scenario: Correlated pair
- **WHEN** `evidence.correlation.pairs` is `[{"groups": ["e", "f"], "rho": 0.99}]`
- **THEN** a `correlated_groups` finding lists `e` and `f`

#### Scenario: Evidence stage failures
- **WHEN** `failures` contains a failure with stage `"evidence"`
- **THEN** the `failures` rule emits a finding for that stage with severity `"info"`

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

