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
| `failures` | warning for any `fit` or `ranking` failure, otherwise info | one finding per failure stage present | model names for that stage |
| `few_families` | warning | consensus is non-empty and `n_families` < 3 | `n_families`, `n_models` |
| `tied_groups` | info | two or more consensus groups share the same `median_rank` | the groups and their shared `median_rank`; one finding per tie set |
| `within_noise` | info | a consensus group's importance `mean <= std` in more than half of its voting models | the groups and their counts |

#### Scenario: Gate off
- **WHEN** `report["baseline_score"] is None`
- **THEN** the first finding has code `gate_off` and severity `"warning"`

#### Scenario: Optional dependency missing only
- **WHEN** the only failures have stage `initialization`
- **THEN** the `failures` finding has severity `"info"`

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
`summarize` SHALL raise `TypeError` when given a non-dict. It SHALL raise `ValueError` naming the missing key when a required report key is absent. The required keys are `baseline_score`, `models`, `consensus`, and `failures`.

#### Scenario: Missing key
- **WHEN** `summarize({"models": {}})` is called
- **THEN** a `ValueError` mentioning `baseline_score` is raised

### Requirement: Resampled reports are rejected with guidance
`summarize` SHALL raise `ValueError` when given a report with `evaluation_mode == "resampled"`. The message SHALL say to summarize each entry of `report["splits"]` (its `report` field) instead.

#### Scenario: Resampled report
- **WHEN** `summarize(resampled_report)` is called
- **THEN** a `ValueError` mentioning `splits` is raised

### Requirement: Groups below shadow are reported
When consensus rows include `above_shadow_share`, `summarize` SHALL emit one `below_shadow` finding, with severity `info`, listing every consensus group whose `above_shadow_share < 0.5` along with its share. The finding SHALL come after `within_noise`. Without `above_shadow_share`, no `below_shadow` finding SHALL be emitted.

#### Scenario: Below shadow
- **WHEN** consensus groups `a` and `b` have `above_shadow_share` 1.0 and 0.25
- **THEN** one `below_shadow` finding lists `b` with share 0.25

#### Scenario: No shadows
- **WHEN** the report has no `above_shadow_share`
- **THEN** no `below_shadow` finding is emitted

