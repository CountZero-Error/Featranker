## MODIFIED Requirements

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
