# family-consensus Specification

## Purpose
Combine per-model feature-group ranks into a consensus in which each model family gets one vote, so that near-duplicate built-in models cannot outvote structurally different ones.
## Requirements
### Requirement: Every model belongs to a family
Each model in a report SHALL carry a `family` string. Built-in models SHALL use the family declared for them in the model registry. A model supplied by the caller through `models` SHALL form its own family, named after the model.

#### Scenario: Built-in model reports its family
- **WHEN** a ranker using built-in classification models produces a report
- **THEN** `report["models"]["logistic_regression_l1"]["family"]` equals `report["models"]["logistic_regression"]["family"]`
- **AND** it differs from the family of `random_forest`

#### Scenario: Custom model is its own family
- **WHEN** a ranker is built with `models={"a": est1, "b": est2}`
- **THEN** model `a` has family `"a"` and model `b` has family `"b"`

### Requirement: Every built-in registry entry declares a family
Every entry in the built-in model registry SHALL declare a non-empty family. Each family SHALL contain models from only one of the `linear` and `tree` groups.

#### Scenario: Registry completeness
- **WHEN** the shipped registry is loaded
- **THEN** every classification and regression entry has a non-empty `family`
- **AND** no family name appears under both `linear` and `tree`

### Requirement: Consensus gives each family one vote
Only models with `in_consensus: True` SHALL vote. For each feature group, the consensus SHALL first take the median of the voting models' ranks within each family, giving one family rank per family that has at least one voting model. Then:
- `median_rank` SHALL be the median of those family ranks;
- `mean_rank` SHALL be their mean;
- `rank_std` SHALL be their population standard deviation.

The consensus order SHALL remain: `median_rank`, then `mean_rank`, then group name.

#### Scenario: Large family cannot outvote small families
- **GIVEN** voting models `m1`, `m2`, `m3` in family `F1`, each ranking group `x` first
- **AND** voting models `t1` in family `F2` and `t2` in family `F3`, each ranking group `x` third
- **WHEN** the consensus is built
- **THEN** the family ranks for `x` are `F1: 1.0`, `F2: 3.0`, `F3: 3.0`
- **AND** the row for `x` has `median_rank == 3.0` (a per-model median would give 1.0)

#### Scenario: One model per family matches the per-model consensus
- **WHEN** every voting model is in a different family
- **THEN** `median_rank`, `mean_rank`, `rank_std`, and the consensus order equal those a per-model consensus would produce

#### Scenario: Gated-out models do not vote within their family
- **WHEN** a family has one model with `in_consensus: True` and one with `in_consensus: False`
- **THEN** that family's rank equals the voting model's rank

### Requirement: Consensus rows expose family evidence
Each consensus row SHALL include:
- `n_models`: the number of voting models;
- `n_families`: the number of families with at least one voting model;
- `family_ranks`: a mapping from each of those families to its family rank, as a float.

All existing consensus keys SHALL keep their names and types.

#### Scenario: Row fields
- **WHEN** five voting models in three families rank group `x`
- **THEN** the row for `x` has `n_models == 5`, `n_families == 3`, and `family_ranks` with exactly three keys

#### Scenario: Report stays strict JSON
- **WHEN** a report with family fields is serialized with `json.dumps(report, allow_nan=False)`
- **THEN** serialization succeeds

### Requirement: Empty consensus behavior is unchanged
If no model votes, `consensus` SHALL be an empty list and a `UserWarning` SHALL be emitted, as before.

#### Scenario: No voters
- **WHEN** no model beats the dummy baseline
- **THEN** `consensus == []` and a `UserWarning` is emitted

