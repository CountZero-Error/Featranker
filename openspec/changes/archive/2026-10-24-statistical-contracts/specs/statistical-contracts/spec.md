## ADDED Requirements

### Requirement: Model families
`FeatureRanker(families=...)` SHALL accept `None` or a mapping of model name to non-empty string family (copied to a dict); other values SHALL raise `ValueError` at construction, and a name that is not a configured model SHALL raise `ValueError` in `fit()` and, before any split, in `rank_resampled()` and `ConsensusSelector.fit()`. Built-in models that failed to initialise SHALL count as configured. Mapped models SHALL use the given family in the report and in the consensus vote. `ConsensusSelector(families=...)` SHALL pass it to its rankers. Old pickles SHALL load with `None`.

#### Scenario: Near-identical custom models share one vote
- **WHEN** three custom ridge models are mapped to family `"ridge"` and one custom tree is unmapped
- **THEN** the consensus has two families

#### Scenario: Unknown model name
- **WHEN** `families` names a model that is not configured
- **THEN** `fit()`, `rank_resampled()` and `ConsensusSelector.fit()` raise `ValueError`

#### Scenario: Shadow family evidence follows the mapping
- **WHEN** three custom ridge models are mapped to family `"ridge"`, one tree is unmapped, and shadows are on
- **THEN** `family_above_shadow` has the keys `ridge` and `tree` and the stability `shadow_families` is at most 2

### Requirement: Constant within groups
With `shuffle="within_groups"`, `rank_features()` SHALL record `constant_within_groups`, the sorted names of feature groups whose every column is constant within every shuffle group on the scored rows, and SHALL emit a `UserWarning` when it is non-empty. Without that mode the key SHALL be absent. A resampled report with that mode SHALL have a top-level `constant_within_groups`, the sorted union of its split lists (`[]` when none). `summarize()` SHALL add a `warning` finding `constant_within_groups` for a held-out or resampled report when the list is non-empty.

#### Scenario: Patient-level predictor
- **WHEN** a feature is constant within each patient and `shuffle="within_groups"` with patient labels
- **THEN** the report lists that feature's group in `constant_within_groups` and `summarize()` warns

#### Scenario: Resampled report
- **WHEN** `rank_resampled()` runs with `shuffle="within_groups"` and a patient-level feature
- **THEN** the report's top-level `constant_within_groups` lists that feature's group and `summarize()` warns

#### Scenario: Default mode unchanged
- **WHEN** `shuffle="rows"`
- **THEN** the report has no `constant_within_groups` key

### Requirement: Documented statistical contracts
The docs SHALL state that within-group shuffling measures reliance on within-group variation only; that with sample weights shuffle donors are the scored rows regardless of weight, and shadows are unweighted copies of every row; that a data set used to choose k gives a validation score and final performance needs another test set or outer cross-validation with k chosen inside the outer loop (nested cross-validation); and that custom models are separate families unless `families` maps them. The ablation example SHALL run as shown.

#### Scenario: Ablation example runs
- **WHEN** the ablation example in docs/usage.md is run on its own
- **THEN** it completes and prints output matching the documented output
