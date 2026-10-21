# run-record Specification

## Purpose
Let users tell whether two reports came from the same data, splits, preset, and grouping settings.
## Requirements
### Requirement: Run block
Every report from `rank_features()` and `rank_resampled()` SHALL include `run` with `task`, `group`, and `data_sha256`. In a `rank_features()` report, `data_sha256` SHALL fingerprint the evaluation data and `train_data_sha256` SHALL fingerprint the data passed to `fit()`. In a `rank_resampled()` report, `data_sha256` SHALL fingerprint all data passed, `cv` SHALL be the integer given, the `repr` of the splitter with any ` at 0x...` memory address removed, or `"iterable of splits"` for an iterable of splits, and `splits_sha256` SHALL fingerprint the train and evaluation indices of every split in order. Each split report SHALL carry its own `run`. A fingerprint SHALL be the SHA-256 hex digest of the feature names, the feature values, and the target values; equal inputs under the same library versions SHALL give equal fingerprints, and changing any value, name, or row order SHALL change it. When sample weights are given, the block SHALL also include `weights_sha256`, the SHA-256 hex digest of the weights passed to `rank_features()` (or to `rank_resampled()`), and, in a `rank_features()` report, `train_weights_sha256` for the weights passed to `fit()`; without weights these keys SHALL be absent. The block SHALL be JSON-safe.

#### Scenario: Same inputs
- **WHEN** the same ranker settings rank the same data twice
- **THEN** both reports have equal `run` blocks

#### Scenario: One value changed
- **WHEN** one evaluation value differs between two runs
- **THEN** `data_sha256` differs and `train_data_sha256` is equal

#### Scenario: Different splits
- **WHEN** `rank_resampled` runs twice with `KFold(5, shuffle=True)` and different `random_state` values on the same data
- **THEN** `data_sha256` is equal and `splits_sha256` differs

#### Scenario: Splitter without a stable repr
- **WHEN** `rank_resampled` runs twice with equal `KFold(3, shuffle=True, random_state=RandomState(0))` splitters, or with a splitter class that has no `__repr__`
- **THEN** both reports have equal `run` blocks

#### Scenario: Weights changed
- **WHEN** only the sample weights differ between two runs
- **THEN** `weights_sha256` differs, `data_sha256` is equal, and a run without weights has no `weights_sha256`

### Requirement: Automatic grouping record and threshold
`ConsensusSelector(feature_groups="auto", auto_groups_threshold=0.7)` and the CLI option `--auto-groups-threshold` (default 0.7, used with `--feature-groups auto`) SHALL pass the threshold to `auto_groups`; an invalid threshold SHALL raise `ValueError` as in `auto_groups`. When groups were proposed automatically, the report SHALL include `run["auto_groups"] = {"threshold": t, "method": "spearman-complete-linkage"}`; otherwise no `auto_groups` key SHALL appear in `run`.

#### Scenario: Looser threshold
- **WHEN** two columns have |rho| 0.6 and the selector uses `auto_groups_threshold=0.5`
- **THEN** they form one group and `report_["run"]["auto_groups"]["threshold"] == 0.5`

#### Scenario: Explicit groups
- **WHEN** the selector is given a feature_groups dict
- **THEN** `"auto_groups" not in report_["run"]`

