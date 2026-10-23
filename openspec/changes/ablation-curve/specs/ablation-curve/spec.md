## ADDED Requirements

### Requirement: Ablation curve
`FeatureRanker.ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None, eval_sample_weight=None)` SHALL return a new `Report` equal to `report` plus an `ablation` block, and SHALL NOT modify `report`. For each `k`, it SHALL refit the report's voting models (`in_consensus: true`) on the columns of the top-k consensus groups, with the ranker's preprocessing configuration and without shadows or evidence, and score them on `X_eval` with the same columns. The block SHALL hold `scoring`, `models`, `baseline_score` (the dummy's score from the refit run itself, on every `X_eval` row), `n_groups`, `curve` (one row per `k` with `k`, `feature_groups`, `n_features`, `scores`, `median_score`), and `failures`. Values SHALL be strict-JSON (finite floats or null).

#### Scenario: Signal groups lead the curve
- **WHEN** y depends on two of six features and the report ranks them first
- **THEN** the curve's `median_score` at `k=2` is higher than at `k=1` and close to the one at `k=6`

#### Scenario: Report unchanged
- **WHEN** `ablate()` returns
- **THEN** the input report has no `ablation` key

### Requirement: Ablation inputs
`ablate()` SHALL raise `ValueError` when the ranker is not fitted, the report is resampled, the consensus is empty, `ks` holds a value that is not a positive int at most the number of consensus groups, or the training data fingerprint differs from a non-null `report["run"]["train_data_sha256"]`. When that fingerprint is non-null, `sample_weight` SHALL also fingerprint to `report["run"]["train_weights_sha256"]` (no weights when the key is absent), else `ValueError`; and when `X_eval` and `y_eval` fingerprint to `report["run"]["data_sha256"]`, `eval_sample_weight` SHALL fingerprint to `weights_sha256` the same way. A report without a `run` block SHALL skip these checks. A column in the report's `feature_groups` that the ranker lacks SHALL raise `ValueError`. Default `ks` SHALL be 1, 2, 3, 5, 10, 20, 50, 100, … up to `n` plus `n`. Duplicate `ks` SHALL be removed and sorted.

#### Scenario: Default ks
- **WHEN** the consensus has 7 groups and `ks` is not given
- **THEN** the curve's `k` values are `[1, 2, 3, 5, 7]`

#### Scenario: Other training weights
- **WHEN** the ranking used `sample_weight` and `ablate()` gets none, or the reverse
- **THEN** `ValueError` is raised

#### Scenario: Other training data
- **WHEN** `X` differs from the data the ranker was fit on
- **THEN** `ValueError` is raised

### Requirement: Ablation failures
A model that fails to fit or score at some `k` SHALL get `null` in that row's `scores` and a failure entry with `model`, `stage`, `message`, and `k`; other models and ks SHALL continue.

#### Scenario: One model fails
- **WHEN** one voting model raises on fit for every k
- **THEN** its scores are null and `median_score` uses the remaining models

### Requirement: Ablation summary
`summarize()` SHALL add one `info` finding with code `ablation_curve` when the report has an `ablation` block with a non-null `median_score`, giving the `k` with the highest `median_score` (smallest on ties) and the `median_score` at the largest scored `k` (the largest `k` whose `median_score` is not null).

#### Scenario: Finding present
- **WHEN** a report from `ablate()` is summarized
- **THEN** exactly one finding has code `ablation_curve`

### Requirement: CLI columns and ablation
The CLI SHALL accept `--drop` with comma-separated column names removed from both CSVs before features are read; an unknown name, the target, or the `--sample-weight` column SHALL be an error. `--ablate` SHALL add the `ablation` block to the written report; `--ablate-ks` SHALL take comma-separated positive ints and SHALL require `--ablate`.

#### Scenario: Drop an ID column
- **WHEN** both CSVs have an `id` column and `--drop id` is given
- **THEN** `id` is not in the report's `feature_groups`

#### Scenario: Ablate from the CLI
- **WHEN** `--ablate --ablate-ks 1,2` is given
- **THEN** the report's `ablation.curve` has `k` values `[1, 2]`
