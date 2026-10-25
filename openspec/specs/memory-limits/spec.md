# memory-limits Specification

## Purpose
TBD - created by archiving change memory-limits. Update Purpose after archive.
## Requirements
### Requirement: Category cap
`FeatureRanker(max_categories=None)` SHALL accept None or an int of at least 2; other values SHALL raise `ValueError`. When set, built-in linear models SHALL one-hot code at most that many levels per categorical column, grouping the rest as infrequent; an unseen level SHALL share the infrequent column only for a column with at least that many levels, and otherwise be all zeros. When None, pipelines and reports SHALL be unchanged. `ConsensusSelector` and the CLI (`--max-categories`) SHALL pass it through.

#### Scenario: High-cardinality column capped
- **WHEN** a categorical column has 1000 levels and `max_categories=20`
- **THEN** each built-in linear model's encoder outputs at most 20 columns for it

#### Scenario: Default unchanged
- **WHEN** `max_categories` is not given
- **THEN** reports equal those of the previous version

#### Scenario: Unseen level at equality
- **WHEN** a categorical column has exactly `max_categories` levels and an unseen level is encoded
- **THEN** it falls into the infrequent column

### Requirement: Lower-peak shadows and correlations
Shadow columns and Spearman correlations SHALL be computed with fewer full-size temporaries and SHALL produce values identical (bit for bit) to the previous version. The peak memory of the Spearman step SHALL NOT exceed the previous version's at any width.

#### Scenario: Same shadows
- **WHEN** shadows are built for the same data and seed in joint and independent modes
- **THEN** they equal those of the previous construction

#### Scenario: Same correlations at any size
- **WHEN** Spearman correlations are computed for 1M x 3, 400k x 65, 300 x 1, 300 x 65 or 100 x 129 values
- **THEN** they are bit-identical to the previous implementation's

#### Scenario: Peak not above before
- **WHEN** the Spearman matrix of 20000 x 10, 20000 x 64 and 20000 x 500 values is computed
- **THEN** the peak memory is at most the previous implementation's, and under half of it from 64 columns

### Requirement: Training subsample
`fit(X, y, max_samples=..., random_state=...)` SHALL fit models on a random subsample of training rows (whole groups under `shuffle="within_groups"`, original order under `"circular"`), SHALL raise `ValueError` when the sampled weights sum to zero or the sample misses a class of a classification target, SHALL fingerprint the full data passed, and reports SHALL record `run.n_train_rows`, `run.train_max_samples`, `run.train_random_state` (when given) and `run.train_rows_sha256` (SHA-256 of the sorted row indices). `ablate()` SHALL raise `ValueError` when the report has `train_rows_sha256` and the ranker's rows hash differently. The levels and missing counts a report shows SHALL be those of the sample. Without `max_samples`, behaviour and reports SHALL be unchanged. `rank_resampled(max_train_samples=)`, `ConsensusSelector(max_train_samples=)`, and CLI `--max-train-samples` SHALL pass it through.

#### Scenario: Fit on a share of rows
- **WHEN** `fit(..., max_samples=0.25, random_state=0)` on 400 rows
- **THEN** every model is fit on 100 rows and `run.n_train_rows` is 100

#### Scenario: Reproducible
- **WHEN** the same `random_state` is used twice
- **THEN** the reports are equal

#### Scenario: Class missing from the sample
- **WHEN** a classification target has a class of two rows in 400 and `max_samples=40` draws neither
- **THEN** `fit()` raises `ValueError` naming the class

#### Scenario: Ablation on other rows
- **WHEN** `ablate()` receives a report with `train_rows_sha256` and a ranker fit on other rows
- **THEN** it raises `ValueError`

