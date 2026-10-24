# trust-fixes Specification

## Purpose
TBD - created by archiving change trust-fixes. Update Purpose after archive.
## Requirements
### Requirement: Nested shadow-drop detection
With `shadows=True`, a model in which any fitted `ColumnTransformer`, `FeatureUnion` part or feature selector — in a `Pipeline` or nested in another transformer's branch — leaves a shadow column unused SHALL be recorded as a `shadows` failure and SHALL NOT contribute shadow evidence. When the columns a model keeps cannot be determined — private attributes are not as expected, a meta-estimator wraps a `ColumnTransformer`, or a `ColumnTransformer` follows a `Pipeline` step not known to keep the columns one for one — the model SHALL also get a `shadows` failure whose message says the shadows could not be verified. Other estimators (such as `FunctionTransformer`) are assumed to use every column.

#### Scenario: Inner transformer drops shadows
- **WHEN** an outer `ColumnTransformer` passes all columns to an inner one that keeps only the real columns
- **THEN** the model has a `shadows` failure and is absent from shadow results

#### Scenario: Later transformer after a reordering step
- **WHEN** a `ColumnTransformer` that reorders the columns is followed by another `ColumnTransformer` in a `Pipeline`
- **THEN** the model has a `shadows` failure saying the shadows could not be verified

#### Scenario: Later transformer after a scaler
- **WHEN** a `StandardScaler` is followed by a `ColumnTransformer` that keeps only real columns
- **THEN** the model has a `shadows` failure saying it does not use every shadow column

#### Scenario: Selector, union and meta-estimator
- **WHEN** a `SelectKBest` or a `FeatureUnion` of transformers leaves shadow columns unused, or a `TransformedTargetRegressor` wraps a `ColumnTransformer`
- **THEN** the first two give a "does not use every shadow column" failure and the third a "could not be verified" failure

#### Scenario: Nested passthrough keeps shadows
- **WHEN** nested transformers pass every column through
- **THEN** no `shadows` failure is recorded

### Requirement: Dependency floor
The package metadata SHALL require `scikit-learn>=1.5`.

#### Scenario: Floor declared
- **WHEN** `pyproject.toml` is read
- **THEN** its scikit-learn dependency is `scikit-learn>=1.5`

### Requirement: Lossless fingerprints
Fingerprints SHALL hash integer and boolean columns without conversion to float, tagged by dtype kind. Every `run` block SHALL record the `fingerprint_scheme` its fingerprints were made with: 2, or 1 for a ranker pickled before this scheme. `ablate()` SHALL compare a report without `fingerprint_scheme` using the previous scheme.

#### Scenario: Large integer labels differ
- **WHEN** two classification targets differ only by swapping labels `2**53` and `2**53+1`
- **THEN** their fingerprints differ

#### Scenario: Size does not matter
- **WHEN** two integer columns hold equal values as `int32` and `int64`
- **THEN** their fingerprints are equal

#### Scenario: Old pickled ranker
- **WHEN** a ranker pickled before this scheme ranks and its report is ablated
- **THEN** the run block says scheme 1 and `ablate()` accepts the report

#### Scenario: Old report still ablates
- **WHEN** a report has no `fingerprint_scheme` and its fingerprint was made with the previous scheme from the same data
- **THEN** `ablate()` accepts it

### Requirement: Ablation provenance and cohort
The `ablation` block SHALL record `fingerprint_scheme`, `data_sha256`, `weights_sha256`, `n_eval_rows`, `same_holdout`, and `paired_models`; each curve row SHALL record `n_models_scored` and `paired_median_score` (the median over `paired_models`, null when it is empty). `summarize()` SHALL choose the best k by `paired_median_score` when `paired_models` is non-empty and SHALL add a `warning` finding `ablation_cohort_changed` when the set of models with a score differs across rows. A block without `paired_models` SHALL be described without claiming that no model was paired.

#### Scenario: Cohort change does not create a best k
- **WHEN** one model scores at every k and two others fail at k=1
- **THEN** the best k is chosen from the one paired model and `ablation_cohort_changed` is reported

#### Scenario: Equal counts, different models
- **WHEN** one model fails at k=1 and another at k=2
- **THEN** `ablation_cohort_changed` is reported though both rows score the same number of models

#### Scenario: Third data set recorded
- **WHEN** `ablate()` is given evaluation data other than the ranking holdout
- **THEN** `same_holdout` is false and `data_sha256` is that data's fingerprint

### Requirement: Zero-weight subsample
`rank_features()` SHALL raise `ValueError` when `max_samples` and `sample_weight` are given and the sampled rows' weights sum to zero.

#### Scenario: All sampled weights zero
- **WHEN** only one row has positive weight and `max_samples` draws other rows
- **THEN** `ValueError` is raised before any scoring

