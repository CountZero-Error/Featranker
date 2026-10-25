# prepush-fixes Specification

## Purpose
TBD - created by archiving change prepush-fixes. Update Purpose after archive.
## Requirements
### Requirement: Later selectors are not certified
With shadows on, a model with a selector, `FeatureUnion`, or `ColumnTransformer` after a step whose column mapping is not tracked, or wrapped in a meta-estimator, SHALL get a `shadows` failure saying the shadows could not be verified.

#### Scenario: Selector wrapped in a meta-estimator
- **WHEN** the model is, or contains (after a scaler, in a `FeatureUnion` part, or in a `ColumnTransformer` branch), a meta-estimator such as `TransformedTargetRegressor` or `BaggingRegressor` wrapping a pipeline with `SelectKBest`
- **THEN** the model has a `shadows` failure saying the shadows could not be verified

#### Scenario: Selector after passthrough transformer
- **WHEN** a pipeline is a passthrough `ColumnTransformer`, then `SelectKBest(k=2)`, then `Ridge` on two real and two shadow columns
- **THEN** the model has a `shadows` failure

### Requirement: Only native scorers share predictions
Prediction sharing SHALL apply only to scorers whose class is one of scikit-learn's own scorer classes; subclasses and other callables SHALL be called directly.

#### Scenario: Overridden call
- **WHEN** a secondary scorer subclasses scikit-learn's scorer and overrides `__call__`
- **THEN** the report equals the one produced without sharing

### Requirement: Symmetric training-row check
`ablate()` SHALL raise `ValueError` when a report and the ranker disagree on the training rows, including when one used every row and the other a subsample; a report without a row hash was ranked on every row.

#### Scenario: Full report, subsampled ranker
- **WHEN** a report from a full-data fit is ablated after the ranker was refit with `max_samples`
- **THEN** `ValueError` is raised

#### Scenario: Pre-1.21 report
- **WHEN** a report without a row hash is ablated with a subsampled ranker
- **THEN** `ValueError` is raised, and a full-data ranker is accepted

### Requirement: Consistent fingerprint scheme
Every fingerprint in a run block SHALL use the block's `fingerprint_scheme`, including shuffle-label fingerprints.

#### Scenario: Old pickle with shuffle labels
- **WHEN** a scheme-1 ranker ranks with `shuffle="within_groups"` and integer labels
- **THEN** both shuffle-label fingerprints use scheme 1

### Requirement: Resampled constant-group finding
For resampled reports, the `constant_within_groups` finding SHALL list the splits in which each group was constant and SHALL state the zero importance only for those splits.

#### Scenario: Constant in one split
- **WHEN** a group is constant within shuffle groups in split 0 only
- **THEN** the finding names split 0 for it

