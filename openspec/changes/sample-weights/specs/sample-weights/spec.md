## Purpose

Let users rank features under the same sample weights their models use, so fitting, the dummy baseline, and every score are weighted consistently.

## ADDED Requirements

### Requirement: Weight arguments
`FeatureRanker.fit()`, `rank_features()`, and `rank_resampled()` SHALL accept `sample_weight` (default `None`): a 1-D array-like of finite, non-negative numbers with one value per row and a positive sum; otherwise `ValueError`. Without weights, behaviour and reports SHALL be unchanged. Weights SHALL be copied, like features.

#### Scenario: Invalid weights
- **WHEN** `sample_weight` has the wrong length, a negative value, a NaN, or sums to 0
- **THEN** `ValueError` is raised before any model is fit

#### Scenario: Unweighted unchanged
- **WHEN** no weights are passed
- **THEN** reports equal those of the previous version

### Requirement: Weighted fitting and scoring
With training weights, every model SHALL be fit with them; for built-in pipelines the weights SHALL go to the final estimator. The dummy baseline SHALL be fit with the training weights. With evaluation weights, the baseline score, every permuted score, and the dummy's score SHALL be weighted; the train score SHALL use the training weights when they were given. A model whose final estimator does not accept `sample_weight` SHALL NOT be fit; `failures` SHALL include it with stage `"fit"` and `error_type` `"SampleWeightNotSupported"`.

#### Scenario: Weights change the ranking
- **WHEN** feature `a` drives the target only in rows with weight 1 and feature `b` only in rows with weight 0.001
- **THEN** with weights `a` ranks above `b`, and without weights they are close

#### Scenario: Unsupported model
- **WHEN** built-in classification models are fit with weights
- **THEN** `lda` and `qda` appear in `failures` with `"SampleWeightNotSupported"` and every other built-in model is fit

#### Scenario: Weighted dummy
- **WHEN** weights make class 1 the weighted majority while class 0 is the unweighted majority
- **THEN** the fitted dummy predicts class 1

### Requirement: Weight record
When any weights are used, every report from `rank_features()` SHALL include `sample_weight` with `train` and `eval` booleans. Resampled reports SHALL carry it in each split report. Without weights, no `sample_weight` key SHALL appear. Correlation evidence and the SHAP summary SHALL stay unweighted, and the docs SHALL say so.

#### Scenario: Record
- **WHEN** only training weights are given
- **THEN** `report["sample_weight"] == {"train": true, "eval": false}`

### Requirement: Resampling, selector, and CLI
`rank_resampled(X, y, sample_weight=w)` SHALL pass each split its training and evaluation slices of `w`. `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)` SHALL pass weights to its ranking, and SHALL work inside a `Pipeline` with scikit-learn metadata routing (`set_fit_request(sample_weight=True)`). The CLI SHALL accept `--sample-weight COLUMN`: the column is read from both CSV files as weights and excluded from the features.

#### Scenario: Resampled slices
- **WHEN** `rank_resampled` runs with weights and a recording estimator
- **THEN** each split's model receives exactly the weights of that split's training rows

#### Scenario: Selector in a routed pipeline
- **WHEN** a pipeline with `ConsensusSelector(...).set_fit_request(sample_weight=True)` is fit with `sample_weight=w` under `sklearn.set_config(enable_metadata_routing=True)`
- **THEN** the selector's ranking receives `w`

#### Scenario: CLI
- **WHEN** the CLI runs with `--sample-weight w` on CSV files containing a column `w`
- **THEN** `w` is not a feature and the report includes `sample_weight`
