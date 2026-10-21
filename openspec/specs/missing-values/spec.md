# missing-values Specification

## Purpose
Let users rank features on data with missing values without imputing by hand, while keeping imputation inside each training split so evaluation rows never shape it.
## Requirements
### Requirement: Missing-value policy
`FeatureRanker` SHALL accept `missing` with values `"error"` (default) and `"impute"`; any other value SHALL raise `ValueError` at construction. With `"error"`, NaN in features SHALL raise the existing error and reports SHALL be unchanged. With `"impute"`, NaN (including pandas `NA` in nullable dtypes) SHALL be accepted in features passed to `fit()`, `rank_features()`, and `rank_resampled()`; infinite values SHALL still raise `ValueError`. A ranker unpickled without `missing` SHALL behave as `"error"`.

#### Scenario: Default rejects NaN
- **WHEN** `FeatureRanker("reg").fit(X_with_nan, y)` is called
- **THEN** `ValueError` mentioning finite values is raised

#### Scenario: Impute accepts NaN
- **WHEN** `FeatureRanker("reg", missing="impute").fit(X_with_nan, y)` and then `rank_features(X_eval_with_nan, y_eval, scoring="r2")` are called
- **THEN** a report is returned and no built-in model fails because of NaN

#### Scenario: Infinite still rejected
- **WHEN** features contain `inf` with `missing="impute"`
- **THEN** `ValueError` is raised

### Requirement: Fold-local imputation for built-in models
With `missing="impute"`, every built-in model SHALL be a pipeline whose first step is `SimpleImputer(strategy="median")`, followed by the existing steps. The imputer SHALL be fit only on the rows the model is fit on. Custom `models=` SHALL receive features with NaN unchanged. The dummy baseline SHALL be unaffected.

#### Scenario: Imputer learns from training rows only
- **WHEN** a built-in model is fit with `missing="impute"` on training rows whose column median is 1.0 and evaluation rows whose median is 100.0
- **THEN** the fitted imputer's statistic for that column is 1.0

#### Scenario: Custom models see NaN
- **WHEN** a recording custom estimator is fit with `missing="impute"`
- **THEN** it receives the NaN values unchanged

### Requirement: Missing-value record in reports
With `missing="impute"`, every report from `rank_features()` SHALL include `missing` with `policy` `"impute"`, `strategy` `"median"`, and `train_counts` (feature name → number of missing training values, every feature listed). Resampled reports SHALL carry it in each split report. With `"error"`, no `missing` key SHALL appear.

#### Scenario: Counts
- **WHEN** feature `a` has 3 missing training values and `b` none
- **THEN** `report["missing"]["train_counts"] == {"a": 3, "b": 0}`

### Requirement: Evidence and auto groups with missing values
With NaN present, Spearman correlations used by `evidence.correlation` and by `auto_groups()` SHALL be computed for each column pair on the rows where both values are present; a pair with fewer than 30 such rows SHALL be treated as uncorrelated, so sparse columns that rarely overlap are not grouped by chance. The SHAP summary SHALL treat a leading `SimpleImputer` pipeline step as a per-column step. `auto_groups()` SHALL accept NaN without a policy argument.

#### Scenario: Too few shared rows
- **WHEN** two independent columns are present together in only 10 rows
- **THEN** they are not listed as a correlated pair and `auto_groups(X)` does not group them

#### Scenario: Pairwise-complete correlation
- **WHEN** `f` is a near-copy of `e` and 10% of `e`'s values are NaN
- **THEN** `evidence.correlation.pairs` lists `e` and `f` with `rho` above 0.9, and `auto_groups(X)` groups them

### Requirement: Pass-through
`rank_resampled()` SHALL use the ranker's `missing` policy on every split. `ConsensusSelector` SHALL accept `missing` (default `"error"`), store it unchanged, and pass it to its ranker; a selector unpickled without it SHALL behave as `"error"`. The CLI SHALL accept `--missing {error,impute}` (default `error`).

#### Scenario: Selector in a pipeline
- **WHEN** `make_pipeline(ConsensusSelector("reg", "r2", k=2, missing="impute"), SimpleImputer(), Ridge())` is cross-validated on data with NaN
- **THEN** it runs without error

#### Scenario: CLI
- **WHEN** the CLI runs with `--missing impute` on CSV files with empty cells
- **THEN** the output report includes `missing`

