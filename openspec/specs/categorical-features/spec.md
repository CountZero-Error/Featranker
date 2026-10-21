# categorical-features Specification

## Purpose
Let users rank raw categorical columns as single features, with encoding learned inside each fit so evaluation rows never shape it.
## Requirements
### Requirement: Categorical option
`FeatureRanker` SHALL accept `categorical`: `None` (default), `"auto"`, or a list of feature names. Any other value, or a name that is not a feature, SHALL raise `ValueError` (unknown names at `fit()`). With `None`, behaviour and reports SHALL be unchanged from 1.8, and non-numeric features SHALL raise the existing error. With `"auto"`, DataFrame columns of object, string, or category dtype SHALL be categorical; NumPy input under `"auto"` has no categorical columns. Listed names SHALL be categorical whatever their dtype. A ranker unpickled without `categorical` SHALL behave as `None`.

#### Scenario: Default rejects strings
- **WHEN** a DataFrame with a string column is passed to `FeatureRanker("reg").fit()`
- **THEN** `ValueError` is raised as in 1.7

#### Scenario: Auto detects
- **WHEN** a DataFrame has float column `x` and string column `city`, and `categorical="auto"`
- **THEN** `city` is categorical and `x` is numeric

#### Scenario: Explicit list on NumPy
- **WHEN** a NumPy object array with names `["x", "city"]` is fit with `categorical=["city"]`
- **THEN** `city` is categorical

### Requirement: Fold-local level coding
In each `fit()`, every categorical column's levels SHALL be learned from the training rows only, ordered by the sorted string form of the values, and coded `0..k-1`; a missing value (None or NaN) SHALL be its own level. In `rank_features()` and in shadow blocks, values not seen in training SHALL get one shared "unseen" code. Features passed to models SHALL be the numeric columns as given plus these codes, in the original column order. Each categorical column SHALL remain a single feature (its own group unless `feature_groups` says otherwise).

#### Scenario: Levels from training rows only
- **WHEN** training rows of `city` contain `"a"` and `"b"`, and evaluation rows contain `"c"`
- **THEN** the report shows 2 training levels and 1 unseen evaluation value for `city`, and ranking succeeds

#### Scenario: One feature per raw column
- **WHEN** `city` has 5 levels
- **THEN** the report's `feature_groups` and consensus list `city` once

### Requirement: Built-in model handling
For built-in linear models, the pipeline SHALL one-hot encode categorical code columns (ignoring unknown codes) and standard-scale numeric columns, after median imputation of numeric columns when `missing="impute"`. Built-in tree models SHALL receive the codes unchanged. Shadow columns of categorical columns SHALL be treated as categorical. Custom models SHALL receive the coded matrix, and the fitted ranker SHALL expose `categorical_features_` (the categorical feature names).

#### Scenario: Linear and tree built-ins fit
- **WHEN** `group="fast"` is ranked on data with a categorical column that drives the target
- **THEN** no built-in model fails, and the categorical feature ranks first in the consensus

#### Scenario: Shadows with categoricals
- **WHEN** `shadows=True` and a categorical column is present
- **THEN** ranking succeeds and the column has `above_shadow` flags

### Requirement: Evidence with categorical columns
Categorical columns SHALL be excluded from Spearman correlations in `evidence.correlation` and in `auto_groups()` (they are never paired or grouped). The SHAP summary SHALL explain built-in tree models on the codes; models whose pipeline contains a one-hot step SHALL be listed in `evidence.shap.unsupported`.

#### Scenario: No correlation pairs with categoricals
- **WHEN** a categorical column's codes happen to correlate with a numeric column
- **THEN** no correlated pair involving the categorical column is reported

### Requirement: Categorical record in reports
When categorical columns are present, every report from `rank_features()` SHALL include `categorical` with `columns` (names), `train_levels` (name → number of training levels, missing included), and `unseen_eval` (name → number of evaluation values with the unseen code). Resampled reports SHALL carry it in each split report. Without categorical columns, no `categorical` key SHALL appear.

#### Scenario: Record
- **WHEN** `city` has 3 training levels and 2 unseen evaluation values
- **THEN** `report["categorical"] == {"columns": ["city"], "train_levels": {"city": 3}, "unseen_eval": {"city": 2}}`

### Requirement: Pass-through
`rank_resampled()` SHALL learn levels inside each split's `fit()`. `ConsensusSelector` SHALL accept `categorical` (default `None`), store it unchanged, pass it to its ranker, and select raw columns; a selector unpickled without it SHALL behave as `None`. The CLI SHALL accept `--categorical` with `auto` or a comma-separated list of column names.

#### Scenario: Selector on a mixed DataFrame
- **WHEN** `ConsensusSelector("reg", "r2", k=1, categorical="auto")` is fit on a DataFrame where string column `city` drives the target
- **THEN** `selected_groups_ == ["city"]` and `transform` returns the raw `city` column

#### Scenario: CLI
- **WHEN** the CLI runs with `--categorical auto` on CSV files with a text column
- **THEN** the output report includes `categorical`

