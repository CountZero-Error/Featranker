# named-model-input Specification

## Purpose
Let custom models select and preprocess columns by name, and keep shadow evidence valid when a custom pipeline drops the shadow columns.
## Requirements
### Requirement: Model input option
`FeatureRanker` SHALL accept `model_input` with values `"array"` (default) and `"dataframe"`; any other value SHALL raise `ValueError` at construction. With `"dataframe"`, `pandas` SHALL be importable, else `ImportError` naming `pandas` at construction. With `"array"`, behaviour and reports SHALL be unchanged. A ranker unpickled without `model_input` SHALL behave as `"array"`.

#### Scenario: Default unchanged
- **WHEN** a ranker is built without `model_input`
- **THEN** reports equal those of the previous version for the same inputs

#### Scenario: Invalid value
- **WHEN** `FeatureRanker("reg", model_input="frame")` is called
- **THEN** `ValueError` is raised

### Requirement: Named columns for every model
With `model_input="dataframe"`, every model SHALL be fit, scored, and permuted on a pandas DataFrame whose columns are the feature names in order, followed (with shadows on) by `shadow__<name>` for each shadow column. Importances, scores, train scores, and the SHAP summary SHALL equal those of `"array"` mode for models whose predictions do not depend on column names. Reports SHALL record `model_input: "dataframe"`.

#### Scenario: Select by name
- **WHEN** a custom model is `make_pipeline(ColumnTransformer([("num", StandardScaler(), ["a", "b"])], remainder="passthrough"), Ridge())` and `model_input="dataframe"`
- **THEN** ranking succeeds and the model's importances cover every feature group

#### Scenario: Same numbers as array mode
- **WHEN** the same built-in models rank the same data in both modes with fixed seeds
- **THEN** the reports' importances and consensus are equal

### Requirement: Shadow compatibility check
With shadows on, after fitting, a model SHALL be shadow-incompatible if it is or contains (inside a `Pipeline`) a fitted `ColumnTransformer` whose transformers and `remainder` together do not use every shadow column. Such a model SHALL be left out of every shadow comparison in that report (no `above_shadow` flags, no `shadow_max`, not counted in `family_above_shadow` or `above_shadow_share`), and `failures` SHALL include `{"model": <name>, "stage": "shadows", "error_type": "ShadowColumnsDropped", ...}`. Its importances and ranking vote SHALL be unaffected. This check SHALL apply in both input modes.

#### Scenario: Pipeline drops shadows
- **WHEN** shadows are on and a custom model's `ColumnTransformer` lists only the real columns with `remainder="drop"`
- **THEN** the model has no `above_shadow` flags, a `"shadows"` failure names it, and it still appears in the consensus

#### Scenario: Pipeline keeps shadows
- **WHEN** the same `ColumnTransformer` uses `remainder="passthrough"`
- **THEN** no `"shadows"` failure is recorded

### Requirement: Pass-through
`rank_resampled()` SHALL use the ranker's `model_input` in every split. `ConsensusSelector` SHALL accept `model_input` (default `"array"`), store it unchanged, and pass it to its ranker; a selector unpickled without it SHALL behave as `"array"`.

#### Scenario: Selector
- **WHEN** `ConsensusSelector("reg", "r2", k=2, models={"m": named_pipeline}, model_input="dataframe")` is fit on a DataFrame
- **THEN** fitting succeeds and `selected_groups_` has two groups

