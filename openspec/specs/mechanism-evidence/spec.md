# mechanism-evidence Specification

## Purpose
Add optional evidence to a report that helps explain why the ranking looks the way it does: how much each model overfits, which feature groups are correlated, and how each group pushes each model's predictions.
## Requirements
### Requirement: Evidence flag
`FeatureRanker` SHALL accept a keyword `evidence` (default `False`). A non-bool value SHALL raise `TypeError`. With `evidence=False`, reports SHALL contain none of the keys this capability adds, so they equal reports from 1.0 for the same inputs. A ranker unpickled from 1.0 SHALL behave as `evidence=False`.

#### Scenario: Default unchanged
- **WHEN** a report is made with the default settings
- **THEN** it has no top-level `evidence` key and no model report has `train_score`, `overfit_gap`, or `shap`

#### Scenario: Invalid flag
- **WHEN** `FeatureRanker("reg", evidence="yes")` is called
- **THEN** a `TypeError` is raised

### Requirement: Train score and overfit gap
With `evidence=True`, every model report from `rank_features()` SHALL include `train_score`, the same scorer applied to that model on the data it was fit on, and `overfit_gap = train_score - evaluation_score`. If scoring on the training data fails, both SHALL be `null`, a failure with stage `"evidence"` SHALL be added to `failures`, and the ranking SHALL otherwise be unaffected.

#### Scenario: Overfitting tree
- **WHEN** an unpruned decision tree is ranked with `evidence=True` and `scoring="r2"` on noisy data
- **THEN** its `train_score` is 1.0 and its `overfit_gap` is greater than 0

### Requirement: Correlated groups
With `evidence=True`, the report SHALL include `evidence.correlation` with `method` `"spearman"`, `data` `"train"`, `threshold` `0.5`, and `pairs`: every pair of feature groups whose strongest member-column Spearman correlation on the training data has absolute value at least 0.5. Each pair SHALL be `{"groups": [a, b], "rho": r}`, where `r` is that strongest correlation with its sign. Pairs SHALL be sorted by `|rho|` descending, then by group names. Shadow columns SHALL be excluded. A pair involving a constant column SHALL NOT be listed.

#### Scenario: Planted copy
- **WHEN** feature `f` is feature `e` plus small noise and both are their own groups
- **THEN** `evidence.correlation.pairs` contains `{"groups": ["e", "f"], ...}` with `rho` above 0.9

#### Scenario: Independent features
- **WHEN** all features are independent draws
- **THEN** `evidence.correlation.pairs` is empty

### Requirement: SHAP summary
With `evidence=True` and the `shap` package importable, the report SHALL include `evidence.shap` with `available: true`, `n_rows` (the number of evaluation rows explained), and `unsupported` (a list of `{"model", "reason"}`). For each model whose final estimator is linear (has coefficients) or is a tree model that SHAP's tree explainer supports, and, for a `Pipeline`, whose earlier steps are all per-column scalers (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`) or identity steps (`"passthrough"`, `None`), its model report SHALL include `shap`: `explainer` (`"linear"` or `"tree"`) and `groups`, mapping every feature group to `mean_abs` and `direction`.
- Rows: at most 1000 evaluation rows, chosen evenly spaced and independent of `random_state`.
- A group's SHAP value per row SHALL be the sum over its member columns; `mean_abs` is the mean absolute value of that sum. Shadow columns SHALL NOT be reported.
- `direction` SHALL be the Spearman correlation between a single-column group's feature value and its SHAP value, and `null` for multi-column groups or when undefined.
- Classification: binary models SHALL use the positive-class output; multiclass models SHALL average `mean_abs` over classes and report `direction` as `null`.
- Any other model, or any SHAP error for a model, SHALL add that model to `unsupported` and SHALL NOT fail the ranking.

Without `shap`, `evidence.shap` SHALL be `{"available": false}`, a `UserWarning` SHALL say to install `featranker[shap]`, and the rest of the evidence SHALL still be computed.

#### Scenario: Tree and linear models
- **WHEN** a ridge model and a random forest are ranked with `evidence=True` and `shap` installed
- **THEN** both model reports have `shap.groups` for every feature group, with `explainer` `"linear"` and `"tree"` respectively

#### Scenario: Direction of a planted effect
- **WHEN** the target rises with feature `a` and `a` is its own group
- **THEN** `shap.groups["a"].direction` is above 0.9 for the ridge model

#### Scenario: Unsupported model
- **WHEN** a k-nearest-neighbours model is ranked with `evidence=True`
- **THEN** it has no `shap` key and appears in `evidence.shap.unsupported`

#### Scenario: Pipeline with a non-scaler step
- **WHEN** `make_pipeline(PCA(), Ridge())` is ranked with `evidence=True`
- **THEN** it has no `shap` key and appears in `evidence.shap.unsupported`

#### Scenario: Identity pipeline step
- **WHEN** `Pipeline([("skip", "passthrough"), ("ridge", Ridge())])` is ranked with `evidence=True`
- **THEN** its model report has `shap` with `explainer` `"linear"`

#### Scenario: shap missing
- **WHEN** `shap` cannot be imported and `evidence=True`
- **THEN** a `UserWarning` is raised, `evidence.shap == {"available": false}`, and `train_score` and `evidence.correlation` are still present

### Requirement: JSON safety
Every evidence value SHALL be JSON-serializable with `allow_nan=False`.

#### Scenario: Serialize
- **WHEN** a report made with `evidence=True` is passed to `json.dumps(report, allow_nan=False)`
- **THEN** no error is raised

### Requirement: Resampled and CLI pass-through
`rank_resampled()` SHALL create its per-split rankers with the same `evidence` setting, so each `splits[i].report` carries its own evidence; the resampled report SHALL NOT aggregate evidence across splits. The CLI SHALL accept `--evidence`, which sets `evidence=True`.

#### Scenario: Resampled
- **WHEN** `rank_resampled()` runs on a ranker with `evidence=True`
- **THEN** every `splits[i].report` has an `evidence` key and the top-level report does not

#### Scenario: CLI
- **WHEN** the CLI runs with `--evidence --output out.json`
- **THEN** `out.json` has an `evidence` key

### Requirement: Input ownership
`FeatureRanker.fit()` and `rank_features()` SHALL work on their own copies of the features and target they receive. Fitted models and any training data the ranker keeps SHALL NOT share memory with the caller's arrays or DataFrame, so modifying them after a call SHALL NOT change later reports (importances, scores, and evidence).

#### Scenario: Caller mutates training arrays
- **WHEN** a ranker is fit with `evidence=True` on NumPy arrays `X` and `y`, and then `X[:] = 0` and `y[:] = 0` are executed before `rank_features()`
- **THEN** the report equals that of a ranker fit on unmodified copies

#### Scenario: Caller mutates a DataFrame
- **WHEN** the same is done with a float DataFrame
- **THEN** the report is likewise unchanged

#### Scenario: Custom model that stores its training data
- **WHEN** a ranker with `models={"knn": KNeighborsRegressor()}` is fit, the caller then sets `X[:] = 0` and `y[:] = 0`, and `rank_features()` runs on separate evaluation data
- **THEN** the KNN model's `evaluation_score`, importances, and `train_score` equal those of a ranker fit on unmodified copies

