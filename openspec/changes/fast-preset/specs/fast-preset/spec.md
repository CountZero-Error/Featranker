## Purpose

Offer a cheaper built-in model set that keeps one representative per model family, so that each family still casts its consensus vote at a fraction of the compute of the full registry.

## ADDED Requirements

### Requirement: Fast group
`FeatureRanker`, `ConsensusSelector`, and the CLI `--group` option SHALL accept `"fast"` in addition to `"linear"`, `"tree"`, and `"all"`. Any other value SHALL still raise `ValueError` (CLI: an argument error). Combining `group="fast"` with `models=` SHALL raise `ValueError`, as for the other non-default groups.

#### Scenario: Accepted
- **WHEN** `FeatureRanker("reg", group="fast")` is created
- **THEN** no error is raised and `ranker.group == "fast"`

#### Scenario: With custom models
- **WHEN** `FeatureRanker("reg", group="fast", models={"m": Ridge()})` is created
- **THEN** a `ValueError` is raised

### Requirement: One representative per family
With `group="fast"`, the ranker SHALL fit exactly the built-in models marked as fast representatives for the task, drawn from both the linear and tree groups:
- classification: `logistic_regression`, `ridge_classifier`, `lda`, `naive_bayes_gaussian`, `decision_tree`, `random_forest`, `hist_gradient_boosting`;
- regression: `ridge_regression`, `svr_regression`, `decision_tree_regressor`, `random_forest_regressor`, `hist_gradient_boosting_regressor`.

Every family present in the task's full registry SHALL have exactly one representative, so the set of consensus families is the same as with `group="all"`. No representative SHALL need an optional dependency. Built-in linear representatives SHALL keep the standard-scaler pipeline used for all built-in linear models.

#### Scenario: Classification set
- **WHEN** `FeatureRanker("clf", group="fast").fit(X, y)` runs
- **THEN** `trained_models` has exactly the seven classification representatives

#### Scenario: Same families as all
- **WHEN** the families of the fast models and of all built-in models for a task are compared
- **THEN** they are the same set, and each family appears once among the fast models

#### Scenario: No optional dependency
- **WHEN** `xgboost` and `catboost` are not installed and `group="fast"`
- **THEN** no initialization failure or warning is produced

### Requirement: Pass-through
`rank_resampled()` SHALL use the fast set on every split when the ranker has `group="fast"`, and `ConsensusSelector(group="fast")` SHALL rank with the fast set. The CLI `--group fast` SHALL produce a report whose `models` are the fast set.

#### Scenario: Selector
- **WHEN** `ConsensusSelector("reg", "r2", k=2, group="fast").fit(X, y)` runs
- **THEN** every split report in `report_` lists exactly the five regression representatives
