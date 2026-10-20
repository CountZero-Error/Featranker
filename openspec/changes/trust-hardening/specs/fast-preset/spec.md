## MODIFIED Requirements

### Requirement: One representative per family
With `group="fast"`, the ranker SHALL fit exactly the built-in models marked as fast representatives for the task, drawn from both the linear and tree groups:
- classification: `logistic_regression`, `ridge_classifier`, `lda`, `naive_bayes_gaussian`, `decision_tree`, `random_forest`, `hist_gradient_boosting`;
- regression: `ridge_regression`, `svr_regression`, `decision_tree_regressor`, `random_forest_regressor`, `hist_gradient_boosting_regressor`.

Every family present in the task's full registry SHALL have exactly one representative, so the same candidate families are represented as with `group="all"`; which families actually vote can differ, because initialization, fitting, scoring, and dummy gating apply to each model separately. No representative SHALL need an optional dependency. Built-in linear representatives SHALL keep the standard-scaler pipeline used for all built-in linear models.

#### Scenario: Classification set
- **WHEN** `FeatureRanker("clf", group="fast").fit(X, y)` runs
- **THEN** `trained_models` has exactly the seven classification representatives

#### Scenario: Same candidate families as all
- **WHEN** the families of the fast models and of all built-in models for a task are compared
- **THEN** they are the same set, and each family appears once among the fast models

#### Scenario: No optional dependency
- **WHEN** `xgboost` and `catboost` are not installed and `group="fast"`
- **THEN** no initialization failure or warning is produced
