## MODIFIED Requirements

### Requirement: Extras match the registry
Every optional-dependency extra SHALL install something FeatRanker uses. The `lightgbm` extra SHALL be removed. `all` SHALL install `xgboost` and `catboost`, a `plot` extra SHALL install `matplotlib`, and a `shap` extra SHALL install `shap`.

#### Scenario: Extras
- **WHEN** `pyproject.toml` is read
- **THEN** the extras are `xgboost`, `catboost`, `all` (xgboost and catboost), `plot` (matplotlib), `shap` (shap), and `test`
