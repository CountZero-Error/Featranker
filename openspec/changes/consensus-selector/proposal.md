## Why

FeatRanker can now say which feature groups matter (0.5), how stable that is (0.6), and whether they beat noise (0.7). The roadmap thesis ends with "a leakage-safe way to use the answer". Today users must pick features by hand from a report. The common mistake is to rank on all the data and then cross-validate a model on the chosen features, which overstates performance. A scikit-learn selector that ranks inside `fit()` makes the safe path the easy one: put it in a `Pipeline`, and `cross_val_score` or `GridSearchCV` reruns the ranking on each training fold automatically.

## What Changes

- New `featranker.ConsensusSelector`, a scikit-learn feature selector (`BaseEstimator` + `SelectorMixin`). `fit(X, y)` runs `FeatureRanker.rank_resampled` on the data it is given, then keeps the feature groups that pass a selection rule. `transform` keeps every column of each kept group.
- The selection rule is set by `k` and/or `min_shadow_share`:
  - `k`: keep the first `k` groups in the stability order.
  - `min_shadow_share`: keep groups whose stability `above_shadow_share` is at least this value. This turns shadows on automatically.
  - With both, the `min_shadow_share` filter applies first, then the first `k` of the groups that pass.
  - At least one must be set.
- Parameters mirror `FeatureRanker` and `rank_resampled`: `task`, `scoring`, `group`, `models`, `feature_groups`, `cv`, `n_repeats`, `random_state`, `shadow_seed`, `n_jobs`, `verbose`. Following scikit-learn convention, they are stored unchanged and validated in `fit()`.
- Fitted attributes:
  - `report_`, the resampled report;
  - `selected_groups_`;
  - `n_features_in_`;
  - `feature_names_in_`, for DataFrame input;
  - support masks through the standard `get_support()` and `get_feature_names_out()`.
- If nothing is selected, a `UserWarning` is emitted and `transform` returns zero columns, matching scikit-learn's own selectors.

**Report schema:** unchanged. `report_` is a 0.6/0.7 resampled report.

## Non-goals

- Passing every scikit-learn `check_estimator` check. FeatRanker requires named, finite, numeric input with at least two folds' worth of rows. The selector supports `clone`, `get_params`/`set_params`, `Pipeline`, `cross_val_score`, and `GridSearchCV`.
- A model score or final evaluation inside the selector.
- Group selection rules beyond `k` and `min_shadow_share`, such as `top_k_share` thresholds or percentiles.
- A CLI.
- Caching rankings across `GridSearchCV` candidates.

## Capabilities

### New Capabilities

- `consensus-selector`: covers
  - the selector's parameters and validation;
  - the selection rules;
  - group-to-column support;
  - feature names;
  - fitted attributes;
  - empty selection;
  - scikit-learn composition, including nested-CV safety by construction.

### Modified Capabilities

(none)

## Impact

- **Code:** a new `featranker/selection.py`, and an export in `featranker/__init__.py`. No change to `FeatureRanker`.
- **Cost:** each `fit()` is one `rank_resampled` run, so `cv` rankings. Inside an outer cross-validation with `n` folds, that is `n * cv` rankings; doubled with shadows.
- **Docs:**
  - `docs/usage.md`: a new "Feature selection in a pipeline" section, with a nested-CV example and the cost;
  - `README.md`: a line in "How it works" and a docs-table entry;
  - `docs/README.md`.
- **Tests:** a new `tests/test_selection.py`.
- **Version:** 0.8.0.
