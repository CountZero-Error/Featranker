## 1. Construction and validation

- [x] 1.1 Write tests:
  - parameters are stored raw, and `clone`/`get_params`/`set_params` round-trip;
  - `fit` raises `ValueError` before ranking when no rule is set, `k` is not a positive int, or `min_shadow_share` is outside [0, 1] or not a number.

  Confirm they fail.
- [x] 1.2 Create `featranker/selection.py` with `ConsensusSelector(BaseEstimator, SelectorMixin)`, its `__init__`, and `_validate_rule`. Export it from `featranker`.

## 2. Fit and selection

- [x] 2.1 Write tests:
  - `report_` is a resampled report built from exactly `X`/`y`;
  - shadows are on only when `min_shadow_share` is set;
  - the `k`-only, shadow-only, and both rules work, using a patched `rank_resampled` that returns a synthetic stability table;
  - `selected_groups_` is in stability order;
  - empty selection warns and gives an all-False support.

  Confirm they fail.
- [x] 2.2 Implement `fit` (names, `rank_resampled` call, selection) and `_get_support_mask`.

## 3. Columns and names

- [x] 3.1 Write tests:
  - a group selects all its columns;
  - `transform` keeps the original column order;
  - array input gets `x0..` names and works with `feature_groups` that use them;
  - DataFrame input sets `feature_names_in_`, and `get_feature_names_out()` returns the selected names.

  Confirm they fail.
- [x] 3.2 Implement names, the support mask mapping, and `n_features_in_`/`feature_names_in_`.

## 4. scikit-learn composition

- [x] 4.1 Write tests:
  - `cross_val_score` on a `Pipeline` gives finite scores;
  - a recording estimator inside the selector sees only outer-training rows;
  - `GridSearchCV` over `select__k` runs and picks a `k`;
  - a real-data end-to-end run (diabetes, `k=3`, `group="tree"`, `cv=3`).

  Confirm they fail where applicable.
- [x] 4.2 Fix any composition issues found.

## 5. Docs

- [x] 5.1 `docs/usage.md`: add a "Feature selection in a pipeline" section with:
  - a nested-CV example and its real output;
  - the parameter table;
  - the selection rules;
  - the cost formula;
  - the note that `report_` reflects one training set;
  - the empty-selection behavior.

  Mention it in the API intro.
- [x] 5.2 `README.md`: one line in "How it works" plus the docs-table entry. `docs/README.md` index.
- [x] 5.3 Verify the doc claims and the example byte-for-byte against a real run, then render through the GitHub API and check anchors and escaped tags.

## 6. Release prep

- [x] 6.1 Bump to 0.8.0. Add a CHANGELOG entry.
- [x] 6.2 Run the full suite outside the sandbox.
