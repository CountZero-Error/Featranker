# consensus-selector Specification

## Purpose
Turn FeatRanker's resampled, shadow-aware ranking into a scikit-learn feature selector, so features are chosen inside `fit()` on training data only and can be cross-validated safely inside a `Pipeline`.
## Requirements
### Requirement: Selector construction follows scikit-learn conventions
`ConsensusSelector(task, scoring, *, k=None, min_shadow_share=None, max_shadow_q=None, group="all", models=None, feature_groups=None, cv=5, n_repeats=10, random_state=None, shadow_seed=None, n_jobs=None, verbose=False)` SHALL store every argument unchanged as an attribute of the same name and SHALL NOT validate or fit anything in `__init__`. `clone`, `get_params`, and `set_params` SHALL round-trip all parameters.

#### Scenario: Clone keeps parameters
- **WHEN** `clone(ConsensusSelector("reg", "r2", k=3, cv=4, max_shadow_q=0.05))` is called
- **THEN** the clone's `get_params()` equals the original's

### Requirement: Parameter validation in fit
`fit()` SHALL raise `ValueError`, before any ranking, in each of these cases:
- none of `k`, `min_shadow_share`, and `max_shadow_q` is set;
- `k` is not a positive int;
- `min_shadow_share` is not a number in `[0, 1]`;
- `max_shadow_q` is not a number in `(0, 1]`.

Other parameters SHALL be validated as `FeatureRanker` and `rank_resampled` do.

#### Scenario: No rule
- **WHEN** `ConsensusSelector("reg", "r2").fit(X, y)` is called
- **THEN** `ValueError` is raised and no model is fit

#### Scenario: Bad values
- **WHEN** `k=0`, `min_shadow_share=1.5`, or `max_shadow_q=0` is given
- **THEN** `fit()` raises `ValueError`

### Requirement: Ranking uses only the data passed to fit
`fit(X, y, groups=None)` SHALL obtain its ranking from `FeatureRanker.rank_resampled` on exactly `X` and `y`, passing `groups` through to the splitter. `k` MAY be any integral type (Python or NumPy) and SHALL be passed to `rank_resampled` as a Python `int`. The ranker SHALL be configured from the selector's parameters, with `shadows=True` exactly when `min_shadow_share` or `max_shadow_q` is set. `fit()` SHALL store that report as `report_`.

#### Scenario: Group-aware inner cv
- **WHEN** `cv=GroupKFold(2)` and `fit(X, y, groups=g)` is called, directly or through a `Pipeline` as `select__groups`
- **THEN** ranking succeeds with those row groups

#### Scenario: NumPy integer k
- **WHEN** `k=np.int64(1)`, or a `GridSearchCV` grid of `np.arange(1, 3)`, is used
- **THEN** fitting succeeds

#### Scenario: Training rows only
- **WHEN** the selector is fit inside `cross_val_score` with a recording estimator as its only model
- **THEN** every row that estimator sees during the selector's fit belongs to the outer training fold

#### Scenario: q rule turns shadows on
- **WHEN** only `max_shadow_q` is set
- **THEN** `report_` has `shadow_test` and its stability rows have `shadow_q_value`

### Requirement: Selection rules
Groups are considered in `report_["stability"]` order.
- With `min_shadow_share`, only groups whose stability `above_shadow_share >= min_shadow_share` SHALL pass.
- With `max_shadow_q`, only groups whose stability `shadow_q_value <= max_shadow_q` SHALL pass.
- With `k`, the first `k` of the groups that pass every set filter SHALL be selected, or all of them if there are fewer. Without `k`, every group that passes SHALL be selected.

`selected_groups_` SHALL list the selected group names in stability order.

#### Scenario: Top k
- **WHEN** `k=2` and the stability order is `[a, b, c]`
- **THEN** `selected_groups_ == ["a", "b"]`

#### Scenario: Shadow filter then k
- **WHEN** `min_shadow_share=0.5`, `k=1`, the order is `[a, b, c]`, and the shares are `0.4`, `0.9`, and `0.8`
- **THEN** `selected_groups_ == ["b"]`

#### Scenario: q filter
- **WHEN** `max_shadow_q=0.05`, the order is `[a, b, c]`, and the q-values are `0.01`, `0.2`, and `0.04`
- **THEN** `selected_groups_ == ["a", "c"]`

### Requirement: Column support and names
`get_support()` SHALL mark every column belonging to a selected group, and only those columns. `transform(X)` SHALL return those columns in their original order.

Feature names:
- For DataFrame input, the column names SHALL be used, and `feature_names_in_` SHALL be set.
- For array input, the names SHALL be `x0, x1, ...`. In that case `feature_groups` refers to those names.

`get_feature_names_out()` SHALL return the selected column names.

#### Scenario: Group keeps all its columns
- **WHEN** `feature_groups={"seg": ["a", "b"]}`, the data has columns `a`, `b`, and `c`, and only `seg` is selected
- **THEN** `get_support()` is `[True, True, False]` and `transform` returns columns `a` and `b`

#### Scenario: DataFrame names out
- **WHEN** a DataFrame with columns `p`, `q`, and `r` is fit and `q` is the only selected group
- **THEN** `get_feature_names_out()` returns `["q"]`

### Requirement: Fitted state is replaced atomically
`fit()` SHALL remove `support_`, `selected_groups_`, `report_`, `n_features_in_`, and `feature_names_in_` before doing any work, and SHALL set them only after ranking and selection succeed. After a failed `fit()`, `get_support()` and `transform()` SHALL raise `NotFittedError`.

#### Scenario: Failed refit
- **WHEN** a fitted selector's `fit()` raises
- **THEN** none of the fitted attributes exist and `get_support()` raises `NotFittedError`

### Requirement: DataFrame labels are strings
`fit()` and `transform()` SHALL compare DataFrame column labels as strings, using a relabeled copy, and SHALL NOT modify the caller's DataFrame. Labels that collide after conversion SHALL raise `ValueError`.

#### Scenario: Mixed labels
- **WHEN** a DataFrame with columns `[0, "b", "c"]` is passed to `fit_transform`
- **THEN** it succeeds, and the caller's columns are unchanged

### Requirement: Target is required
The selector's scikit-learn tags SHALL declare that `y` is required (`requires_y`, or `target_tags.required` on scikit-learn 1.6 and later).

#### Scenario: Tags
- **WHEN** the selector's tags are read
- **THEN** they report that a target is required

### Requirement: Empty selection
If no group is selected, `fit()` SHALL emit a `UserWarning` and succeed, and `get_support()` SHALL be all `False`.

#### Scenario: Nothing beats the threshold
- **WHEN** `min_shadow_share=1.0` and no group has a share of 1.0
- **THEN** a `UserWarning` is emitted and `selected_groups_ == []`

### Requirement: Pipeline composition
The selector SHALL work as a step in `sklearn.pipeline.Pipeline` under `cross_val_score` and `GridSearchCV`, including tuning `k`.

#### Scenario: Nested CV
- **WHEN** `cross_val_score(Pipeline([("select", ConsensusSelector(...)), ("model", Ridge())]), X, y, cv=3)` is run
- **THEN** it returns three finite scores

### Requirement: Older pickles stay usable
A `ConsensusSelector` unpickled from a version without `max_shadow_q` SHALL behave as `max_shadow_q=None`: `transform()`, `get_params()`, `clone()`, and `fit()` SHALL work, and a refit SHALL NOT fail on the missing attribute.

#### Scenario: Pickle from 1.3
- **WHEN** a fitted selector whose state lacks `max_shadow_q` is unpickled
- **THEN** `transform(X)` returns the same columns as before pickling, `get_params()["max_shadow_q"] is None`, and `clone(selector).fit(X, y)` succeeds

