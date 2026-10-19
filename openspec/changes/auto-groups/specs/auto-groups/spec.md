## Purpose

Propose feature groups from correlation so that permutation importance is not split between near-duplicate features, without using the target and without letting evaluation data shape the groups inside a pipeline.

## ADDED Requirements

### Requirement: auto_groups function
`featranker.auto_groups(X, *, threshold=0.7, feature_names=None)` SHALL return a dict mapping group names to lists of column names, for use as `feature_groups`.
- `X` SHALL be validated like `FeatureRanker.fit()` input (2-D, finite, at least two rows); column names come from a DataFrame, else from `feature_names`, else `x0, x1, ...`.
- `threshold` SHALL be a number with `0 < threshold <= 1`, else `ValueError`.
- Columns SHALL be clustered by complete-linkage hierarchical clustering on the distance `1 - |rho|`, where `rho` is the Spearman correlation between columns; a column that is constant SHALL be treated as uncorrelated with every other column. Clusters SHALL be cut at distance `1 - threshold`, so every pair of columns in a group has `|rho| >= threshold`, up to a floating-point tolerance of `1e-12`; a constant column SHALL never be grouped, whatever the threshold.
- Only clusters with two or more columns SHALL be returned. Members SHALL be in column order. The group name SHALL be the member names joined by `+`. Groups SHALL be ordered by the position of their first member.
- If a generated group name equals an existing column name or another generated group name, `ValueError` SHALL be raised naming it.
- The function SHALL NOT use a target and SHALL NOT modify `X`.

#### Scenario: Near-copy grouped
- **WHEN** column `f` is column `e` plus small noise and the other columns are independent
- **THEN** `auto_groups(X)` returns `{"e+f": ["e", "f"]}`

#### Scenario: Independent columns
- **WHEN** all columns are independent draws
- **THEN** `auto_groups(X)` returns `{}`

#### Scenario: Complete linkage
- **WHEN** `a`–`b` and `b`–`c` have `|rho|` 0.9 but `a`–`c` has `|rho|` 0.5, with `threshold=0.7`
- **THEN** `a` and `c` are not in the same group

#### Scenario: Usable as feature_groups
- **WHEN** the result is passed to `rank_features(..., feature_groups=groups)`
- **THEN** the report's `feature_groups` contains each returned group

#### Scenario: Exact copies at threshold 1
- **WHEN** column `b` is an exact copy (or a strictly monotone transform) of column `a` and `threshold=1.0`
- **THEN** `auto_groups(X, threshold=1.0)` returns a group containing `a` and `b`

#### Scenario: Bad threshold
- **WHEN** `auto_groups(X, threshold=0)` is called
- **THEN** a `ValueError` is raised

### Requirement: Selector auto groups
`ConsensusSelector` SHALL accept `feature_groups="auto"`. In `fit()`, it SHALL compute the groups with `auto_groups` (default threshold) on the `X` passed to `fit()` only, and rank with them. `report_["feature_groups"]` SHALL show the groups used. Any other string SHALL raise `ValueError` in `fit()`. `get_params()` SHALL return `"auto"` unchanged.

#### Scenario: Pipeline fold
- **WHEN** a pipeline with `ConsensusSelector(feature_groups="auto")` is fit on a training fold that contains a near-copy pair
- **THEN** the groups are computed from that fold's rows only and `report_["feature_groups"]` contains the pair as one group

### Requirement: CLI auto groups
The CLI `--feature-groups` option SHALL accept the literal value `auto`, which computes the groups with `auto_groups` (default threshold) on the training CSV's feature columns. Any other value SHALL still be read as a YAML file path.

#### Scenario: CLI
- **WHEN** the CLI runs with `--feature-groups auto` on training data containing a near-copy pair
- **THEN** the output report's `feature_groups` contains that pair as one group
