## Purpose

Shuffle features in a way that respects grouped or time-ordered rows, for both permutation importance and shadow columns.

## ADDED Requirements

### Requirement: Shuffle option
`FeatureRanker(shuffle=...)` SHALL accept `"rows"` (default), `"within_groups"`, or `"circular"`; any other value SHALL raise `ValueError` at construction. With `"rows"`, behaviour and reports SHALL be unchanged. Reports SHALL include `shuffle` when it is not `"rows"`. Old pickles SHALL load with `"rows"`.

#### Scenario: Default unchanged
- **WHEN** `shuffle` is not given
- **THEN** reports equal those of the previous version

### Requirement: Within-group shuffles
With `"within_groups"`, `fit(X, y, shuffle_groups=g)` and `rank_features(X, y, shuffle_groups=g)` SHALL require labels with one entry per row (otherwise `ValueError`), and every shadow column and every permuted column SHALL move values only among rows with the same label. Rows in a group of one SHALL keep their values. When every group has one row, `ValueError` SHALL be raised. `rank_resampled(..., groups=g)` SHALL use `g` sliced per split; without `groups`, `ValueError`. Passing `shuffle_groups` with another mode SHALL raise `ValueError`.

#### Scenario: Values stay in their group
- **WHEN** rows have labels `[0, 0, 0, 1, 1, 1]` and a recording estimator sees a permuted column
- **THEN** the multiset of values in rows with label 0 is unchanged

#### Scenario: Missing labels
- **WHEN** `shuffle="within_groups"` and `fit()` is called without `shuffle_groups`
- **THEN** `ValueError` is raised

### Requirement: Circular shuffles
With `"circular"`, every shadow column and every permuted column SHALL be the original column cyclically shifted by an offset drawn uniformly from 1 to n−1 (n rows), from the same seeded generator as today. With fewer than 2 rows, `ValueError`. Joint shadows SHALL share one offset; independent shadows SHALL draw one per column.

#### Scenario: Shifted copy
- **WHEN** a recording estimator sees a permuted column under `"circular"`
- **THEN** it equals `np.roll(column, k)` for some k in 1..n−1

### Requirement: Selector pass-through
`ConsensusSelector(shuffle="rows")` SHALL pass `shuffle` to its ranker; with `"within_groups"` it SHALL use the `groups` given to `fit`.

#### Scenario: Selector with groups
- **WHEN** `ConsensusSelector(shuffle="within_groups", cv=GroupKFold(3))` is fit with `groups=g`
- **THEN** every split report has `shuffle == "within_groups"`
