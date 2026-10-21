## Why

Permutations and shadows shuffle rows freely (L22). With grouped data (patients, stores) or time series this breaks the dependence between rows: a shuffled feature mixes values across groups or destroys autocorrelation. Its importance then measures a mix of the feature's signal and the broken structure, and shadows look less like real features than they should. Users with `GroupKFold` or `TimeSeriesSplit` need a shuffle that respects the structure they already declared.

## What Changes

- `FeatureRanker(shuffle="rows")` gains two modes, applied both to permutation importance and to shadow columns:
  - `"within_groups"`: values are shuffled only among rows that share a group label. Labels are passed as `shuffle_groups` to `fit()` (shadows) and `rank_features()` (permutations).
  - `"circular"`: each shuffle is a cyclic shift of the rows by a random offset, which keeps a series' autocorrelation. It assumes the rows are in time order.
- `rank_resampled(..., groups=)` uses the same `groups` for `"within_groups"`; `ConsensusSelector(shuffle=...)` passes it through.
- Reports record `shuffle` when it is not `"rows"`.
- A group with a single row cannot be shuffled; if every row is in such a group, `ValueError`.
- Version 1.15.0.

Report schema: additive (`shuffle` only when not `"rows"`).

## Non-goals

- Block bootstrap or block permutation with a chosen block length.
- Shuffling whole groups between each other (groups of unequal size have no clean mapping).
- Inferring groups or time order from the data.
- CLI support (the CLI has no group column; add when asked).

## Capabilities

### New Capabilities
- `dependent-shuffling`: the `shuffle` option for permutations and shadows, its group labels, and its record.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`_permutation_importances` shuffle, `_with_shadows`), `featranker/selection.py`.
- Docs: docs/usage.md (the time-series recipe and a grouped-data recipe now use `shuffle`), docs/report.md, README; CHANGELOG (local); roadmap L22.
