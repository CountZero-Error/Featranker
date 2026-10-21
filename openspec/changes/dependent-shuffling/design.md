## Context

`_permutation_importances` draws `rng.permutation(n)` per group and repeat; `_with_shadows` draws one permutation (joint) or one per column (independent). See proposal.md.

## Decisions

### D1. One index helper
`_shuffle_index(rng, n, mode, labels) -> np.ndarray` returns a row index: `rng.permutation(n)` for `"rows"` (same call as today, so `"rows"` stays bit-identical), a per-label permutation for `"within_groups"` (labels visited in sorted order for determinism), and `np.roll(np.arange(n), rng.integers(1, n))` for `"circular"`. Permutations and shadows both call it.

### D2. Storage
`fit()` uses `shuffle_groups` for shadows only and does not store it; `rank_features()` takes its own labels for the evaluation rows. `rank_resampled` slices `groups` into train and evaluation labels per split.

### D3. Evidence unaffected
Correlations and SHAP do not shuffle and are unchanged.

## Risks / Trade-offs

- [`"circular"` on data not in time order is meaningless] → documented; the option is named for what it does.
- [Small groups give few distinct within-group permutations] → documented; single-row groups keep their values.
