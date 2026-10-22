## Context

`_permutation_importances` draws `rng.permutation(n)` per group and repeat; `_with_shadows` draws one permutation (joint) or one per column (independent). See proposal.md.

## Decisions

### D1. One index helper
`_shuffle_index(rng, n, mode, labels) -> np.ndarray` returns a row index: `rng.permutation(n)` for `"rows"` (same call as today, so `"rows"` stays bit-identical), a per-label permutation for `"within_groups"` (one `rng.random(n)` key per row; `argsort(labels)` gives the label-sorted positions and `lexsort((keys, labels))` the rows in (label, random key) order, which are assigned to those positions, so every group is shuffled among its own positions, a singleton maps to itself, and one draw of `n` numbers is deterministic given the generator), and `np.roll(np.arange(n), rng.integers(1, n))` for `"circular"`. Permutations and shadows both call it.

### D2. Storage
`fit()` uses `shuffle_groups` for shadows only and does not store it; `rank_features()` takes its own labels for the evaluation rows. `rank_resampled` slices `groups` into train and evaluation labels per split.

### D2a. Sampling, ordering, fingerprints, labels
With `max_samples` and `"within_groups"`, `rank_features()` samples whole groups: labels in random order from the same subsample generator, taking groups until the row count reaches the target (it may exceed it; `n_eval_rows` is the actual count). A random row subsample would leave rows alone in their group, which never move. Under `"circular"`, `rank_resampled` sorts each split's train and evaluation indices (splitters may return them unordered), and the row subsample is already sorted. The run block gets `shuffle_groups_sha256` (and `train_shuffle_groups_sha256` from `fit()`, like the weights) when labels are given, and `rank_resampled` the first. Labels with `None`, NaN or mixed types that `np.unique` cannot sort raise `ValueError`; `fit()` with `shadows=False` skips the all-singleton check since the labels are unused.

### D3. Evidence unaffected
Correlations and SHAP do not shuffle and are unchanged.

## Risks / Trade-offs

- [`"circular"` on data not in time order is meaningless] → documented; the option is named for what it does.
- [Small groups give few distinct within-group permutations] → documented; single-row groups keep their values.
