## Context

`_build_consensus` already collects per-family `above_shadow` flags (`above_by_group[group][family]`) and averages them into `above_shadow_share`; `_build_stability` averages per-split shares. `_with_shadows(X, seed)` appends one shared row permutation of all columns. See proposal.md for motivation.

## Decisions

### D1. Expose what is already computed
`family_above_shadow` is the per-family mean of member flags that `_build_consensus` already computes; `above_shadow_share` stays the mean of those values, so the existing number is unchanged. Family keys are sorted for deterministic output.

### D2. Stability aggregation over all splits
A family's stability share is the sum of its per-split fractions divided by `n_splits`; splits where it did not vote (its members failed the dummy gate or failed), failed splits, and empty splits count as 0, matching `shadow_hits`. Review found the first design (average over voting splits only) is selection-biased: a family that passes the dummy gate only in the few splits where it fits noise counted fully (null data: `min_shadow_families=2` selected 5–15 % of noise groups); dividing by `n_splits` brought that to the share rule's level with no loss of interaction recall. `shadow_families` counts families with share strictly above 0.5, consistent with the hit rule's strict majority.

### D3. Rule semantics
`min_shadow_families=N` answers "do at least N families find signal?". N=1 is the most permissive (any one family); docs present N=2 as a heuristic starting point (false-selection rate not yet measured) and point to the calibration study, which measures it. Filters combine with AND, then `k`, as today.

### D4. Shadow mode
`shadow_mode="independent"`: for each column j, `rng.permutation(n_rows)` drawn in column order from one generator seeded as today (`shadow_seed` for fit, `[shadow_seed, 1]` for evaluation), so results are reproducible and joint mode's stream is unchanged. The report records `shadow_mode` next to `shadows` and `shadow_seed`. `rank_resampled` and the selector pass it through; `__setstate__` defaults it to `"joint"` for 1.0–1.5 pickles.

### D5. Interpret module
`to_prompt` passes consensus/stability rows through, so the new keys reach the deprecated prompt automatically; no change there.

## Risks / Trade-offs

- [`min_shadow_families=1` lets one overfitting family (e.g. a single decision tree) select noise] → docs default advice N=2; the calibration study measures false selection on null data.
- [Report size grows by one dict per group] → only with shadows.
- [Independent mode changes which shadows are strongest; results differ from joint mode] → opt-in; recorded in the report.
