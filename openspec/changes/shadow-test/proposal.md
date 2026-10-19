## Why

Shadow features tell a user what share of model families rank a group above noise, but not how sure that is: FeatRanker has no significance test, while Boruta offers one. Resampled ranking with shadows already records, per split, whether a group beat its shadows, so a Boruta-style hits test across splits costs no extra model fits.

## What Changes

- `rank_resampled()` with shadows adds, to every stability row: `shadow_hits` (splits in which the group's consensus `above_shadow_share > 0.5`), `shadow_p_value` (one-sided binomial test of the hits against chance 0.5 over all `n_splits`), and `shadow_q_value` (Benjamini–Hochberg across groups).
- The resampled report gains `shadow_test`: the method, the hit rule, the correction, and `min_p_value` (the smallest p-value the number of splits allows), so users can see when there are too few splits for the test to say anything.
- `ConsensusSelector(max_shadow_q=None)`: a third selection rule that keeps groups with `shadow_q_value <= max_shadow_q`; it turns shadows on, like `min_shadow_share`.
- Version 1.4.0.

Report schema: additive only, and only when shadows are on in a resampled run: three stability-row keys and one top-level key.

## Non-goals

- PIMP or any test that refits models on permuted targets.
- Tests on single-split reports (one split is one trial).
- Changing `above_shadow_share` or how a split's hit is defined elsewhere.

## Capabilities

### New Capabilities

### Modified Capabilities
- `shadow-significance`: the hits test, its report keys, and `shadow_test` metadata in resampled reports.
- `consensus-selector`: the `max_shadow_q` parameter, its validation, shadow activation, and selection rule.

## Impact

- Code: `featranker/importance.py` (`_build_stability` and the resampled report), `featranker/selection.py`.
- Dependencies: none new (`scipy.stats.binom`).
- Docs: docs/usage.md (significance section with the caveats), docs/report.md (new keys), README, CHANGELOG (local).
