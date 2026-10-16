## Context

- `fit()` validates `X`, encodes labels, and fits each candidate model on the full feature matrix. The dummy is fit separately.
- `rank_features()` resolves feature groups to column indices, then `_rank_model` computes permutation importance per group and `_assign_ranks` ranks them.
- Feature groups are only known at rank time.
- Requirements are in specs/. Motivation is in proposal.md.

## Goals / Non-Goals

**Goals:**
- `shadows=False` follows exactly the 0.6 code path and produces exactly 0.6 output.
- Shadows reuse the existing permutation machinery. The shadow group for a real group is the same column indices shifted by `n_features`.

**Non-Goals:**
- Exposing shadow columns to callers. The shadow block is built and dropped inside `fit()` and `rank_features()`.

## Decisions

**1. One row permutation for the whole shadow block, not one per column.**

Groups are only known at rank time, so shadows are built per column at fit. Using one row order for all columns keeps every within-row relationship. One-hot blocks still sum to 1, and correlated features stay correlated. So a shadow group has the same structure as its real group, with rows randomly reassigned relative to `y`. A particular shuffle can still retain chance association with the target, so it is a randomized reference, not a guaranteed null.

Alternative considered: shuffling each column independently, as in classic Boruta. Rejected because it destroys group structure, so a shadow one-hot block would no longer be a valid one-hot block. That makes shadow importance an unfair comparison for grouped features.

**2. The shadow block is appended after the real columns, as `X_aug = [X | X[perm]]`.**

The real columns keep positions `0..n-1`, so positional pipelines written for 0.6 still select the same real columns. Shadow group indices are `real_indices + n_features`.

Alternative considered: interleaving shadow and real columns. Rejected because it breaks positional selectors.

**3. Seeds.**

- The ranker holds `shadow_seed`. `fit()` draws the training order from `default_rng(shadow_seed)`.
- `rank_features()` draws the evaluation order from `default_rng([shadow_seed, 1])`. That is a distinct, reproducible stream; with `None`, it is fresh entropy.
- The permutation-importance `random_state` is unchanged.

Alternative considered: reusing the permutation `random_state`. Rejected because shadows must exist at fit time, before `random_state` is known.

**4. Real ranks exclude shadows.**

`_assign_ranks` receives the real-group means only, so ranks and the consensus keep their meaning. Shadow means are compared, not ranked.

Alternative considered: ranking real and shadow groups together, as in a Boruta "hit" when a real group outranks all shadows. Rejected because it would shift every real rank and break 0.5/0.6 comparability.

**5. `above_shadow` is strict (`mean > shadow_max`), and the share is family-weighted.**

This matches the family voting principle from 0.5. Strict comparison means a group that ties the strongest shadow, for example all zeros, does not count as beating it.

**6. Where the code changes.**

- `_rank_model` gains `n_shadow_offset` (or `None`). When set, it computes shadow importances with the same rng stream after the real groups, and adds `shadow_importance`, `shadow_max`, and `above_shadow`.
- `_build_consensus` adds `above_shadow_share` when the voting reports have `shadow_max`.
- `_build_stability` averages it when present.
- `rank_resampled` passes `shadows` and `shadow_seed + i` through.
- `__setstate__` defaults both attributes.

## Risks / Trade-offs

- [Custom pipelines that drop columns, such as a positional `ColumnTransformer` with `remainder="drop"`, never see the shadows. Shadow importance is then 0, and everything "beats" the shadow] → Docs warning. `summarize` cannot detect it.
- [Extra noise columns change model behavior, for example `max_features="sqrt"` in forests, so real importances and ranks can differ between `shadows=True` and `False`] → Documented. The spec only promises equal ranks for models unaffected by extra columns.
- [Doubled fit and rank cost and memory] → Documented. Shadows are off by default.
- [`above_shadow_share` is mistaken for a p-value] → Docs call it descriptive. A `below_shadow` finding is info, not warning.

## Migration Plan

- Additive and off by default. Release as 0.7.0. Rollback means reverting the release.
