## Why

`above_shadow_share` averages one vote per model family, so it measures agreement across families. The benchmark showed what that costs: in classification, four additive families cannot rank an interaction above the shadows, the share stays below 0.5, and interaction features are dropped (recall 0.44–0.64), while bagging and boosting find them. Users need to see which families find a group, and to choose a rule that asks "does any suitable family find signal?" without changing the default's meaning. Separately, the shadow block shuffles all columns with one shared row order, so collinear real pairs stay collinear in the shadows; whether that matters is untested, and users cannot try the alternative.

## What Changes

- Per-family shadow evidence: with shadows, every consensus row gains `family_above_shadow` (family → fraction of its voting members above that model's strongest shadow), and every resampled stability row gains `family_above_shadow_share` (family → sum of its per-split fractions divided by `n_splits`; splits where it did not vote count as 0) and `shadow_families` (how many families have a share above 0.5).
- `ConsensusSelector(min_shadow_families=None)`: an opt-in filter keeping groups with `shadow_families >= min_shadow_families`; it turns shadows on. Defaults and existing rules are unchanged.
- `FeatureRanker(shadow_mode="joint")` and `ConsensusSelector(shadow_mode="joint")`: `"independent"` shuffles each shadow column with its own row order. With shadows on, reports record `shadow_mode`.
- Docs: how to read family evidence; when to use `min_shadow_families` (suspected interactions or effects only some model types capture) and its risk (one overfitting family can pass noise; quantified in the next change, `calibration-study`).
- Version 1.6.0.

Report schema: additive only, and only with shadows on.

## Non-goals

- Changing `min_shadow_share`, `max_shadow_q`, their defaults, or any threshold.
- Calibration claims (next change).
- Choosing which families count as "suitable" automatically.

## Capabilities

### New Capabilities

### Modified Capabilities
- `shadow-significance`: shadow mode option and construction; per-family evidence in consensus and stability rows.
- `consensus-selector`: `min_shadow_families` and `shadow_mode` parameters, validation, shadow activation, selection rule.

## Impact

- Code: `featranker/importance.py` (`_with_shadows`, `_build_consensus`, `_build_stability`, report keys), `featranker/selection.py`.
- Docs: docs/usage.md, docs/report.md, README; CHANGELOG (local). Typing consumer.
