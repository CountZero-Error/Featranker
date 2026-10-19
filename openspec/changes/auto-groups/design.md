## Context

1.1 added `featranker/evidence.py` with `_spearman_matrix` (rank-and-center Spearman, NaN for constant columns) and reports correlated pairs. `_resolve_feature_groups` accepts a partial mapping; unlisted columns become singletons. `ConsensusSelector.fit()` passes `feature_groups` to `rank_resampled()` on the `X` it was given. The CLI reads `--feature-groups` as a YAML path. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:** explicit, deterministic groups the user can read; no target and no evaluation data shaping them inside pipelines.

**Non-Goals:** `"auto"` on `FeatureRanker` methods (proposal Non-goals).

## Decisions

### D1. Complete linkage on 1 − |rho|
`scipy.cluster.hierarchy.linkage(squareform(1 - |rho|), method="complete")` then `fcluster(..., t=1 - threshold, criterion="distance")`. Complete linkage guarantees every pair in a group meets the threshold, which is easy to state and check. Alternatives: average linkage (no per-pair guarantee), connected components of the threshold graph (chains unrelated columns, the case in the "Complete linkage" scenario). NaN (constant column) becomes distance 2 (see below). The diagonal is forced to 0 and the matrix symmetrised before `squareform` to avoid float asymmetry errors.

Floating point: computed |rho| for exact copies can be 1 − 2e-16, so the cut is `t = 1 - threshold + 1e-12` (the spec's tolerance). NaN (constant column) maps to distance 2.0, not 1.0, so no threshold can merge it. A generated name that collides with a column or another generated name raises.

### D2. Location and reuse
`auto_groups` lives in `featranker/evidence.py`, next to `_spearman_matrix`, and is exported from `featranker`. Input validation reuses `_validate_X` from `importance.py` (names from DataFrame, `feature_names`, or `x0..`).

### D3. Naming
`"+".join(members)` in column order; groups sorted by first member's column index. A name colliding with an existing column raises `ValueError` (it would otherwise collide with that column's implicit singleton in `_resolve_feature_groups`).

### D4. Selector
In `fit()`, after names are known and before ranking: if `feature_groups == "auto"`, replace it locally with `auto_groups(X, feature_names=names)`; the attribute stays `"auto"` for `get_params`/`clone`. Other strings raise `ValueError` in `fit()` (sklearn convention: no validation in `__init__`). The inner `rank_resampled()` then uses fixed groups across its splits, so stability rows stay comparable. The groups are computed on all rows passed to `fit()`, including the inner evaluation folds; this is target-free, and it is the outer training fold only.

### D5. CLI
`--feature-groups auto` (exact string) → `auto_groups` on the training CSV's feature columns (target dropped). A file literally named `auto` would need `./auto`; documented.

## Risks / Trade-offs

- [Grouping hides which member carries the signal] → the group lists its members; users can rank members separately afterwards.
- [Threshold choice matters] → default 0.7, documented; the 1.1 evidence lists pairs from 0.5 so users can see what is near the cut.
- [O(p²) correlation and clustering] → fine for hundreds of columns (measured in 1.1 review: 300 groups 0.4 s); documented.
