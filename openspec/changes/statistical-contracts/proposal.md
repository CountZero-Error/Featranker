## Why

The external review (Signal, 2026-10-23) found statistical contracts that the docs understate:

1. `shuffle="within_groups"` measures reliance on within-group variation only. A predictor constant within each patient gets zero importance even when it drives the target (reproduced: R² ≈ 1.0, importance 0 in every fold; free shuffling gave ≈ 2). The recipe presents the mode only as "preserving dependence".
2. With `sample_weight`, shuffled values come from every evaluation row, zero-weight rows included; the scores are weighted, the donor distribution is not. That suits cost weights, not survey/frequency weights.
3. The ablation docs suggest a third data set removes optimism. It does for prespecified k; once used to choose k, its best score is a validation score, not a final estimate.
4. Each custom model is its own family, so several near-identical custom estimators can outvote other model types; the family-balancing claim does not hold for them, and there is no way to say which custom models belong together.
5. The ablation example in usage.md does not run after the README quick start (different models, scorer, and group count).

## What Changes

- `FeatureRanker(families={model_name: family})` assigns families for any configured model (custom or built-in); unknown names or non-string families raise `ValueError`. `ConsensusSelector(families=...)` passes it through. Reports already carry each model's `family`.
- With `shuffle="within_groups"`, `rank_features()` records `constant_within_groups`: the feature groups whose every column is constant within every shuffle group on the scored rows. `summarize()` adds a `warning` finding `constant_within_groups` naming them. A `UserWarning` is emitted.
- Docs: within-group estimand (what it measures, what it cannot, when to use free or group-level shuffling instead); weighted donor law (cost vs population weights); validation-set wording for choosing k with a separate test set or outer CV for final performance; custom-family risk next to the consensus claim; self-contained ablation example.
- Version 1.18.0.

Report schema: additive (`constant_within_groups` only with `shuffle="within_groups"`).

## Non-goals

- A weighted donor option (resampling shuffle donors by weight); document first, add when asked.
- Whole-group (between-group) permutation.
- CLI `--families` (the CLI has no custom models).

## Capabilities

### New Capabilities
- `statistical-contracts`: custom model families, the within-group constant-feature record and finding, and the documented contracts for within-group shuffles, weighted donors, and ablation validation.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`families`, `constant_within_groups`), `featranker/selection.py`, `featranker/summary.py`.
- Docs: docs/usage.md (grouped-data recipe, sample weights, ablation, consensus/custom models), docs/report.md, docs/api.md, README if it states the consensus claim; CHANGELOG (local); roadmap 1.18.
