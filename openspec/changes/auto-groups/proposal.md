## Why

Permutation importance splits credit between correlated features: permuting one of two near-copies barely hurts a model that can use the other, so both look unimportant. FeatRanker already fixes this when the user groups them by hand (`feature_groups`), and 1.1 reports correlated pairs, but users must still build the groups themselves. A target-free helper that proposes the groups, plus an `"auto"` option where the user cannot compute them (inside a scikit-learn pipeline), closes the loop.

## What Changes

- New public function `featranker.auto_groups(X, *, threshold=0.7, feature_names=None) -> Dict[str, List[str]]`: clusters columns by Spearman correlation (complete linkage on `1 - |rho|`), so every pair inside a returned group has `|rho| >= threshold`. Returns only groups with two or more columns, ready to pass as `feature_groups`. Uses `X` only, never a target.
- `ConsensusSelector(feature_groups="auto")`: computes the groups with `auto_groups` on the `X` passed to `fit()`, so inside a pipeline the groups come from the training fold only. The groups used are visible in `report_["feature_groups"]`.
- CLI: `--feature-groups auto` computes the groups on the training CSV.
- Version 1.3.0.

Report schema: unchanged. Auto groups appear in the existing `feature_groups` key like hand-written ones.

## Non-goals

- `"auto"` in `FeatureRanker.rank_features()` or `rank_resampled()`: users call `auto_groups()` on their training or development data and pass the dict, which keeps the grouping explicit and visible in their code.
- Conditional permutation importance or other correlation-aware importance methods.
- Choosing the threshold automatically.

## Capabilities

### New Capabilities
- `auto-groups`: the `auto_groups` function, `ConsensusSelector(feature_groups="auto")`, and CLI `--feature-groups auto`.

### Modified Capabilities
- `public-api`: `__all__` gains `auto_groups`.

## Impact

- Code: `featranker/evidence.py` or a new small module for the clustering (reusing the 1.1 Spearman helper), `featranker/__init__.py`, `featranker/selection.py`, `featranker/cli.py`.
- Dependencies: none new (`scipy.cluster.hierarchy` ships with scipy, which scikit-learn requires).
- Docs: README, docs/usage.md (correlated features section), docs/api.md (new public name), CHANGELOG (local).
