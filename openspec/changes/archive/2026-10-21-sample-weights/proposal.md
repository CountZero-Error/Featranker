## Why

Survey data, imbalanced classes, and aggregated rows come with sample weights. FeatRanker ignores them: models are fit unweighted, and scores (baseline, permuted, train) are unweighted, so the ranking answers a different question than the user's weighted model would. scikit-learn's `permutation_importance` and feature-engine's selectors accept weights; FeatRanker should too, consistently through fitting, the dummy baseline, scoring, and resampling.

## What Changes

- `FeatureRanker.fit(..., sample_weight=None)`, `rank_features(..., sample_weight=None)` (evaluation weights), and `rank_resampled(..., sample_weight=None)` (sliced per split for training and evaluation).
- With weights: every model is fit with its training weights (routed to the final step of built-in pipelines); the dummy baseline is fit and scored with weights; baseline, permuted, and train scores are weighted. A model whose final estimator does not accept `sample_weight` (built-in: `lda`, `qda`, `pls_regression`) is not fit and is recorded as a `fit` failure with `error_type` `"SampleWeightNotSupported"`, rather than silently fit unweighted.
- Reports record `sample_weight`: whether training and evaluation weights were used.
- `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)`, usable in pipelines through scikit-learn metadata routing (`set_fit_request(sample_weight=True)`).
- CLI: `--sample-weight COLUMN` reads a weight column from both CSVs and excludes it from the features.
- Evidence: correlations and the SHAP summary stay unweighted (documented).
- Version 1.11.0.

Report schema: additive (`sample_weight` only when weights are given).

## Non-goals

- Weighted Spearman correlations or weighted SHAP summaries.
- Class-weight presets or automatic weight computation.

## Capabilities

### New Capabilities
- `sample-weights`: weights through fit, baseline, scoring, resampling, selector, and CLI; unsupported-model handling; report record.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (fit/score call sites, validation), `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md (weights section), docs/report.md, README; CHANGELOG (local).
