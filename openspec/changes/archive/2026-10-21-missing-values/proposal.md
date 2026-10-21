## Why

FeatRanker rejects any NaN, so users must impute before calling it. Imputing on all development rows before resampling leaks evaluation rows into the imputation, which undermines FeatRanker's leakage-safe promise; doing it per split by hand is exactly the pipeline work FeatRanker is meant to remove. Missing values are the most common complaint in comparable tools' issue trackers (BorutaShap #44, #59, #122).

## What Changes

- `FeatureRanker(missing="error")` gains `missing="impute"`: NaN is accepted in features (infinite values are still rejected); every built-in model gets a median `SimpleImputer` as the first pipeline step, so imputation is learned on each training split only. Custom `models=` receive NaN unchanged and must handle it.
- With `"impute"`, reports record `missing`: the policy, the strategy, and the number of missing values per feature in the training data.
- Evidence with NaN: Spearman correlations (correlated groups and `auto_groups`) use pairwise-complete rows; SHAP unwraps the imputer as a per-column step.
- `ConsensusSelector(missing=...)`, `rank_resampled`, and CLI `--missing impute` pass the policy through.
- Default `"error"` keeps 1.x behaviour, including the error message.
- Version 1.7.0.

Report schema: additive (`missing` only with `"impute"`).

## Non-goals

- Categorical or string columns (next change, `categorical-features`).
- Native missing-value handling per model or a missingness-indicator feature (documented as a limitation; users can pass custom models).
- Missing values in the target.

## Capabilities

### New Capabilities
- `missing-values`: the `missing` option, fold-local imputation for built-in models, NaN-aware evidence, report record, and pass-through.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`_validate_X`, `init_models`, report), `featranker/evidence.py` (pairwise Spearman, imputer in `_unwrap`), `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md, docs/report.md, README; CHANGELOG (local).
