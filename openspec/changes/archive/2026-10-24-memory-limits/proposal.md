## Why

The memory study (spikes/memory/FINDINGS.md, 2026-10-24) found one cliff and several avoidable copies:

- High-cardinality categorical columns are one-hot coded densely for built-in linear models: 10k rows with one cardinality-1000 column passed 6 GB before fitting finished; cardinality 10000 is infeasible.
- Shadow columns are built with a gather plus `hstack`, a 3X transient where 2X is needed.
- `_spearman_matrix` (correlation evidence, `auto_groups`) holds ranks, centered ranks, and squared ranks at once (+1.6 GB for 381 MB of X).
- `max_samples` caps scoring rows only; fitting always uses every training row, and tree fits at 100k rows take minutes.

## What Changes

- `FeatureRanker(max_categories=None)`: when set, built-in linear models' one-hot encoders keep at most that many levels per column and bucket the rest as infrequent (scikit-learn `OneHotEncoder(max_categories=..., handle_unknown="infrequent_if_exist")`). Default unchanged. CLI `--max-categories`; `ConsensusSelector(max_categories=...)` passes it through.
- Shadow block preallocated and filled in place (same values, lower peak).
- `_spearman_matrix` ranks column blocks into one preallocated array with fewer full-size temporaries (same values, bit for bit).
- `fit(..., max_samples=None, random_state=None)` fits models (and builds training shadows and evidence copies) on a random subsample of training rows: whole groups under `shuffle="within_groups"`, row order kept under `"circular"`, and the zero-weight check as in `rank_features`. Reports record `run.n_train_rows`, `train_max_samples`, `train_random_state` and `train_rows_sha256` when it is used. `rank_resampled(..., max_train_samples=)`, `ConsensusSelector(max_train_samples=)` and CLI `--max-train-samples` pass it through. `train_data_sha256` still fingerprints the data passed to `fit()`.
- Version 1.21.0.

Report schema: additive (`run.n_train_rows`, `train_max_samples`, `train_random_state`, `train_rows_sha256` only with a training subsample; `max_categories` recorded in `run` when set).

## Non-goals

- Sparse model input (changes solver choice and numerics for linear models).
- float32 storage (changes numerics and fingerprints).
- Out-of-core or chunked estimators.
- Row-sampled correlation evidence (changes values).

## Capabilities

### New Capabilities
- `memory-limits`: category cap, training-row subsample, and lower-peak shadows and correlations.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`_with_one_hot`, `_with_shadows`, `fit`, `rank_resampled`, `__init__`, pickles), `featranker/evidence.py` (`_spearman_matrix`), `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md (large data section: what each option bounds, with the study's numbers), docs/report.md, README; CHANGELOG (local); roadmap 1.21; config limitations L11/L12.
