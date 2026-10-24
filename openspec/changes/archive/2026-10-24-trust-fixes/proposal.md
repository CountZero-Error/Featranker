## Why

An external review (Signal, 2026-10-23) reproduced six defects that weaken FeatRanker's trust claims:

1. A `ColumnTransformer` nested inside another one's branch is not inspected, so a model can drop shadow columns without a `shadows` failure; its shadow importances are then zero and it still contributes shadow evidence.
2. `scikit-learn` has no version floor, but `_unrouted()` needs `config_context(enable_metadata_routing=...)` (1.3+) and imputation needs a 1.2+ option. Older installs satisfy the metadata and then fail.
3. Fingerprints hash every numeric column as float64, so distinct integers above 2**53 collide (swapped labels `2**53` and `2**53+1` give equal fingerprints).
4. The `ablation` block does not record which evaluation data, weights, or row count produced the curve.
5. The ablation `median_score` compares changing model sets: a model failing at small k and joining at larger k moves the median although no model's score changed, and `summarize()` reports that as a best k.
6. `max_samples` with weights can sample only zero-weight rows, and every scorer then fails with `ZeroDivisionError`.

## What Changes

- Shadow-drop check walks into `ColumnTransformer` branches (and pipelines inside them), mapping each branch's columns back to model-input columns. A model whose shadow columns cannot be confirmed as kept gets a `shadows` failure instead of a warning.
- `scikit-learn>=1.5` (the tested floor).
- Fingerprint scheme 2: integer and boolean columns are hashed losslessly, with a dtype tag. Reports record `run.fingerprint_scheme: 2`. `ablate()` checks reports without the key with scheme 1.
- `ablation` gains `data_sha256`, `weights_sha256`, `n_eval_rows`, and `same_holdout`; each curve row gains `n_models_scored`; the block gains `paired_models` and each row `paired_median_score` (median over models scored at every k). `summarize()` uses the paired median and adds a `warning` finding when the set of scored models changes across k.
- `rank_features(max_samples=..., sample_weight=...)` raises `ValueError` when the sampled rows' weights sum to zero.
- Version 1.17.0.

Report schema: additive (`run.fingerprint_scheme`; new `ablation` keys). Fingerprint values change for integer and boolean data; that is the point of the fix and is noted in the changelog.

## Non-goals

- Mapping columns through non-`ColumnTransformer` steps that change column count before a later `ColumnTransformer` (still only checked when that transformer sees the full model input).
- A different sampling policy for weighted subsampling (resampling until weight is positive).
- Family-balanced ablation medians (documented as model-level).

## Capabilities

### New Capabilities
- `trust-fixes`: nested shadow-drop detection, dependency floor, fingerprint scheme 2, ablation provenance and cohort, zero-weight subsample error.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`_column_transformers`/`_drops_shadows`, `_hash_column`/`_fingerprint`, `rank_features` subsample check), `featranker/ablation.py`, `featranker/summary.py`, `pyproject.toml`.
- Docs: docs/usage.md (named-input shadow check, ablation section, comparing runs), docs/report.md (`run.fingerprint_scheme`, ablation keys), CHANGELOG (local), roadmap 1.17.
