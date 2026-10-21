## Why

Permutation importance costs (models × feature groups × repeats) predictions over every evaluation row (L11). On large evaluation sets most of that time buys little precision, and there is no way to trade rows for speed. scikit-learn's `permutation_importance(max_samples=)` offers this; FeatRanker should too, with the dummy gate and every score computed on the same rows.

## What Changes

- `rank_features(..., max_samples=None)` and `rank_resampled(..., max_samples=None)`: an int (rows) or a float in (0, 1] (share of evaluation rows). One subsample, drawn without replacement with `random_state`, is used for the baseline score, the dummy score, and every permuted score. Train scores and evidence are unchanged.
- `ConsensusSelector(max_samples=None)`; CLI `--max-samples`.
- Reports record `max_samples` and `n_eval_rows` (rows actually scored) when subsampling is on.
- Version 1.13.0.

Report schema: additive (`max_samples`, `n_eval_rows` only when `max_samples` is given).

## Non-goals

- Stratified subsampling (a plain random sample; documented).
- Skipping permutation for models that lose to the dummy (it would remove their importances from the report).
- A cost estimator.

## Capabilities

### New Capabilities
- `eval-subsampling`: `max_samples` through ranking, resampling, selector, and CLI.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py`, `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md (speed section), docs/report.md, README; CHANGELOG (local).
