## Why

Users judge models on more than one metric: accuracy and log loss, R² and MAE. FeatRanker ranks on one scorer, so checking whether a ranking holds under another metric means a second full run, refitting every model. scikit-learn's `permutation_importance` accepts several scorers and reuses each permuted copy; FeatRanker should too, while keeping one primary scorer for the gate, consensus, shadows, and selection so the existing evidence is unchanged.

## What Changes

- `scoring` in `rank_features()`, `rank_resampled()`, and `ConsensusSelector` accepts a list or tuple of scorer names or callables (at least one, unique names). The first is primary: the dummy gate, consensus, shadow evidence, stability, and selection use it alone, exactly as today.
- Secondary scorers are computed on the same permuted copies. Each model report gains `secondary`: per secondary scorer, `evaluation_score` and `importance` (`mean`, `std` per group). The report gains `secondary_scoring` (names) and `secondary_baseline_scores` (the dummy's score per secondary scorer, or `null`).
- A secondary scorer that fails for a model is an `evidence` failure (the model stays ranked); that scorer's entry for the model is `null`.
- CLI: `--scoring` accepts comma-separated names (`--scoring accuracy,neg_log_loss`).
- Version 1.14.0.

Report schema: additive (`secondary`, `secondary_scoring`, `secondary_baseline_scores` only when more than one scorer is given).

## Non-goals

- A consensus or stability table per secondary scorer.
- Choosing the primary scorer automatically.
- Dict-of-scorers input (names come from strings or callables, as today).

## Capabilities

### New Capabilities
- `multiple-scorers`: list-valued `scoring`, secondary importances, failures, pass-through.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (scorer resolution, permutation loop, model report), `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md (several metrics), docs/report.md, README; CHANGELOG (local).
