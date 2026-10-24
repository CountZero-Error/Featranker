## Why

With several scorers (1.14), every permuted copy is scored by each scorer separately, so the model predicts once per scorer on identical data. scikit-learn's own `permutation_importance` caches predictions across metrics. For slow models (SVR, large forests) the extra predictions dominate run time (external review, 2026-10-23).

## What Changes

- When `scoring` lists several scorers, the baseline and every permuted copy are scored with one shared prediction per response method (`predict`, `predict_proba`, `decision_function`) for the scorers that support it, through scikit-learn's multi-metric scorer. Scorers it cannot cache (plain callables) are called as today.
- Scores, importances, failures and reports are unchanged; a secondary scorer that fails still fails alone, and a primary scorer's failure keeps its current exception and message.
- If the multi-metric scorer is unavailable or behaves unexpectedly, scoring falls back to today's per-scorer calls.
- Version 1.19.0.

Report schema: unchanged.

## Non-goals

- Caching across permuted copies or groups (each copy is new data).
- Caching for a single scorer (one prediction already).
- Parallelism inside a model's permutation loop.

## Capabilities

### New Capabilities
- `scorer-cache`: shared predictions across multiple scorers.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (`_group_importances`, baseline scoring in `_rank_model`).
- Docs: docs/usage.md (multiple scorers: cost note), CHANGELOG (local), roadmap 1.19.
