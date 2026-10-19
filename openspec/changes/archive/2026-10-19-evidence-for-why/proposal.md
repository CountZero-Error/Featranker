## Why

A FeatRanker report says which feature groups matter, but not why: it cannot tell a user that a model overfits, that two groups share credit because they are correlated, or which way a feature pushes the prediction. The investigator spike (archived `2026-10-19-spike-investigator-value`) found that this evidence, not an LLM, carried the gains, and its preregistered rule ended in NO-GO for LLM use. 1.1 therefore adds the evidence and deprecates the LLM functions.

## What Changes

- New constructor flag `FeatureRanker(..., evidence=False)`. When `True`, `fit()` keeps the training data and `rank_features()` adds mechanism evidence to the report:
  - per model: `train_score` (the scorer on the training data) and `overfit_gap` (`train_score - evaluation_score`);
  - a report-level `evidence` block with feature-group pairs whose Spearman correlation on the training data reaches a fixed threshold;
  - with the new `shap` extra installed: per model and group, mean |SHAP| and direction on the evaluation rows, for tree models (TreeExplainer) and linear models (LinearExplainer) only. Other models are listed as unsupported. Without `shap`, the SHAP part is skipped with a warning and the rest is still computed.
- `rank_resampled()` passes `evidence` to its per-split rankers, so each split report carries its own evidence. No cross-split aggregation.
- `summarize()` gains two findings: `overfit` (a model lost at least half of its training advantage over the dummy on the evaluation data) and `correlated_groups` (importance may be split between correlated groups).
- CLI: `--evidence` flag.
- New extra `shap = ["shap"]`; `all` is unchanged.
- `explain()` and `to_prompt()` emit `FutureWarning` and are documented as deprecated; removal in 2.0 per the stability policy. They keep working unchanged in 1.x.
- Version 1.1.0.

Report schema (public API): additive only. New keys appear only when `evidence=True`: `models[*].train_score`, `models[*].overfit_gap`, `models[*].shap` (when computed), and top-level `evidence`. Existing keys and their meaning are unchanged. Reports made without `evidence` are identical to 1.0 reports.

## Non-goals

- No LLM features; no new behaviour in `explain()`/`to_prompt()`.
- No automatic feature grouping from correlation (planned separately as `auto-groups`).
- No KernelExplainer or other model-agnostic SHAP (too slow for a default); no SHAP interaction values.
- No significance test, no speed presets (planned separately).
- No change to ranking, consensus, shadows, or `ConsensusSelector`.

## Capabilities

### New Capabilities
- `mechanism-evidence`: the `evidence` flag, train score and overfit gap, correlated groups, optional SHAP summary, resampled pass-through, and the CLI flag.

### Modified Capabilities
- `report-summary`: two new finding rules, `overfit` and `correlated_groups`.
- `llm-interpretation`: `explain()` and `to_prompt()` are deprecated with a `FutureWarning`.
- `public-api`: new `shap` extra.

## Impact

- Code: `featranker/importance.py` (flag, stored training data, evidence assembly), a new `featranker/evidence.py` (correlation and SHAP helpers), `featranker/summary.py`, `featranker/interpret.py` (warnings), `featranker/cli.py`.
- Dependencies: optional `shap` only; scipy (Spearman) already comes with scikit-learn.
- Pickles: a ranker fitted with `evidence=True` also stores its training data.
- Docs: README, docs/usage.md, docs/report.md, CHANGELOG (local), deprecation notes.
- Tests: new evidence tests; SHAP tests skip when `shap` is not installed.
