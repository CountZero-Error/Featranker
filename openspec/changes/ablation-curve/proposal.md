## Why

Permutation importance asks how much a fitted model's score drops when one group becomes noise. It does not show whether the ranking is useful: how well models retrained on the top groups score, and how many groups are needed. Correlated groups share permutation credit, so a model can lose little from each shuffle yet much when both are removed. An ablation curve (retrain on the top-k consensus groups, score on evaluation data) checks the ranking with a second, refit-based view and gives the user a direct answer to "how many features do I need".

The CLI also uses every non-target, non-weight column as a feature. ID or timestamp columns in a CSV become features unless the user edits the file.

## What Changes

- New `FeatureRanker.ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None, eval_sample_weight=None)`. It returns a new `Report`: a copy of `report` with an added `ablation` block. For each `k`, every voting model of `report` is refit on the columns of the top-k consensus groups and scored on the evaluation data.
- `summarize()` gains an `ablation_curve` info finding when the block is present.
- CLI: `--ablate` (adds the block) and `--ablate-ks 1,2,5`; `--drop col1,col2` removes columns from both CSVs before features are read.
- Version 1.16.0.

Report schema: additive (`ablation`, only from `ablate()` or `--ablate`).

## Non-goals

- Drop-one-group (leave-one-group-out) ablation; add later, capped to the top groups, if asked.
- Ablation of resampled reports.
- Choosing k automatically or with a tolerance threshold; the curve is descriptive.
- A plot helper for the curve.

## Capabilities

### New Capabilities
- `ablation-curve`: the `ablate()` method, its report block, its summary finding, and the CLI flags (`--ablate`, `--ablate-ks`, `--drop`).

### Modified Capabilities

## Impact

- Code: new `featranker/ablation.py`; a thin `FeatureRanker.ablate` in `featranker/importance.py`; `featranker/summary.py`; `featranker/cli.py`.
- Docs: docs/usage.md (ablation section with the reuse-of-evaluation-data caveat; CLI `--drop`), docs/report.md, docs/api.md, README; CHANGELOG (local); roadmap 1.16.
