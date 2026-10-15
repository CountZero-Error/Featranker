## Context

- `FeatureRanker` holds fitted state: `trained_models`, `label_encoder`, `feature_names_`, and `baseline_model_`.
- `fit()` handles input validation, label encoding, and the dummy baseline. `rank_features()` does the ranking.
- Construction reads the registry, or takes `models=`.
- This change builds on `trust-the-consensus` (0.5): split reports include family fields, and `summarize` exists.
- Motivation is in proposal.md; requirements are in specs/.

## Goals / Non-Goals

**Goals:**
- Reuse `fit()` and `rank_features()` unchanged for each split, so a split report is exactly what a user would get by running those two calls by hand on that split.
- The stability table is a pure function of the split reports, so it can be unit-tested with synthetic reports.

**Non-Goals:**
- Caching or reusing model fits across splits.
- A progress bar over splits beyond the existing per-split bars.

## Decisions

**1. A method on `FeatureRanker`, not a free function or a new class.**

`FeatureRanker(task="reg").rank_resampled(X, y, scoring=..., cv=5)` is the easiest form to discover and reuses the constructor's settings. The method builds a fresh ranker per split and never touches `self`, so there is no hidden refit.

Alternatives considered:
- a free function taking a template ranker, which is awkward;
- a `ResampledRanker` class, which duplicates the constructor and adds a second class to the API.

**2. Fresh rankers per split come from the stored constructor arguments.**

- `FeatureRanker(task, group, models=self.user_models, n_jobs, verbose)`. With `models=`, `group` must stay `"all"`, which is already the case for such rankers.
- Rebuilding the registry per split repeats initialization warnings, so a run emits the warnings for missing optional libraries once per split. Accepted, because it keeps each split identical to a manual run.

Alternative considered: deep-copying `self.models`. Rejected because it shares no clear advantage and diverges from the manual-run equivalence.

**3. Normalize `cv` once, up front, and materialize the index pairs before any fit.**

- An integer `cv` builds a shuffled `StratifiedKFold` or `KFold` with `random_state`.
- An object with `split` is called as `split(X_values, y_values, groups)`.
- Anything else must iterate to `(train, eval)` pairs.

Each pair is converted to an integer index array and checked to be non-empty, so an invalid `cv` fails before any model is fit, as the spec requires. Validating `X` and `y` once, through the same helpers `fit()` uses, gives positional arrays for indexing. Slices are passed to `fit()` and `rank_features()` with `feature_names`, so DataFrame column names survive.

Alternative considered: `sklearn.model_selection.check_cv`. Rejected because its integer path does not shuffle, and unshuffled folds on sorted data are a real trap.

**4. Per-split seed is `random_state + i`.**

This is simple and visible in the report.

Alternative considered: `SeedSequence.spawn`. It has better statistical independence, but the seeds are opaque to users. It doesn't matter here, because permutations within a split are already independent.

**5. Stability is built from each group's place in each split's sorted consensus, via pure helpers `_consensus_positions` and `_build_stability(split_reports, n_splits, top_k)`.**

- A group's place is its 1-based row position. Equal (`median_rank`, `mean_rank`) share the better place, using competition ranking.
- Splits whose consensus is empty contribute `null` and count as "not top k".
- Groups always share the same `n_splits_ranked`, because a non-empty consensus ranks every group.
- `top_k_share` uses `n_splits` (all attempted splits) as its denominator, so failed splits lower the share instead of being silently dropped.

Alternative considered: thresholding the consensus `median_rank` score (`median_rank <= top_k`). This was the first implementation. Review showed it misleads, because the score is a median of family ranks and so is not a place. On diabetes, `bp` was third in every split, yet had scores like 4.5 and only 40% "top 3". The place is what users mean by "top k".

**6. Failure handling.**

- A split-level exception, such as `fit()` raising or a stratification problem showing up as `rank_features` rejecting unseen labels, becomes one `stage: "split"` failure, and the loop continues.
- Model-level failures are copied from each split report with `split` added.
- If zero splits rank, the method raises `RuntimeError`.

**7. `summarize` guard.**

The check runs before the required-key check and looks at `evaluation_mode == "resampled"`, so users get guidance instead of a confusing "missing baseline_score" error.

## Risks / Trade-offs

- [Cost is about `n_splits` times one ranking; `group="all"` with `cv=10` can be slow] → Docs give the cost formula and recommend starting with `cv=5` and `group="tree"` or fewer repeats.
- [Users pass their final test set into `rank_resampled`, which leaks] → A leakage note in docs: pass development data only. The splits are internal validation, not a final test.
- [`top_k=5` is arbitrary] → It is a parameter, echoed in the report. Docs explain how to choose it.
- [Repeated initialization warnings per split] → Accepted (decision 2). Users can install the extras or set `verbose=False`; warnings are not suppressed.

## Migration Plan

- Additive: a new method and a new report shape. Nothing existing changes. Release as 0.6.0.
- Rollback means reverting the release.
