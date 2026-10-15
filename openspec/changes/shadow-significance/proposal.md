## Why

FeatRanker can say which feature groups rank highest (0.5) and how stable that order is across splits (0.6), but not whether a group carries any more signal than noise. A group can rank 5th out of 10 purely because the other five are worse. The roadmap's next question is "does this feature beat noise?" The standard, model-agnostic answer is a shadow-feature comparison in the style of Boruta: give each model a shuffled copy of every feature, and ask whether a real group beats the best shadow.

## What Changes

- **New constructor options.** `FeatureRanker(..., shadows=False, shadow_seed=None)`. With `shadows=True`, `fit()` appends one shadow column per feature. The shadow block is the training rows in one shuffled order, so the shadows keep the features' values and within-row relationships while rows are randomly reassigned. This gives a randomized reference; a particular shuffle can retain chance association with the target.
- **Ranking with shadows.** `rank_features()` builds an evaluation shadow block the same way from the evaluation rows. It then measures permutation importance for every real group and for its shadow group.
- **Ranks and consensus unchanged.** Real groups are ranked among themselves only, so ranks and the consensus keep their 0.5 meaning.
- **Report additions,** only when shadows are on; reports without shadows are unchanged:
  - top level: `shadows: true` and `shadow_seed`;
  - each model: `shadow_importance` (per group: `values`, `mean`, `std`) and `shadow_max`, the largest shadow mean;
  - each real group's importance entry: `above_shadow`, true when its mean is strictly greater than `shadow_max`;
  - each consensus row: `above_shadow_share`, the average across voting families of each family's fraction of voting members beating their model's `shadow_max`.
- **`rank_resampled()`.** It passes the shadow settings to every split ranker. Split `i` uses `shadow_seed + i`. Stability rows gain `above_shadow_share`, the mean of the per-split values over ranked splits.
- **`summarize()`.** A new info rule, `below_shadow`: groups whose `above_shadow_share` is below 0.5.

**Report schema:** additive only, and only when `shadows=True`.

## Non-goals

- p-values, binomial tests, or multiple-testing correction. `above_shadow_share` is descriptive, because families and splits are not independent trials.
- Iterative Boruta-style elimination, or several shadow copies per feature.
- A CLI flag for shadows.
- Automatically excluding or selecting features (0.8).

## Capabilities

### New Capabilities

- `shadow-significance`: covers
  - the shadow construction at fit and rank time;
  - shadow importance and the `above_shadow` comparison;
  - the family-weighted `above_shadow_share` in the consensus and in resampled stability;
  - seeding;
  - validation;
  - compatibility with 0.6 pickles.

### Modified Capabilities

- `report-summary` (from the `trust-the-consensus` change): adds the `below_shadow` rule.

## Impact

- **Code:** `featranker/importance.py` (constructor, `fit`, `rank_features`, `_rank_model`, `_build_consensus`, `_build_stability`, `rank_resampled`, `__setstate__`) and `featranker/summary.py`.
- **Cost:** with shadows on, fitting sees twice the columns and permutation importance runs twice as many groups, so expect roughly double the time and memory.
- **Docs:**
  - `docs/usage.md`: a new "Shadow features" section, with a caveat about custom pipelines that drop columns;
  - `docs/report.md`: the new fields and how to read them;
  - `README.md`: a docs-table entry.
- **Tests:** a new `tests/test_shadows.py`, plus additions to the summary and resampled tests.
- **Version:** 0.7.0.
