## Why

Every FeatRanker result today comes from one training/evaluation split. `rank_std` measures disagreement between model families, not sampling uncertainty, so users cannot tell whether a ranking would survive a different split. The roadmap thesis for 1.0 is "which features matter, **how sure**, and a leakage-safe way to use the answer". This change delivers "how sure".

## What Changes

- New method `FeatureRanker.rank_resampled(X, y, *, scoring, cv=5, ...)`. It splits the data the caller passes with a cross-validation splitter. For each split it builds a fresh ranker with the same settings, fits it on the split's training rows, and ranks on that split's evaluation rows. It then aggregates each group's place in every split's consensus into a stability table.
- `cv` accepts:
  - an integer number of folds, giving a shuffled `StratifiedKFold` for classification or `KFold` for regression;
  - any scikit-learn splitter, such as `RepeatedKFold` or `GroupKFold` with `groups=`;
  - an iterable of `(train_indices, eval_indices)` pairs.
- New report shape with `evaluation_mode: "resampled"`. It holds:
  - every per-split report, unchanged from 0.5;
  - a `stability` list with, per feature group:
    - the median, minimum, and maximum of its competition-ranked place in each split's sorted consensus;
    - its place in each split;
    - `top_k_share`, the fraction of splits in which its place was within `top_k`;
  - per-split failures, tagged with a split index.
- The method does not change the ranker's own fitted state.
- `summarize()` raises a clear `ValueError` for a resampled report and points to summarizing each split's report.

**Report schema:** the single-split report from `rank_features()` does not change. The resampled report is a new, separate shape. Its per-split entries embed 0.5 single-split reports unchanged.

## Non-goals

- A CLI for resampled ranking. The CLI keeps its train/evaluation-file workflow; a resampling CLI can follow.
- Stability rules in `summarize()`, such as flagging unstable top groups. They come after real usage.
- Running splits in parallel. `n_jobs` still parallelizes models within each split.
- Bootstrap confidence intervals or any significance test (0.7).
- Nested cross-validation or feature selection (0.8). This method ranks; it does not choose features.
- Guarding against users passing their final test set. Docs state the rule, but the library cannot detect it.

## Capabilities

### New Capabilities

- `resampled-ranking`: the `rank_resampled` method, covering:
  - splitter handling;
  - per-split isolation, seeding, and failure handling;
  - the resampled report shape;
  - the stability aggregation.

### Modified Capabilities

- `report-summary` (from change `trust-the-consensus`, not yet archived): `summarize` rejects resampled reports with guidance. Specified as an added requirement in this change's `report-summary` delta.

## Impact

- **Code:**
  - `featranker/importance.py`: the new method plus a small stability aggregation helper; it reuses `fit`/`rank_features` on fresh ranker instances.
  - `featranker/summary.py`: the resampled-report guard.
- **Dependencies:** none. Splitters come from scikit-learn.
- **Cost:** roughly `n_splits` times one `fit` plus `rank_features`. Docs state this.
- **Docs:**
  - `docs/usage.md`: a new "Resampled ranking" section, with a leakage note saying to pass development data only, never the final test set.
  - `docs/report.md`: the resampled report structure and how to read stability.
  - `README.md`: one line plus a docs-table entry.
- **Tests:** a new `tests/test_resampled.py`.
- **Version:** 0.6.0.
