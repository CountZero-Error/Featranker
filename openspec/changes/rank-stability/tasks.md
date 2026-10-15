## 1. Stability aggregation

- [x] 1.1 Write unit tests for `_build_stability` using synthetic split reports:
  - the spec's arithmetic scenario (ranks 1, 2, 4 with `top_k=2`);
  - a split with an empty consensus (`null`, not in top k);
  - all splits empty (`[]` plus a `UserWarning`);
  - sort order (`median_rank`, then `top_k_share` descending, then name);
  - failed splits lower `top_k_share` through the `n_splits` denominator.

  Confirm they fail.
- [x] 1.2 Implement `_build_stability(split_reports, n_splits, top_k)` in `featranker/importance.py`. Tests from 1.1 pass.

## 2. Splitter handling

- [x] 2.1 Write tests:
  - an integer `cv` on clf is stratified (both classes in every evaluation fold);
  - an integer `cv` on reg is shuffled and seeded (same folds for the same `random_state`);
  - `GroupKFold` with `groups` keeps groups disjoint;
  - explicit index pairs give the right `n_train`/`n_eval`;
  - `cv=1`, `cv="five"`, an empty side, and a non-positive `top_k` raise `ValueError` before any fit.

  Confirm they fail.
- [x] 2.2 Implement `cv` normalization into materialized index pairs, with validation. Tests from 2.1 pass.

## 3. rank_resampled

- [x] 3.1 Write tests:
  - a fitted ranker's state is identical after the call;
  - an unfitted ranker works and stays unfitted;
  - each split ranker is fit only on its training rows (a recording estimator sees exactly those rows);
  - the per-split seed is `random_state + i`;
  - two calls are reproducible;
  - one failing split is recorded with `stage: "split"` and the run continues;
  - all splits failing raises `RuntimeError`;
  - per-model failures carry `split`;
  - DataFrame input keeps column names;
  - the report is strict JSON and has every top-level key from the spec;
  - `custom models=` works.

  Confirm they fail.
- [x] 3.2 Implement `FeatureRanker.rank_resampled`. Tests from 3.1 pass.
- [x] 3.3 Add an end-to-end test on a public dataset (diabetes, `cv=3`, `group="tree"`) asserting that the stability rows cover every feature group and that the top group's `top_k_share` lies in [0, 1].

## 4. summarize guard

- [x] 4.1 Write a test: `summarize` on a resampled report raises a `ValueError` mentioning `splits`. Confirm it fails.
- [x] 4.2 Implement the guard before the required-key check in `featranker/summary.py`.

## 5. Docs

- [x] 5.1 `docs/usage.md`:
  - add a "Resampled ranking" section with a signature table, an example, the cost formula, choosing `top_k`, and a leakage note;
  - list `rank_resampled` in the API intro.
- [x] 5.2 `docs/report.md`:
  - add the resampled report structure;
  - explain how to read `stability` (`median_rank` vs the min/max range, and `top_k_share`);
  - state that `rank_std` within a split is still family disagreement.
- [x] 5.3 `README.md`: one line in "How it works" and the Usage docs-table entry. `docs/README.md` index entry.
- [x] 5.4 Verify every new doc claim against the code. Render the changed docs through the GitHub markdown API and check anchors and escaped tags.

## 6. Release prep

- [x] 6.1 Bump to 0.6.0. Add a CHANGELOG entry.
- [x] 6.2 Run the full suite outside the sandbox; everything passes.
