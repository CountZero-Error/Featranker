# FeatRanker memory and wall-time study (HEAD 40ad82a, v1.16.0)

Question: does large tabular data (wide / mixed / high-cardinality) need dedicated large-data work?
No package code was changed. Measured on a frozen `git archive HEAD featranker` copy (`featranker.__file__` is asserted to be under it).
Machine: 16 GB-class MacBook Air, Python 3.x (anaconda), numpy 1.26.4, scikit-learn 1.5.1, pandas 2.2.2, shap 0.52. Other work (another agent) ran on the machine throughout.

## Method

- `measure.py` runs ONE configuration in a fresh subprocess and records `ru_maxrss` (bytes on macOS) = peak RSS, plus current RSS before fit / after fit. `run_grid.py` is the driver; `table.py` renders `results.jsonl`; raw rows are in `results.jsonl`.
- Call: `FeatureRanker("clf", group="fast", verbose=False, shadows=..., evidence=..., categorical=[...]).fit(X, y, feature_names=...).rank_features(X_eval, y_eval, scoring="roc_auc", n_repeats=3, random_state=0, max_samples=...)`, `n_jobs=None`. group="fast" = 7 models (logistic, ridge classifier, LDA, GaussianNB, decision tree, random forest, HistGradientBoosting).
- Data: standard-normal float64; binary target. Train and eval arrays are both `rows` x `cols` (independent draws), so "input MB" = train + eval (train alone is half). Mixed = DataFrame with 50 float columns + 10 string categorical columns (shared string objects), input counted as numeric bytes + 8 B per categorical cell. "RSS before fit" = interpreter + sklearn imports (~175 MB) + the input. Peak/input ratios are therefore inflated for tiny inputs; `(peak - before)/input` is the framework's own multiplier.
- Three row kinds in the table:
  - plain names (`numeric-...`, `mixed-...`, `autogroups-...`): full end-to-end runs that finished within the timeout.
  - `probe-...`: harness patch scores ONE group x ONE repeat (`_resolve_feature_groups` monkeypatched inside the measuring process only). Peak RSS and fit seconds are real; rank seconds are not, so "est. full rank s" extrapolates (probe rank time minus evidence time) x (full scorings / probe scorings). This is an estimate, not a measurement. Status `timeout_*s` rows: fit had not finished; peak shown is a lower bound (parent-polled RSS, `>=`).
  - `overhead-...`: same probe, but the only model is GaussianNB, so the numbers are the FeatRanker framework's own copies (validation, shadows, evidence) with the estimator cost removed. Its "est. full rank s" column is blank on purpose (NB is not representative).
- Guards: 8 GB RSS kill, per-config timeout 120 s (180 s with evidence), dense one-hot estimate above 4 GB per copy is skipped (not attempted).

## Honesty notes

- **Wall times are noisy and mostly inflated.** The grid ran in two parallel lanes plus the other agent's load; HistGradientBoosting/BLAS oversubscribed. The only serial, clean calibration (10k x 20 numeric): fit 2.6 s + rank 5.6 s (shadows: 4.4 + 12.0 s; evidence: 2.9 + 22.1 s, of which SHAP 16.2 s). The same config later measured 3.7 / 8.4 s and shadows 21 / 22 s under contention. Treat fit/rank seconds in the table as upper-ish figures; relative scaling is still visible.
- I cut the grid: the first pass (full runs) timed out for most 100k rows and wide cases, so I switched to probes. Not measured end to end: any 100k-row case with >=100 columns (rank), anything at 500k rows, widest + shadows/evidence. 500k x 20 never finished fit in 120 s.
- n_jobs>1 (joblib) was not measured.
- Skipped as obviously over budget: mixed 10k rows x card 10000 (dense one-hot 7.5 GB per copy, 14.9 GB with shadows), mixed 100k x card 1000 (7.5 GB per copy). Substituted a scaled probe: 2k rows x card 10000.
- Timeouts for 100k x 500 stopped inside fit, so peak RSS there (3.6-4.8 GB) is a lower bound.

## Results (all rows; MB, seconds)

| config | status | input MB (train+eval) | RSS before fit MB | peak RSS MB | peak/input | (peak-before)/input | fit s | rank s (measured) | est. full rank s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| autogroups-10k-c500 | ok | 76.3 | 248 | 576 | 7.5 | 4.3 |  |  |  |
| autogroups-100k-c500 | ok | 762.9 | 939 | 3595 | 4.7 | 3.5 |  |  |  |
| numeric-10k-c20 | ok | 3.1 | 176 | 274 | 88.4 | 31.6 | 3.69 | 8.41 |  |
| mixed-10k-card10 | ok | 9.2 | 182 | 533 | 57.9 | 38.2 | 6.71 | 35.8 |  |
| numeric-10k-c100 | ok | 15.3 | 187 | 440 | 28.8 | 16.5 | 14.01 | 81.53 |  |
| mixed-10k-card10-ms5000 | ok | 9.2 | 184 | 549 | 59.7 | 39.7 | 56.57 | 17.09 |  |
| mixed-10k-card10000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| mixed-10k-card10000-shadows | skipped: dense one-hot estimate 14.9 GB per copy | | | | | | | | |
| mixed-10k-card10000-ms5000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| numeric-10k-c20-shadows | ok | 3.1 | 176 | 326 | 105.2 | 48.4 | 21.43 | 21.54 |  |
| probe-mixed-10k-card10-shadows | ok | 9.2 | 183 | 623 | 67.7 | 47.8 | 72.77 | 1.09 | 131 |
| probe-numeric-10k-c500 | timeout_120s |  |  | 708 (>=, in fit) |  |  |  |  |  |
| probe-mixed-10k-card1000 | timeout_120s |  |  | 6131 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c500-shadows | timeout_120s |  |  | 996 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c100-shadows | ok | 15.3 | 185 | 382 | 25.0 | 12.9 | 69.28 | 2.16 | 433 |
| probe-mixed-10k-card1000-shadows | timeout_120s |  |  | 7377 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c100-evidence | ok | 15.3 | 187 | 548 | 35.8 | 23.6 | 14.49 | 56.45 | 75 |
| probe-mixed-10k-card1000-ms5000 | timeout_120s |  |  | 6203 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c500-evidence | ok | 76.3 | 251 | 1041 | 13.6 | 10.4 | 73.55 | 39.1 | 638 |
| probe-mixed-100k-card10 | ok | 91.6 | 279 | 1978 | 21.6 | 18.5 | 70.92 | 7.06 | 639 |
| probe-numeric-100k-c20 | ok | 30.5 | 206 | 680 | 22.3 | 15.5 | 39.46 | 2.89 | 88 |
| probe-numeric-100k-c20-shadows | ok | 30.5 | 204 | 826 | 27.1 | 20.4 | 60.23 | 4.67 | 188 |
| probe-mixed-100k-card10-shadows | timeout_120s |  |  | 3004 (>=, in rank) |  |  |  |  |  |
| probe-numeric-100k-c20-ms5000 | ok | 30.5 | 205 | 651 | 21.3 | 14.6 | 37.68 | 0.33 | 10 |
| probe-mixed-100k-card10-ms5000 | ok | 91.6 | 276 | 1908 | 20.8 | 17.8 | 71.68 | 3.09 | 280 |
| probe-mixed-100k-card1000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| probe-mixed-10k-card10000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| probe-mixed-2k-card10000 | ok | 1.8 | 177 | 2667 | 1481.7 | 1383.3 | 55.77 | 0.8 | 72 |
| probe-numeric-100k-c20-shadows-evidence-ms5000 | timeout_180s |  |  | 1042 (>=, in rank) |  |  |  |  |  |
| probe-mixed-2k-card10000-shadows | timeout_120s |  |  | 3968 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100 | timeout_120s |  |  | 1013 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100-shadows | timeout_120s |  |  | 1925 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100-ms5000 | ok | 152.6 | 328 | 1352 | 8.9 | 6.7 | 112.12 | 0.35 | 53 |
| probe-numeric-100k-c500 | timeout_120s |  |  | 3555 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c500-ms5000 | timeout_120s |  |  | 3921 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c500-shadows-evidence-ms5000 | timeout_180s |  |  | 4829 (>=, in fit) |  |  |  |  |  |
| probe-numeric-500k-c20 | timeout_120s |  |  | 1630 (>=, in fit) |  |  |  |  |  |
| probe-numeric-500k-c20-ms5000 | timeout_120s |  |  | 1671 (>=, in fit) |  |  |  |  |  |
| overhead-numeric-100k-c100 | ok | 152.6 | 328 | 710 | 4.7 | 2.5 | 0.19 | 0.23 |  |
| overhead-numeric-100k-c100-shadows | ok | 152.6 | 327 | 1214 | 8.0 | 5.8 | 0.2 | 0.87 |  |
| overhead-numeric-100k-c500 | ok | 762.9 | 939 | 2707 | 3.5 | 2.3 | 0.76 | 1.01 |  |
| overhead-numeric-100k-c500-shadows | ok | 762.9 | 937 | 4664 | 6.1 | 4.9 | 1.67 | 2.9 |  |
| overhead-numeric-100k-c500-ms5000 | ok | 762.9 | 936 | 2033 | 2.7 | 1.4 | 0.54 | 0.42 |  |
| overhead-numeric-100k-c500-shadows-ms5000 | ok | 762.9 | 940 | 3292 | 4.3 | 3.1 | 1.08 | 0.64 |  |
| overhead-numeric-100k-c500-shadows-evidence-ms5000 | ok | 762.9 | 939 | 5933 | 7.8 | 6.5 | 1.03 | 6.64 |  |
| overhead-numeric-100k-c500-evidence | ok | 762.9 | 937 | 4803 | 6.3 | 5.1 | 0.7 | 7.99 |  |
| overhead-numeric-500k-c20 | ok | 152.6 | 341 | 846 | 5.5 | 3.3 | 0.25 | 0.62 |  |
| overhead-numeric-500k-c20-shadows | ok | 152.6 | 353 | 1330 | 8.7 | 6.4 | 0.32 | 0.96 |  |
| overhead-mixed-100k-card10 | ok | 91.6 | 287 | 779 | 8.5 | 5.4 | 4.54 | 2.94 |  |
| overhead-mixed-100k-card10-shadows | ok | 91.6 | 286 | 1062 | 11.6 | 8.5 | 3.2 | 3.73 |  |
| overhead-mixed-10k-card1000 | ok | 9.2 | 183 | 242 | 26.3 | 6.4 | 0.27 | 0.24 |  |
| overhead-mixed-10k-card1000-shadows | ok | 9.2 | 184 | 296 | 32.2 | 12.2 | 0.28 | 0.24 |  |

## Headline numbers

| case | input (train+eval) | peak RSS | note |
|---|---:|---:|---|
| numeric 100k x 500, framework only (NB), no shadows | 763 MB | 2.7 GB | 3.5x input |
| same, shadows | 763 MB | 4.7 GB | 6.1x |
| same, evidence | 763 MB | 4.8 GB | `correlation_evidence` alone +1.6 GB, 6.2 s |
| same, shadows + evidence + max_samples=5000 | 763 MB | 5.9 GB | 7.8x; max_samples does not help evidence/shadows |
| numeric 100k x 500, real fast models, base | 763 MB | >= 3.6 GB | fit not finished in 120 s |
| numeric 100k x 500, real models, shadows+evidence+ms5000 | 763 MB | >= 4.8 GB | fit not finished in 180 s |
| `auto_groups` 100k x 500 | 381 MB (X only) | 3.6 GB | 6.9 s; 10k x 500: 0.7 s, 0.58 GB |
| mixed 100k x (50 num + 10 cat x card 10), real models | 92 MB | 1.98 GB | 21x input; NB-only framework overhead 0.78 GB |
| mixed 10k x card 1000, built-in linear models | 9 MB | >= 6.1 GB (7.4 GB shadows) | fit not finished; one dense one-hot = 10k x 10k x 8 B = 800 MB |
| mixed 2k x card 10000 | 1.8 MB | 2.7 GB (>= 4.0 GB shadows) | 100k one-hot columns x 2k rows |
| mixed 10k x card 10000 | - | not run | estimated 7.5 GB per dense copy |

## Where the memory goes (HEAD code, confirmed by event logs where noted)

Let X = one `rows x cols` float64 matrix.

1. **Validation copies (1X each, train and eval).** `_validate_X` always copies (`np.array` / `to_numpy(copy=True)`) plus a temporary boolean from `np.isfinite` (~X/8). Measured: +429 MB for a 381 MB X at 100k x 500. `_validate_y` / weights also copy (small).
2. **Shadow doubling (3X transient, 2X retained).** `_with_shadows` does `X[idx]` (1X) then `np.hstack` (2X): measured +1144 MB for X=381 MB, i.e. 3X peak on top of the validated X. This happens in fit() and again in rank_features() (eval side). With `evidence=True` the 2X `train_X_` stays on the ranker.
3. **Per-model scratch.** `_group_importances` allocates `X_values.copy()` (1X of the scored matrix, 2X with shadows) per model, one model at a time. `max_samples` shrinks only this and the scoring rows (measured: 100k x 500 framework-only peak 2.7 GB -> 2.0 GB; with shadows 4.7 -> 3.3 GB). The fit-side validation and shadow doubling still see every training row, so the peak moves to fit and stays at 3.3 GB.
4. **Estimators.** Real models add on top: RandomForest/DecisionTree cast to float32 and sort (the 100k x 500 real-model peak is >= 3.6 GB vs 2.7 GB framework-only; 500k x 20: >= 1.6 GB vs 0.85 GB). Tree fits at 100k rows also dominate wall time (many did not finish within 120 s).
5. **Evidence.** `correlation_evidence` -> `_spearman_matrix`: `rankdata(X, axis=0)`, centering, and `centered**2` are each ~1X float64 plus argsort temporaries; measured +1.6 GB for 381 MB X and +6 s. SHAP is capped at 1000 rows (+0.1 GB, 0.7 s for NB; 16 s for the 7 real models at 10k x 20). `auto_groups` runs the same code on all rows: 3.6 GB peak at 100k x 500 (X validated copy + rank/centered temporaries ~ 7X). The p x p matrix itself is small (500^2 = 2 MB); the O(columns^2) cost is the matmul and pair loop, not memory.
6. **Categorical as object array.** With `categorical=`, `_validate_X` builds an object array: each numeric cell becomes a Python float (~32 B vs 8 B). Measured at 100k x 60: +287 MB for a 46 MB train input (~6x), then `_encode_categorical` builds a float copy. `_fingerprint` of object columns uses `repr(tolist())` chunks, which was the slowest framework step (3.1 s fit + 1.8 s eval at 100k rows).
7. **Dense one-hot for built-in linear models (dominant for high cardinality).** `_with_one_hot` uses `sparse_threshold=0`, so every fit and every one of the (groups x repeats) scoring calls builds a dense `rows x (n_cat x card)` float matrix, then stacks with the remainder and scalers/LR/LDA copy it again. Measured 6.1-7.4 GB peaks at only 10k rows x card 1000; custom models (and NB-only) skip this entirely (242 MB framework-only at the same shape). Shadows double the categorical columns, so one-hot width doubles.
8. **joblib.** Not measured (n_jobs=None only). From code: `_run_models` passes `X_models` (full, shadow-doubled) in every job tuple; joblib memmaps large arrays, but a per-process 1X scratch copy plus estimator copies would add per worker.

Wall time structure: rank cost scales as models x groups x n_repeats x (1+shadows) x scoring-rows. Estimated full rank (probe extrapolation): 100k x 20 = ~90 s, with max_samples=5000 ~10 s (measured probe rank 2.9 s -> 0.33 s, ~9x). Measured full run 10k x 100: rank 82 s (contended). Wide/long cases are time-bound before they are memory-bound on this machine, except for high-cardinality categoricals (memory-bound) and evidence on wide X.

## Ranked recommendations

1. **Sparse (or index-based) one-hot for high-cardinality categoricals** — highest payoff. Today a 10k-row, 10-column, card-1000 problem needs 6-7 GB and card 10000 is impossible on this machine at 10k rows. Keeping `OneHotEncoder` sparse (only LogisticRegression/Ridge/LDA-style solvers that accept sparse; densify or skip the ones that do not), or capping/bucketing levels, turns an n x (cat x card) x 8 B matrix into n x cat nonzeros: roughly 100-1000x less for card 1000-10000. Also stop rebuilding it per permutation scoring if a cached transform can be reused for unpermuted columns.
2. **Cut the avoidable copies in validation/shadows/evidence.** (a) Validate once without the unconditional second copy where the input is already a finite float64 ndarray (the code copies deliberately to avoid aliasing; a read-only view or documented ownership could remove 1X per call). (b) Build shadows into one preallocated `rows x 2p` array (fill the second half by `np.take(..., out=)`) instead of gather + hstack: peak 3X -> 2X, ~0.76 GB saved at 100k x 500 per call. (c) Compute Spearman with chunked column blocks (or on a row subsample, as SHAP already is at 1000 rows): removes the ~4X temporaries (1.6 GB at 100k x 500) and the 3.6 GB `auto_groups` peak; evidence/auto_groups need ranks of a sample, not all 100k rows. Estimated saving 1-2.5 GB at 100k x 500, no model-quality impact for the fit.
3. **Let `max_samples`-style capping also cover training rows / evidence rows, and consider float32.** At 100k x 500 the peak after `max_samples=5000` is still 3.3 GB because fit and shadows see every row, and tree fits at 100k rows exceed 2 minutes. An optional `max_train_samples` (or documenting that users pre-sample) cuts fit time and memory ~linearly (100k -> 10k rows is ~10x). float32 inputs (optional) would halve every X copy but changes numerics/fingerprints, so it is a lower-priority, opt-in change. Joblib memory and 500k-row behaviour should be measured before any further large-data work.

No evidence here that general large-data infrastructure (out-of-core, chunked estimators) is needed: numeric 100k x 500 stays under ~6 GB even with shadows+evidence on the framework side, and wall time is dominated by the estimators and the groups x repeats loop that users can cut with `max_samples`, `n_repeats`, `feature_groups`/`auto_groups`. The memory cliff is specific to dense one-hot with high cardinality (and secondarily shadows/evidence on very wide data).

## Files

`measure.py`, `run_grid.py`, `table.py`, `results.jsonl` (all rows), `results_full_partial.jsonl` (first-pass copy), `table.md` (same table as below).
Reproduce: `MPLCONFIGDIR=... MEM_PYTHONPATH=<frozen copy> python run_grid.py [numeric- | mixed- | probeA | probeB | overhead]`.
