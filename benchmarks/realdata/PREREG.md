# Real-data study: preregistration (fixed 2026-11-20, before any run)

Question: on real tabular data, does FeatRanker's selection keep held-out performance close to using all features, while dropping features? Descriptive study; no significance test. Every dataset and every outcome is reported, including where selection loses.

## Datasets (rule, then list)

Source suites: OpenML-CC18 (study 99, classification) and the AutoML Benchmark regression suite (study 269), as returned by the OpenML API on 2026-11-20.
Eligible: 500 to 20,000 rows; 10 to 200 features (target excluded); no symbolic feature besides a classification target. Missing values allowed.
Eligible: 26 classification, 4 regression. Chosen: all 4 regression; classification sorted by data_id, every third starting with the first.

| data_id | name | task | rows | features | missing cells |
|---:|---|---|---:|---:|---:|
| 6 | letter | clf | 20000 | 16 | 0 |
| 22 | mfeat-zernike | clf | 2000 | 47 | 0 |
| 44 | spambase | clf | 4601 | 57 | 0 |
| 458 | analcatdata_authorship | clf | 841 | 70 | 0 |
| 1053 | jm1 | clf | 10885 | 21 | 25 |
| 1067 | kc1 | clf | 2109 | 21 | 0 |
| 1487 | ozone-level-8hr | clf | 2534 | 72 | 0 |
| 1510 | wdbc | clf | 569 | 30 | 0 |
| 40499 | texture | clf | 5500 | 40 | 0 |
| 201 | pol | reg | 15000 | 48 | 0 |
| 216 | elevators | reg | 16599 | 18 | 0 |
| 287 | wine_quality | reg | 6497 | 11 | 0 |
| 42730 | us_crime | reg | 1994 | 126 | 39202 |

No dataset is added or removed after this point. A dataset that fails to load or run is reported as failed with its error, not replaced.

## Protocol

- Outer split: 70/30 train/test, stratified for classification, seeds 0, 1, 2. The test rows are used only for the final scores.
- Scorer: `balanced_accuracy` (classification), `r2` (regression), for ranking and for test scores.
- Preprocessing: median imputation where a dataset has missing cells (FeatRanker with `missing="impute"`; downstream pipelines start with `SimpleImputer(strategy="median")`).
- Arms (all fit on the outer training rows only):
  - `all`: every feature.
  - `fr_share`: `ConsensusSelector(group="fast", min_shadow_share=0.5, cv=5, n_repeats=5, random_state=seed, max_samples=5000, max_train_samples=5000)`.
  - `fr_half`: same selector with `k` = half the consensus groups (rounded up) and no shadow rule.
  - `rf_median`: `SelectFromModel(RandomForest(200 trees, random_state=seed), threshold="median")`, a common single-model comparator.
- Downstream models, each refit on the arm's columns: logistic regression or ridge (with `StandardScaler`), `RandomForest` (200 trees), `HistGradientBoosting`; `random_state=seed`.
- Recorded per dataset, seed, arm, model: test score, number of features kept, selector fit seconds; failures with error type and message.

## Outcomes (descriptive)

- Per dataset and model: mean over seeds of (arm score - `all` score), with the range, and mean share of features kept.
- Counts: datasets where an arm is within 0.01 of `all` (mean), better by more than 0.01, worse by more than 0.01.
- No arm is declared a winner by a test; small differences on three seeds are noise.

## Environment

FeatRanker 1.22.0 (commit 482b62d) from a frozen copy, the benchmark venv of docs/benchmark.md (scikit-learn 1.5.1, NumPy 1.26.4), `group="fast"` uses only scikit-learn models.
