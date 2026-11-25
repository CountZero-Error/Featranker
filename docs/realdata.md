# FeatRanker ▍ Real-data study

<sub><a href="README.md">◂ Back to docs</a></sub>

Does FeatRanker's selection keep held-out performance close to using all features while dropping features, on real tabular data? This is a descriptive study, fixed in [`PREREG.md`](https://github.com/CountZero-Error/Featranker/blob/main/benchmarks/realdata/PREREG.md) on 2026-11-20 before any run: no significance test, and every dataset and outcome is reported, including where selection loses. The code is [`benchmarks/realdata/run.py`](https://github.com/CountZero-Error/Featranker/tree/main/benchmarks/realdata) (not part of the package, not run by CI). Every number below comes from `benchmarks/results/realdata/realdata-20261120-final.json` (rendered as `realdata-final.md`). FeatRanker 1.22.0 (an internal build; first released as part of 2.0), scikit-learn 1.5.1, NumPy 1.26.4.

## Short version

- **`fr_share` (`min_shadow_share=0.5`) is conservative to a fault on this data.** It kept very few features on several datasets (share kept: ozone-level-8hr 0.01, us_crime 0.05, analcatdata_authorship 0.09, wdbc 0.10, kc1 0.11) and lost held-out score on most of them (on kc1 only `HistGradientBoosting` lost more than 0.01). Of 39 (dataset, model) pairs it was within 0.01 of all features in 20, worse by more than 0.01 in 19, and **never better by more than 0.01**.
- **`fr_half` (keep half the groups, no shadow rule) and `rf_median` mostly stayed within 0.01** of all features: 29 and 25 of 39 pairs.
- **Letter is the reverse.** `fr_share` kept every feature (share 1.00, delta 0), while `fr_half` and `rf_median`, which cut to half, lost about 0.10 with the linear model (-0.1052 and -0.1000).
- **Selection bought fewer features, not accuracy.** Only one of 117 pairs was better than all features by more than 0.01 (`rf_median`, kc1, `HistGradientBoosting`).

## Datasets

13 datasets: 9 classification and 4 regression, from OpenML-CC18 and the AutoML Benchmark regression suite, 569 to 20,000 rows, 11 to 126 features, chosen by a rule fixed before any run (all 4 regression datasets; every third eligible classification dataset by data id). The full list, with OpenML ids, is in [`PREREG.md`](https://github.com/CountZero-Error/Featranker/blob/main/benchmarks/realdata/PREREG.md). All 13 loaded with the expected shape.

## Protocol

- Outer split 70/30 train/test, stratified for classification, seeds 0, 1, 2. Test rows are used only for the final scores. Scorer: `balanced_accuracy` (classification), `r2` (regression), for ranking and for test scores.
- Median imputation (FeatRanker `missing="impute"`; downstream pipelines start with `SimpleImputer(strategy="median")`).
- Arms, all fit on the training rows only: `all` (every feature); `fr_share` = `ConsensusSelector(group="fast", min_shadow_share=0.5, cv=5, n_repeats=5, max_samples=5000, max_train_samples=5000)`; `fr_half` = the same selector with `k` = half the consensus groups (rounded up) and no shadow rule; `rf_median` = `SelectFromModel(RandomForest(200 trees), threshold="median")`.
- Downstream models, each refit on the arm's columns: logistic regression or ridge (with `StandardScaler`), random forest (200 trees), `HistGradientBoosting`.
- Outcome: per dataset and model, the mean over seeds of (arm score minus `all` score), its range, and the share of features kept. A pair is "within" if the mean delta is within 0.01.

## Deviations

Recorded in [`DEVIATIONS.md`](https://github.com/CountZero-Error/Featranker/blob/main/benchmarks/realdata/DEVIATIONS.md), items 1 to 8 before any run and item 9 after the first run: (1) `shadow_seed=seed` for reproducibility; (2) `n_jobs=-1`, speed only; (3) `LogisticRegression(max_iter=2000)`; (4) an empty selection is recorded as `EmptySelection` and scored with a dummy model; (5) median imputation in every pipeline, the identity where nothing is missing; (6) counts are over (dataset, model) pairs, not datasets; (7) a dataset that fails to load is reported as failed and not retried by `--resume`; (8) the script records the imported source hash and checks shapes; (9) in the first run four datasets (letter, jm1, spambase, ozone-level-8hr) failed to download with transient network errors, so they were run again unchanged into a second file. The final tables use the second run's rows for those four and the first run's rows for the other nine; both files are kept. No dataset failed in the final tables, and no empty selection appears in them.

## Results

### Counts

(dataset, model) pairs, 39 per arm (13 datasets x 3 models); none excluded:

| arm | within 0.01 | better | worse |
|---|---:|---:|---:|
| fr_share | 20 | 0 | 19 |
| fr_half | 29 | 0 | 10 |
| rf_median | 25 | 1 | 13 |

The one "better" pair is `rf_median` with `HistGradientBoosting` on kc1 (+0.0185).

### Per dataset

Mean delta against `all` over 3 seeds, with [min, max]. Negative means the arm scored lower on the test rows. Classification deltas are balanced accuracy, regression deltas are r2. The share kept is the mean fraction of features the arm kept.

#### `fr_share`

| dataset | share kept | linear / ridge | random forest | hist GB |
|---|---:|---|---|---|
| mfeat-zernike | 0.27 | -0.0506 [-0.0583, -0.0383] | -0.0250 [-0.0433, -0.0050] | -0.0483 [-0.0533, -0.0450] |
| analcatdata_authorship | 0.09 | -0.1041 [-0.1852, -0.0402] | -0.0789 [-0.1627, -0.0221] | -0.0941 [-0.1546, -0.0592] |
| kc1 | 0.11 | -0.0059 [-0.0144, -0.0001] | -0.0077 [-0.0201, +0.0170] | -0.0288 [-0.0451, +0.0018] |
| wdbc | 0.10 | -0.0322 [-0.0531, -0.0156] | -0.0151 [-0.0296, +0.0000] | -0.0338 [-0.0546, -0.0093] |
| texture | 0.93 | +0.0002 [+0.0000, +0.0006] | -0.0012 [-0.0030, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| pol | 0.35 | -0.0003 [-0.0007, +0.0000] | +0.0001 [-0.0000, +0.0002] | +0.0003 [+0.0003, +0.0004] |
| elevators | 0.80 | -0.0002 [-0.0007, +0.0002] | +0.0004 [-0.0007, +0.0015] | -0.0007 [-0.0031, +0.0029] |
| wine_quality | 1.00 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| us_crime | 0.05 | -0.0354 [-0.0600, -0.0227] | -0.0416 [-0.0603, -0.0158] | -0.0563 [-0.0850, -0.0247] |
| letter | 1.00 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| spambase | 0.23 | -0.0224 [-0.0288, -0.0120] | -0.0148 [-0.0302, -0.0029] | -0.0198 [-0.0302, -0.0133] |
| jm1 | 0.30 | -0.0070 [-0.0154, -0.0020] | -0.0054 [-0.0067, -0.0036] | -0.0050 [-0.0124, +0.0018] |
| ozone-level-8hr | 0.01 | -0.1353 [-0.1784, -0.0965] | -0.0218 [-0.0722, +0.0187] | -0.1185 [-0.1430, -0.0708] |

#### `fr_half`

| dataset | share kept | linear / ridge | random forest | hist GB |
|---|---:|---|---|---|
| mfeat-zernike | 0.51 | -0.0039 [-0.0100, +0.0017] | -0.0006 [-0.0050, +0.0067] | -0.0106 [-0.0150, -0.0067] |
| analcatdata_authorship | 0.50 | -0.0096 [-0.0201, +0.0022] | -0.0009 [-0.0028, +0.0000] | -0.0050 [-0.0147, +0.0026] |
| kc1 | 0.52 | -0.0135 [-0.0274, -0.0019] | -0.0173 [-0.0334, -0.0065] | -0.0016 [-0.0149, +0.0204] |
| wdbc | 0.50 | -0.0067 [-0.0218, +0.0078] | +0.0068 [+0.0000, +0.0203] | +0.0047 [-0.0047, +0.0156] |
| texture | 0.50 | -0.0024 [-0.0061, -0.0006] | -0.0008 [-0.0036, +0.0012] | -0.0004 [-0.0012, +0.0012] |
| pol | 0.50 | +0.0000 [-0.0001, +0.0002] | +0.0000 [+0.0000, +0.0000] | -0.0001 [-0.0001, -0.0001] |
| elevators | 0.50 | -0.0014 [-0.0015, -0.0013] | +0.0083 [+0.0050, +0.0110] | -0.0005 [-0.0040, +0.0031] |
| wine_quality | 0.55 | -0.0032 [-0.0041, -0.0021] | -0.0245 [-0.0313, -0.0155] | -0.0313 [-0.0415, -0.0250] |
| us_crime | 0.50 | -0.0003 [-0.0075, +0.0041] | +0.0010 [-0.0039, +0.0052] | +0.0087 [+0.0022, +0.0189] |
| letter | 0.50 | -0.1052 [-0.1096, -0.0999] | -0.0387 [-0.0411, -0.0353] | -0.0426 [-0.0459, -0.0385] |
| spambase | 0.51 | -0.0114 [-0.0183, -0.0023] | +0.0015 [-0.0034, +0.0044] | -0.0057 [-0.0073, -0.0036] |
| jm1 | 0.52 | -0.0049 [-0.0065, -0.0034] | -0.0008 [-0.0086, +0.0053] | -0.0014 [-0.0040, +0.0022] |
| ozone-level-8hr | 0.50 | -0.0641 [-0.0701, -0.0597] | -0.0030 [-0.0097, +0.0007] | -0.0012 [-0.0236, +0.0187] |

#### `rf_median`

| dataset | share kept | linear / ridge | random forest | hist GB |
|---|---:|---|---|---|
| mfeat-zernike | 0.51 | -0.0044 [-0.0100, +0.0050] | -0.0106 [-0.0183, -0.0067] | -0.0117 [-0.0167, -0.0017] |
| analcatdata_authorship | 0.50 | -0.0038 [-0.0109, +0.0048] | -0.0043 [-0.0074, +0.0000] | -0.0050 [-0.0175, +0.0048] |
| kc1 | 0.52 | -0.0133 [-0.0181, -0.0056] | -0.0167 [-0.0306, -0.0005] | +0.0185 [+0.0051, +0.0324] |
| wdbc | 0.50 | -0.0239 [-0.0328, -0.0156] | -0.0083 [-0.0156, -0.0047] | -0.0119 [-0.0390, +0.0078] |
| texture | 0.50 | -0.0057 [-0.0067, -0.0042] | -0.0051 [-0.0079, -0.0030] | -0.0028 [-0.0048, -0.0012] |
| pol | 0.50 | +0.0000 [-0.0001, +0.0002] | -0.0000 [-0.0001, +0.0000] | -0.0001 [-0.0003, +0.0001] |
| elevators | 0.50 | -0.0063 [-0.0156, -0.0015] | +0.0058 [+0.0046, +0.0077] | -0.0053 [-0.0114, -0.0006] |
| wine_quality | 0.55 | -0.0032 [-0.0041, -0.0021] | -0.0245 [-0.0313, -0.0155] | -0.0313 [-0.0415, -0.0250] |
| us_crime | 0.50 | -0.0058 [-0.0146, +0.0035] | -0.0003 [-0.0031, +0.0051] | +0.0048 [-0.0104, +0.0163] |
| letter | 0.50 | -0.1000 [-0.1044, -0.0966] | -0.0243 [-0.0252, -0.0237] | -0.0268 [-0.0273, -0.0265] |
| spambase | 0.51 | -0.0125 [-0.0141, -0.0098] | -0.0005 [-0.0036, +0.0016] | -0.0046 [-0.0094, +0.0000] |
| jm1 | 0.52 | -0.0016 [-0.0024, -0.0009] | -0.0056 [-0.0082, -0.0022] | -0.0059 [-0.0116, +0.0012] |
| ozone-level-8hr | 0.50 | -0.0271 [-0.0514, -0.0111] | +0.0095 [-0.0021, +0.0208] | +0.0035 [-0.0097, +0.0208] |

## Findings

- **The shadow-share rule dropped real signal.** On ozone-level-8hr (0.01 kept) the mean delta was -0.1353 with the linear model and -0.1185 with `HistGradientBoosting`; on analcatdata_authorship (0.09 kept) -0.1041 and -0.0941; on us_crime (0.05 kept) -0.0354 to -0.0563; on wdbc (0.10 kept) -0.0151 to -0.0338; on kc1 (0.11 kept) `HistGradientBoosting` -0.0288, while the linear model (-0.0059) and the random forest (-0.0077) stayed within 0.01. The random forest was hurt least on ozone-level-8hr (-0.0218).
- **It is not a uniform loss.** On texture, pol, elevators, wine_quality and letter it kept 0.35 to 1.00 of the features and stayed within 0.002 of all features (mean delta) in every model.
- **Letter shows that the other arms can lose where the shadow rule does not.** `fr_half` and `rf_median` cut half of 16 features and lost -0.1052 and -0.1000 (linear), -0.0387 and -0.0243 (random forest), -0.0426 and -0.0268 (`HistGradientBoosting`). On wine_quality (11 features) both lost -0.0245 (random forest) and -0.0313 (`HistGradientBoosting`) while `fr_share` kept everything.
- **Where half the features were enough, `fr_half` and `rf_median` stayed close.** Their losses beyond 0.01 are concentrated in letter, wine_quality, ozone-level-8hr (linear: -0.0641 for `fr_half`), kc1 (`fr_half`: linear -0.0135, random forest -0.0173; `rf_median`: linear -0.0133, random forest -0.0167), wdbc (`rf_median`: 2 pairs, worst -0.0239 linear), mfeat-zernike (`rf_median`: 2 pairs) and a few single pairs.

### What this means for users

- `min_shadow_share=0.5` is a strict filter, and on real data it can keep far fewer features than the signal needs. Do not use it as a default way to shrink a feature set.
- Choose the subset by held-out score instead: use the [ablation curve](usage.md#ablation-curve) to pick `k`, make that choice on training or validation rows only, and compare against all features once on rows nobody has touched (the [10-minute recipe](../README.md#-10-minute-recipe) does this; this study itself is an outer 70/30 train/test split per seed with `ConsensusSelector` cross-validating inside the training part, not nested cross-validation). Include all features (the largest `k`) in that curve, so the subset-versus-all decision is also made before the final held-out rows are used.
- Treat the shadow rule as evidence that a group is distinguishable from noise, not as proof that the groups it drops are useless.

### What this does not show

- Only 3 seeds and 13 datasets; small differences are noise, and no test was run.
- Numeric features only, with one fixed selector setting: `group="fast"`, `min_shadow_share=0.5`. Other thresholds, `max_shadow_q`, `min_shadow_families`, and `group="all"` were not studied.
- It does not show that `fr_half` or `rf_median` are better rules: `fr_half` was given a free choice of `k` (half), and only one of the 117 pairs beat all features by more than 0.01: `rf_median` with `HistGradientBoosting` on kc1 (+0.0185).
- It measures held-out score, not whether the dropped features are causally or scientifically relevant.

## Reproduce

Use the benchmark environment of [benchmark.md](benchmark.md#reproduce) (scikit-learn 1.5.1, NumPy 1.26.4) and run from a frozen copy of the 1.22.0 source (commit `482b62d`) put on `PYTHONPATH`, so that the code imported is the code recorded:

```bash
PYTHONPATH=<frozen-copy> python benchmarks/realdata/run.py --data-home <openml-cache> --commit 482b62d --out realdata.json
PYTHONPATH=<frozen-copy> python benchmarks/realdata/run.py --data-home <openml-cache> --commit 482b62d --out realdata.json --resume   # continue an interrupted run
python benchmarks/realdata/run.py --render realdata.json                                                                      # print the tables
```

A run holds the exclusive lock file `<out>.lock` beside the resolved `--out` until it ends, also on errors; a second run on the same file exits with an error naming the lock (delete it only if no run is active). `--render` takes no lock. `--resume` skips dataset-and-seed pairs already in `--out`. `--resume` also refuses a different FeatRanker version or commit (`--commit ''` is a value, not an omission). Duplicate seeds or dataset ids are refused, and each save writes a temporary file beside `--out` and replaces `--out` with it, so a failed save leaves the previous file. A dataset that failed to load is written as error rows and is not retried by `--resume`. To retry failed downloads, as in deviation 9: delete the partial files for those datasets from the data cache, run only those datasets again with `--datasets <ids>` into a second `--out` file, and merge by keeping the second file's rows for those datasets. Keep both files and record the step. The run needs network access to OpenML on the first run.
