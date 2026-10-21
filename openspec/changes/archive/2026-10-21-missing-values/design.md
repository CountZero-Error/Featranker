## Context

`_validate_X` casts to float and rejects non-finite values. `init_models()` wraps built-in linear models in `make_pipeline(StandardScaler(), est)` and leaves tree models bare. `_spearman_matrix` (evidence.py) assumes complete data. The dummy baselines ignore feature values (verified: they fit and predict with NaN). See proposal.md.

## Decisions

### D1. One rule for all built-in models
Median `SimpleImputer` first in every built-in pipeline, including models that handle NaN natively (HistGradientBoosting, XGBoost, CatBoost, sklearn trees ≥ 1.4). One predictable rule beats per-model special cases, and it keeps results identical across scikit-learn versions. Cost: native missingness handling is lost; documented as a limitation (custom models can keep it).

### D2. Validation
`_validate_X(..., allow_nan=False)`; with `allow_nan` it rejects only `±inf`. The ranker passes `allow_nan = (missing == "impute")` in `fit`, `rank_features`, `rank_resampled`. Error text for the default path is unchanged.

### D3. Where the imputer goes
`init_models()` builds `make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), est)` for linear and `make_pipeline(SimpleImputer(strategy="median"), est)` for tree models when `missing="impute"`; otherwise exactly as today (so `"error"` reports are byte-identical to 1.6). Shadows: the imputer sees real and shadow columns; shadow columns are permutations of real columns, so their medians equal the real ones.

### D4. Pairwise Spearman
In `_spearman_matrix`, if the input has no NaN, keep the current fast path (identical numbers). Otherwise loop over column pairs, rank each pair's complete rows, and compute the centered correlation; fewer than 30 complete rows → NaN (treated as uncorrelated by both callers). Review measured that with 3 shared rows, independent columns reach |rho| ≥ 0.7 about a third of the time; at 30 rows chance |rho| ≥ 0.5 is below 1 %. O(p²·n); documented.

### D5. SHAP
`SimpleImputer` joins the per-column step set in `_unwrap` (an imputer maps each column independently). With `add_indicator=True` it would add columns, so only `add_indicator=False` qualifies.

### D6. Record
`missing = {"policy": "impute", "strategy": "median", "train_counts": {...}}` from the validated training matrix (real columns only), added in `rank_features()`.

## Risks / Trade-offs

- [Median imputation leaves a spike at the median that tree and kernel models learn, so a feature's importance can mix its values and its missingness; linear models largely cannot use it (review: tree importance about 1.9 vs ridge 0.006 when only missingness drives the target)] → documented; use custom models with `add_indicator=True` or native NaN handling to separate the two, and check `train_counts`.
- [A column missing in every training row is a constant 0 after imputation (`keep_empty_features=True` keeps its position); its importance is meaningless] → documented; `train_counts` shows it.
