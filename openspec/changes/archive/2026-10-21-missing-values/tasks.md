## 1. Policy and validation

- [x] 1.1 Test: `missing` validation; default rejects NaN with the same message; `"impute"` accepts NaN in fit/rank_features/rank_resampled, rejects inf; `"error"` reports byte-identical to HEAD; old pickles default to `"error"`
- [x] 1.2 Implement the option and `_validate_X(allow_nan=...)` (D2)

## 2. Imputation and record

- [x] 2.1 Test: every built-in model (clf and reg, all groups incl. fast) is a pipeline starting with a median SimpleImputer under `"impute"`; imputer statistic learned from training rows only (scenario); custom models receive NaN; shadows with NaN work; `missing.train_counts` (scenario); resampled split reports carry it; no key under `"error"`; strict JSON
- [x] 2.2 Implement (D3, D6)

## 3. Evidence and auto groups

- [x] 3.1 Test: pairwise-complete Spearman (scenario) for evidence and `auto_groups`; complete data gives numbers identical to HEAD; fewer than 3 complete rows → uncorrelated; SHAP explains an imputer+scaler+Ridge pipeline as linear and an imputer+forest as tree
- [x] 3.2 Implement (D4, D5)

## 4. Pass-through, docs, checks

- [x] 4.1 Test: selector `missing` round-trips clone/get_params and old pickles; pipeline scenario; CLI `--missing impute` with empty CSV cells; typing consumer
- [x] 4.2 Implement pass-through; version 1.7.0
- [x] 4.3 Docs: docs/usage.md (missing values section: fold-local median imputation, custom models see NaN, informative-missingness limitation, pairwise correlations), docs/report.md (`missing`), README, CHANGELOG 1.7.0 (local); runnable example with real output
- [x] 4.4 Full suite passes; mypy clean; `openspec validate missing-values --strict` passes
