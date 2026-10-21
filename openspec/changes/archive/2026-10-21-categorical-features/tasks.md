## 1. Option and coding

- [x] 1.1 Test: `categorical` validation (bad value, unknown name at fit); default rejects strings as in 1.7; reports byte-identical to HEAD without categorical columns; `"auto"` detection by dtype; explicit list on NumPy object arrays; old pickles default to None
- [x] 1.2 Test: level coding learned from training rows only (sorted, missing as its own level), unseen evaluation values share one code, shadows permute codes, each raw column is one feature
- [x] 1.3 Implement validation and per-fit coding (D2, D5)
- [x] 1.4 Test and fix (from the 1.7 review): a categorical column whose categories are datetime/timedelta, and NumPy datetime64/timedelta64 arrays, raise the "X must be numeric" error under every policy instead of being converted to nanoseconds

## 2. Models, evidence, record

- [x] 2.1 Test: built-in linear pipelines one-hot categorical columns and scale numeric ones (with and without `missing="impute"`); trees use codes; scenario "linear and tree built-ins fit" with the categorical ranking first; shadows with categoricals; custom models receive codes and `categorical_features_` is set
- [x] 2.2 Test: categorical columns excluded from correlation evidence and `auto_groups`; SHAP explains trees on codes and lists one-hot linear pipelines as unsupported; `report["categorical"]` record; resampled split reports carry it; strict JSON
- [x] 2.3 Implement (D3, D4, D6)

## 3. Pass-through, docs, checks

- [x] 3.1 Test: selector scenario on a mixed DataFrame (selects and returns the raw column); selector clone/get_params/old pickles; `rank_resampled` learns levels per split; CLI `--categorical auto` and a name list; typing consumer
- [x] 3.2 Implement pass-through; version 1.9.0
- [x] 3.3 Docs: docs/usage.md (mixed tables: categorical option, fold-local coding, one feature per raw column, linear one-hot vs tree codes, limits), docs/report.md (`categorical`), README, CHANGELOG 1.9.0 (local); runnable example with real output
- [x] 3.4 Full suite passes; mypy clean; `openspec validate categorical-features --strict` passes
