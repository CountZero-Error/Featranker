## 1. Model input

- [ ] 1.1 Test: `model_input` validation and pandas-missing ImportError; default reports identical to HEAD; "dataframe" gives every model a DataFrame with real and `shadow__` names (recording estimator); built-in reports equal between modes; select-by-name scenario; SHAP summary unchanged between modes; old pickles default to "array"; `model_input` recorded only in "dataframe" mode
- [ ] 1.2 Implement the wrapper at every model call (D1)

## 2. Shadow compatibility

- [ ] 2.1 Test: ColumnTransformer with remainder="drop" listing real columns → excluded from shadow evidence, `"shadows"` failure, still in consensus; remainder="passthrough" → no failure; nested pipelines; index, name, slice, and boolean-mask selectors; an unresolvable callable selector → UserWarning, no failure; summarize() reports the stage as a warning; works in both input modes
- [ ] 2.2 Implement (D2, D3)

## 3. Pass-through, docs, checks

- [ ] 3.1 Test: selector scenario; selector clone/get_params/old pickles; rank_resampled uses the mode per split; typing consumer
- [ ] 3.2 Implement pass-through; version 1.10.0
- [ ] 3.3 Docs: docs/usage.md (custom pipelines by column name, shadow column names, the shadow contract and the check), docs/report.md (`model_input`, `"shadows"` failures), README; CHANGELOG 1.10.0 (local); runnable example with real output
- [ ] 3.4 Full suite passes; mypy clean; `openspec validate named-model-input --strict` passes
