## Context

Models are fit in `_fit_model` and scored in `_rank_model` / `_group_importances` (`scorer(model, X, y)` on NumPy arrays, with `X_permuted` rewritten in place per repeat). Shadows append a block of columns. Failures are dicts with `model`, `stage`, `error_type`, `error`. See proposal.md.

## Decisions

### D1. Wrap at the call sites
A small helper `_model_X(X, names)` returns `X` in `"array"` mode and `pd.DataFrame(X, columns=names, copy=False)` in `"dataframe"` mode; it is used for every fit, baseline/permuted/train score, and SHAP call. `names` = feature names plus `shadow__<name>` for shadows. Permutation keeps mutating the NumPy buffer; the DataFrame is rebuilt from it for each scorer call (cheap relative to prediction). The dummy baseline keeps receiving arrays (it ignores features).

### D2. Shadow-compatibility check
After fit, walk the fitted model: if it is a `Pipeline`, inspect its steps (recursively into nested pipelines); for each fitted `ColumnTransformer`, collect the input columns used by `transformers_` (resolving names, indices, slices, boolean masks, and callables through the fitted object's `_columns`/`output_indices_` where available) and whether `remainder` is not `"drop"`. If some shadow column index is unused, the model is incompatible. Only `ColumnTransformer` is inspected; other column-dropping estimators are documented as the user's responsibility. Incompatible models get a `"shadows"` failure and are skipped when shadow flags, `shadow_max`, and family/consensus shares are computed.

### D3. Records
`report["model_input"] = "dataframe"` only in that mode. Stage `"shadows"` joins the summary's failure stages with severity `"warning"` (shadow evidence is affected).

## Risks / Trade-offs

- [Built-in estimators fit on DataFrames store `feature_names_in_`; scikit-learn warns if later given arrays] → every call in a ranker uses the same mode, so no mixed calls happen.
- [DataFrame construction per permutation repeat costs time on wide data] → only in `"dataframe"` mode; documented.
- [Column selectors that cannot be resolved (custom callables)] → treated as "cannot verify": no failure, a `UserWarning` once per model.
