## Context

1.7 (`missing-values`) adds `_validate_X(..., allow_nan=...)` and median imputers in built-in pipelines. `_validate_X` casts everything to float. Permutation importance, shadows, and resampling all operate on the float matrix by column index. See proposal.md.

## Decisions

### D1. Codes, not one-hot, at the ranker level
The ranker works on a float matrix where each categorical column holds integer codes. Permuting a code column equals permuting the raw values, so grouped permutation, shadows, stability, and the selector keep working by column index with no change, and each raw column stays one feature. One-hot encoding happens only inside built-in linear pipelines.

### D2. Where coding happens
`_validate_X` gains `categorical: Optional[List[int]]`: those columns are taken as object values and not cast; numeric columns are cast as today. `fit()` learns a level map per categorical column from the training rows (`sorted(set(map(str, values)))`, missing → its own level, appended last) and stores it (`categorical_levels_`); `rank_features()` applies it (unknown → code `k`, one past the last level, shared "unseen" code). `rank_resampled()` and the selector validate raw input without coding and slice it per split, so every split learns its own map. The shadow block is built after coding (it permutes codes).

### D3. Built-in pipelines
Linear: `make_pipeline(ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx)], remainder=<numeric steps>), estimator)` where `<numeric steps>` is `StandardScaler()` or `make_pipeline(SimpleImputer(strategy="median"), StandardScaler())` under `missing="impute"`; `cat_idx` covers real and shadow categorical columns. Tree: unchanged (codes as numbers; under `missing="impute"` the median imputer stays first and leaves code columns untouched because codes are never NaN). Built at `fit()` time, when the column indices are known; without categorical columns, models are built exactly as in 1.7.

### D4. Evidence
`correlation_evidence` and `auto_groups` receive the categorical indices and skip them. `_unwrap` already rejects non-per-column steps, so a `ColumnTransformer` makes linear models unsupported with a clear reason; trees are explained on codes.

### D5. Detection under "auto"
DataFrame dtypes `object`, `string`, `category` → categorical. Bool and numeric dtypes stay numeric. NumPy input has no dtypes per column, so `"auto"` finds none there; users list names.

### D6. Record and attribute
`categorical_features_` on the fitted ranker; `report["categorical"]` as in the spec.

## Risks / Trade-offs

- [Codes impose an arbitrary order on tree models] → trees split on codes as ordinal values; documented; one-hot or native categorical support is available through custom models.
- [High-cardinality columns make one-hot pipelines wide] → documented; the record shows `train_levels`.
- [Level order uses string forms, so `1` and `"1"` collide] → documented; such columns should be cleaned first.
