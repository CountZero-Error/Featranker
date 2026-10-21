## Why

Custom models receive a plain float array, so users cannot select columns by name in their own pipelines (`ColumnTransformer([... , ["age", "income"]])`), which is the normal scikit-learn way to preprocess mixed data. And with shadows on, a custom pipeline that keeps only named or listed columns silently drops the shadow block: its shadow importances are all zero, every real feature "beats" the shadows, and the shadow evidence is wrong without any warning.

## What Changes

- `FeatureRanker(model_input="array")` gains `"dataframe"`: every model (built-in and custom) is fit and scored on a pandas DataFrame whose columns are the feature names, followed by `shadow__<name>` for shadow columns. Permutation importance, the baseline, train scores, and SHAP keep working; values are the same as in array mode.
- Shadow-compatibility check: with shadows on, after fitting, any model whose pipeline contains a `ColumnTransformer` that drops shadow columns (they are neither selected by a transformer nor kept by `remainder`) is excluded from shadow evidence for that report, and a failure with stage `"shadows"` names it. Its importances and ranking are otherwise kept.
- `ConsensusSelector(model_input=...)` and `rank_resampled` pass the option through; the CLI keeps array input.
- `pandas` is required only for `model_input="dataframe"`; without it, a clear `ImportError` at construction.
- Version 1.10.0.

Report schema: additive (stage `"shadows"` failures only when a model is excluded; `model_input` recorded when `"dataframe"`).

## Non-goals

- Changing what built-in models receive in the default `"array"` mode.
- Detecting every way a custom model can ignore shadow columns (only `ColumnTransformer` column selection is inspected).

## Capabilities

### New Capabilities
- `named-model-input`: the `model_input` option, shadow column names, the shadow-compatibility check, and pass-through.

### Modified Capabilities
- `report-summary`: `"shadows"` failures are warnings.

## Impact

- Code: `featranker/importance.py` (model calls wrapped when `"dataframe"`; post-fit check), `featranker/selection.py`.
- Docs: docs/usage.md (custom pipelines by column name; the shadow contract), docs/report.md, README; CHANGELOG (local).
