## Why

FeatRanker accepts numeric features only, so a categorical column must be encoded before the call. Encoding on all development rows leaks evaluation categories into the preprocessing, and one-hot encoding splits a single raw feature across many columns, so permutation importance no longer answers "how much does this raw feature matter?". Users with ordinary mixed tables (Signal's requirement R3; competitors' categorical complaints) need FeatRanker to accept the raw column, encode it inside each fit, and rank it as one feature.

## What Changes

- `FeatureRanker(categorical=None)`: `None` keeps 1.x behaviour (non-numeric columns raise). `"auto"` treats DataFrame columns with object, string, or category dtype as categorical; a list of feature names marks those columns explicitly (works for NumPy object arrays too).
- Categorical columns are mapped to integer codes per `fit()`: levels are learned from the training rows only (sorted, deterministic); missing values get their own level; levels unseen in training map to an "unseen" code. Permuting codes is permuting raw values, so each raw categorical column stays one feature group.
- Built-in models: linear models one-hot encode categorical codes (unknown codes ignored) and scale numeric columns, through a `ColumnTransformer` inside their pipeline; tree models use the codes directly. Custom models receive the code matrix; the ranker exposes which columns are categorical.
- With `missing="impute"`, numeric columns are imputed as in 1.7; categorical missing values are already a level.
- Evidence: categorical columns are left out of Spearman correlations (correlated groups, `auto_groups`); the SHAP summary supports tree models on codes and lists linear models with one-hot steps as unsupported.
- Reports record `categorical`: the columns, their number of training levels, and how many evaluation values were unseen.
- `rank_resampled`, `ConsensusSelector(categorical=...)`, and CLI `--categorical auto|name,name` pass it through; encoding is learned inside each split.
- Version 1.9.0.

Report schema: additive (`categorical` only when categorical columns are present).

## Non-goals

- Target encoding or other supervised encoders (leakage-prone; users can pass custom models).
- Passing DataFrames with column names to custom models (next change, `named-model-input`).
- Native categorical support of HistGradientBoosting, CatBoost, or LightGBM.

## Capabilities

### New Capabilities
- `categorical-features`: the `categorical` option, fold-local level coding, built-in model handling, evidence behaviour, report record, and pass-through.

### Modified Capabilities

## Impact

- Code: `featranker/importance.py` (validation keeps raw categorical values until coding; per-fit level maps; model construction), `featranker/evidence.py`, `featranker/selection.py`, `featranker/cli.py`.
- Docs: docs/usage.md (mixed tables), docs/report.md, README; CHANGELOG (local).
