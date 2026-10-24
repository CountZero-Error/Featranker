## Context

`_drops_shadows` only inspects `ColumnTransformer`s whose `n_features_in_` equals the model input width, found by walking `Pipeline` steps. `_hash_column` casts numbers to float64. `ablate()` compares fingerprints with the report's `run` block.

## Decisions

**D1. Column mapping.** A recursive `_kept_inputs(estimator, columns) -> set` where `columns` lists the model-input indices the estimator receives:
- `ColumnTransformer` (fitted): for each entry in `transformers_` (remainder included), the branch columns are `[columns[i] for i in _transformer_to_input_indices[name]]`; `"drop"` adds none, `"passthrough"` adds all, any other estimator recurses.
- `Pipeline`: recurse into the first step that is not `None`/`"passthrough"` with the same `columns`. A later step is mapped (intersected) only while every earlier step is known to keep the columns one for one: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, or a `SimpleImputer` with `add_indicator=False` whose `get_feature_names_out()` is as wide as its input. A later step that is or contains a `ColumnTransformer` after any other step is unknown (a width check is not enough: a reordering transformer keeps the width).
- Fitted `SelectorMixin`: the columns at `get_support(indices=True)`.
- `FeatureUnion`: the union over its non-`"drop"` parts, each given all `columns`.
- Any other estimator whose `get_params(deep=True)` holds a `ColumnTransformer` (a meta-estimator such as `TransformedTargetRegressor`) is unknown.
- Anything else (`FunctionTransformer`, custom transformers, final estimators): all `columns`; documented as outside what the check can see.
Private-attribute access errors make the result unknown. Known or unknown, `_drops_shadows` returns True (a `shadows` failure) when shadows are not provably kept; the failure message says which ("drops shadow columns" vs "cannot verify").

**D2. Fingerprint scheme 2.** `_hash_column` tags each column with its `dtype.kind` only, `b` folded into `i` (a boolean hashes as the int64 0/1, so `int32` equals `int64`; an integer and a float column with equal values still differ). Kinds `b`, `i` hash `astype("<i8")`; `u` hashes `astype("<u8")`; `f` hashes float64 with one NaN pattern; others keep the chunked repr. Keep the scheme 1 function for `ablate()` when a report lacks `run.fingerprint_scheme`. The constant `FINGERPRINT_SCHEME = 2` is recorded in every `run` block (held-out and resampled). `fit()` stores `fingerprint_scheme_` and `rank_features()` records it and fingerprints `data_sha256` with it, so a ranker pickled before 1.17 (`__setstate__` default 1, with a scheme-1 `train_data_sha256_`) writes an internally consistent run block that `ablate()` accepts. `rank_resampled()` fits fresh rankers, so it is scheme 2.

**D3. Ablation provenance.** `data_sha256` = scheme-2 fingerprint of the passed evaluation columns (all features, as for `rank_features`) and `y_eval`; `weights_sha256` of `eval_sample_weight`; `n_eval_rows`; `fingerprint_scheme` (the current scheme, for `data_sha256`); `same_holdout` = data fingerprint equals the report's `run.data_sha256` (same scheme) and weights fingerprint equals `run.weights_sha256`.

**D4. Paired curve.** `paired_models` = models with a non-null score at every k (report order). Each row: `n_models_scored`, `paired_median_score` (null when `paired_models` is empty). `median_score` stays. `summarize()` picks best k on `paired_median_score` when `paired_models` is non-empty, else on `median_score`, and states which; when the set of models with a score differs between rows (not just the count), it adds a `warning` finding `ablation_cohort_changed`.

**D5. Zero-weight subsample.** After drawing the `max_samples` subsample, if weights are given and their sum over sampled rows is 0, raise `ValueError` naming `max_samples` and `sample_weight`. `rank_resampled` applies it per split through `rank_features`.

**D6. Floor.** `scikit-learn>=1.5` in `pyproject.toml`; CI matrix change is 1.20's work.

## Risks

- Fingerprint values change for integer data; reports from 1.12–1.16 cannot be compared by hash with 1.17 ones. Changelog says so; `ablate()` handles old reports via D2.
- Turning "cannot verify" into a failure removes shadow evidence for exotic pipelines that were previously trusted with a warning.
