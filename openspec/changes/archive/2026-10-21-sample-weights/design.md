## Context

Models are fit in `_fit_model(name, model, X, y)` and scored with `scorer(model, X, y)`; scikit-learn scorers accept `sample_weight=` as a keyword. Built-in models may be pipelines (scaler, imputer, ColumnTransformer since 1.9). The dummy baseline is fit in `fit()` and scored in `_score_baseline`. See proposal.md.

## Decisions

### D1. Routing to the final estimator
For a `Pipeline`, fit with `{f"{last_step_name}__sample_weight": w}`; for a bare estimator, `sample_weight=w`. Support is checked with `sklearn.utils.validation.has_fit_parameter` on the final estimator before fitting; unsupported → a `fit` failure (`SampleWeightNotSupported`), not an unweighted fit. Preprocessing steps (imputer, scaler, encoder) stay unweighted, as in scikit-learn's own pipelines.

### D2. Scoring
Every `scorer(...)` call passes `sample_weight=` when evaluation (or, for train scores, training) weights are present; otherwise the call is unchanged, so unweighted reports are identical.

### D3. Validation and storage
`_validate_weights(w, n_rows)` → float copy, 1-D, finite, ≥ 0, sum > 0. `fit()` keeps training weights for train scores (`train_weight_` alongside `train_X_` when evidence is on). `rank_resampled` validates once and slices per split.

### D4. Selector routing
`ConsensusSelector.fit(X, y, groups=None, sample_weight=None)`; scikit-learn's metadata routing handles pipelines when the user opts in. Without routing enabled, passing `sample_weight` to a pipeline raises scikit-learn's own error, as for any estimator.

### D5. CLI
`--sample-weight COLUMN`: read from both CSVs (must exist in both), removed from features, validated like the API.

## Risks / Trade-offs

- [Excluding `lda`, `qda`, `pls_regression` under weights reduces family coverage] → recorded as failures; the `failures` finding surfaces it.
- [Correlations and SHAP stay unweighted] → documented; they describe the data, not the weighted objective.
