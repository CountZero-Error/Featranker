# Troubleshooting

This page lists common errors and warnings, what they mean and what to do, followed by short answers to frequent questions.

[Manual](README.md) | [Getting started](getting-started.md) | [Report](report.md) | [Reference](reference.md)

## Errors

| Error or message | Cause | Fix |
|---|---|---|
| `X must contain only finite values; impute or drop missing ...` | NaN or infinite values in the features. | Impute or drop them first, or pass `missing="impute"` for NaN. Infinite values are always rejected. |
| `feature_names are required for NumPy input.` | `X` is a NumPy array and no names were given. | Pass `feature_names=[...]`, or pass a DataFrame. |
| `Evaluation feature names and order must match fit().` | `X_eval` columns differ from training. | Use the same columns in the same order. |
| `y contains labels not seen during fit(): ...` | The evaluation set has a class the training set lacks. | Stratify the split so every class appears in training. |
| `Regression targets must be numeric.` | `task="reg"` with a text target. | Convert the target to numbers, or use `task="clf"`. |
| `n_repeats must be a positive integer.` | `n_repeats` is 0, negative, or not an `int`. | Use 1 or more. |
| `No models fit successfully.` | Every model failed in `fit()`. The message lists each failure. | Read the listed estimator errors; check the data and any custom estimators. |
| `No models ranked successfully.` | Every model failed in `rank_features()`. The message lists each failure. | Check that the scorer suits the task and the models' outputs ([Choose a scorer](guide/models-and-scoring.md#choose-a-scorer)), and look for non-finite scores. |
| `call fit() again` (`ValueError`) | An option such as `missing`, `model_input`, `max_categories` or `shadows` changed after `fit()`. | Call `fit()` again, or make a new `FeatureRanker`. |
| `X must be numeric` | A text, date or time column without `categorical=`, or a date column even with it. | Encode it, list text columns in `categorical`, or convert dates to numbers or labels ([Categorical features](guide/data.md#categorical-features)). |
| Pickling error with `n_jobs` | A custom estimator or scorer cannot be sent to worker processes. | Make everything it holds serializable by joblib (locks and connections cannot be), or use `n_jobs=None`. |

## Warnings and report entries

| What you see | Meaning | What to do |
|---|---|---|
| `Skipping model '<name>': ...` | An optional library (XGBoost, CatBoost) is missing or broken. | Install the extra, or ignore it; other models still run. |
| `No model beat the dummy baseline, so the consensus is empty.` | No model did better than a model that ignores the features. | See [Reading the report](report.md#reading-the-report): the features may not predict the target, or the evaluation set is too small. |
| `failures` entry with `error_type` `SampleWeightNotSupported` | With weights, a model without `sample_weight` in `fit()` (built-in `lda`, `qda`, `pls_regression`) is not fit unweighted. | Ignore it, or replace the model with a weight-aware one ([Sample weights](guide/data.md#sample-weights)). |
| `failures` entry with stage `shadows` and `ShadowColumnsDropped` | A custom pipeline drops or hides the shadow columns, so it is left out of the shadow evidence. | Keep the shadow columns, for example `remainder="passthrough"` ([The shadow contract](guide/custom-models.md#the-shadow-contract)). |
| `UserWarning` about `shap` | `evidence=True` but `shap` is missing, broken or older than 0.49. | `pip install 'featranker[shap]'`. The rest of the evidence is still computed. |
| `UserWarning` about `constant_within_groups` | With `shuffle="within_groups"`, some groups never vary inside a shuffle group, so their importance is zero by construction. | Use `shuffle="rows"` with a grouped splitter ([What within_groups measures](guide/validation.md#what-within_groups-measures)). |
| `max_shadow_q` experimental warning | The shadow q-values are uncalibrated reference values. | Treat the filter as a heuristic and check held-out scores. |
| `fit()` warns that no group passed | A `ConsensusSelector` rule kept nothing, so it keeps no columns. | Loosen the shadow rules, or use a positive `k` ([Selection](guide/selection.md#consensusselector-in-a-pipeline)). |
| `insufficient_splits` finding | Too few splits for the shadow hits test to reach a q-value of 0.05. | Use `RepeatedKFold` or a splitter with 10 or more splits. |

## FAQ

**Do I need a separate evaluation set?** Yes. `fit()` takes training rows and `rank_features()` takes different rows. FeatRanker cannot detect reuse, so keeping them apart is up to you.

**Why is a model in the report but not in the vote?** It did not score strictly better than the dummy baseline, so it has `in_consensus: False`. See [Concepts](concepts.md#the-dummy-baseline-gate).

**Why do two correlated features both look unimportant?** They share credit. Put them in one `feature_groups` entry, or use [`auto_groups()`](guide/selection.md#grouping-correlated-features-with-auto_groups).

**Why does a half-size evaluation sample change the top three?** The ranking is noisy on small samples. Check stability with [`rank_resampled()`](guide/validation.md#resampled-ranking) before you trade rows for speed.

**Can I use the ranking to pick features for my final model?** Only inside validation. Use [`ConsensusSelector` in a pipeline](guide/selection.md#consensusselector-in-a-pipeline), and report performance on a test set no step touched.

**Are the shadow p-values calibrated?** No. They are reference values for ordering groups ([Shadow hits test](guide/noise-and-evidence.md#shadow-hits-test)).

**How do I make a run reproducible?** Use seeded estimators, a fixed integer `random_state`, and with shadows a fixed `shadow_seed`. See [Parallel models](guide/custom-models.md#parallel-models).

**Where is `to_prompt()`?** It was removed in 2.0, together with `explain()`. See [API](api.md#removed).
