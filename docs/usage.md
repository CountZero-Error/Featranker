# Usage

The usage guide is now a manual of several pages. This page maps each former section to its new place.

[Manual](README.md) | [Getting started](getting-started.md) | [Reference](reference.md)

## Start here

| You want to | Go to |
|---|---|
| Install and rank your first data set | [Getting started](getting-started.md) |
| Understand how the ranking is decided | [Concepts](concepts.md) |
| Look up a parameter, method or error | [Reference](reference.md) |
| Run it from a shell | [Command line](cli.md) |
| Fix an error or a warning | [Troubleshooting](troubleshooting.md) |

## Where each topic went

| Former section | New place |
|---|---|
| API (`FeatureRanker`, `fit`, `rank_features`, `ablate`) | [Reference](reference.md#featureranker) |
| Working with reports (`summary`, `to_frame`, `plot`) | [Reference](reference.md#report) and [Report](report.md) |
| Choosing task, group and scorer | [Models and scoring](guide/models-and-scoring.md) |
| Several metrics | [Models and scoring](guide/models-and-scoring.md#several-metrics) |
| Speed | [Performance](guide/performance.md) |
| Fewer evaluation rows (`max_samples`) | [Performance](guide/performance.md#fewer-evaluation-rows-max_samples) |
| Large data | [Performance](guide/performance.md#large-data) |
| Your own models | [Custom models](guide/custom-models.md#your-own-models) |
| Columns by name | [Custom models](guide/custom-models.md#columns-by-name) |
| Parallel models | [Custom models](guide/custom-models.md#parallel-models) |
| Missing values | [Data](guide/data.md#missing-values) |
| Categorical features | [Data](guide/data.md#categorical-features) |
| Sample weights | [Data](guide/data.md#sample-weights) |
| Labels and data handling | [Data](guide/data.md#labels-and-data-handling) |
| Command line | [Command line](cli.md) |
| Resampled ranking | [Validation](guide/validation.md#resampled-ranking) |
| Time series and grouped data | [Validation](guide/validation.md#time-series-and-grouped-data) |
| Shuffling dependent rows | [Validation](guide/validation.md#shuffling-dependent-rows) |
| Shadow features | [Noise and evidence](guide/noise-and-evidence.md#shadow-features) |
| Shadow hits test | [Noise and evidence](guide/noise-and-evidence.md#shadow-hits-test) |
| Family evidence | [Noise and evidence](guide/noise-and-evidence.md#family-evidence) |
| Shadow mode | [Noise and evidence](guide/noise-and-evidence.md#shadow-mode) |
| Evidence and SHAP | [Noise and evidence](guide/noise-and-evidence.md#evidence) |
| Feature selection in a pipeline | [Selection](guide/selection.md#consensusselector-in-a-pipeline) |
| Ablation curve | [Selection](guide/selection.md#ablation-curve) |
| Summarize a report | [Selection](guide/selection.md#summarize-a-report) |
| Comparing runs | [Selection](guide/selection.md#comparing-runs) |
| Grouped permutation and `auto_groups()` | [Selection](guide/selection.md#grouped-permutation) |
| Troubleshooting | [Troubleshooting](troubleshooting.md) |

## The guides

| Page | Covers |
|---|---|
| [Models and scoring](guide/models-and-scoring.md) | Task, group, presets, scorers, several metrics |
| [Custom models](guide/custom-models.md) | Your own models and pipelines, families, named columns, `n_jobs` |
| [Data](guide/data.md) | Missing values, categorical features, sample weights, labels |
| [Validation](guide/validation.md) | `rank_resampled()`, time series and grouped data, nested validation |
| [Noise and evidence](guide/noise-and-evidence.md) | Shadow features, hits test, family evidence, evidence and SHAP |
| [Selection](guide/selection.md) | `ConsensusSelector`, ablation curve, `summarize()`, comparing runs, `auto_groups()` |
| [Performance](guide/performance.md) | Speed, `max_samples`, large data, `max_categories` |
