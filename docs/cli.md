# Command line

English · [简体中文](zh-CN/cli.md)

This page documents the `featranker` command: what it reads, every flag, and examples.

[Manual](README.md) | [Getting started](getting-started.md) | [Report](report.md) | [Reference](reference.md)

## Quick example

The command takes separate training and evaluation CSV files and writes the report as JSON.

```bash
featranker \
  --task reg \
  --group tree \
  --train-data train.csv \
  --eval-data eval.csv \
  --target target \
  --scoring neg_mean_absolute_error \
  --feature-groups groups.yaml \
  --n-repeats 20 \
  --random-state 42 \
  --n-jobs 2 \
  --output report.json
```

Run `featranker --help` for the flag list.

## Input files

- Both CSV files need headers, the same feature columns in the same order, and the target column.
- Feature values must be numeric unless named by `--categorical`. Targets may be numeric or strings.
- By default empty feature cells are rejected. With `--missing impute` they are read as missing values and imputed per training split ([Data](guide/data.md#missing-values)). Only empty cells and the text `nan` (any case) are read as missing; `NA`, `N/A` and `null` are not, and raise the numeric-values error.

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.2,8.1,1,0,12.4
2.1,7.3,0,1,14.8
3.4,6.8,1,0,18.2
4.0,5.9,0,1,20.1
```

That is `train.csv`. `eval.csv` has the same columns:

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.7,7.8,1,0,13.6
3.8,6.2,0,1,19.4
```

`groups.yaml` maps a group name to its columns:

```yaml
segment:
  - segment_red
  - segment_blue
```

## Required flags

| Flag | Meaning |
|---|---|
| `--task` | `clf` for classification or `reg` for regression. |
| `--train-data` | CSV with the training features and target. |
| `--eval-data` | CSV with the held-out evaluation features and target. |
| `--target` | Name of the target column in both files. |
| `--scoring` | A scikit-learn scorer name, or several separated by commas (`--scoring accuracy,neg_log_loss`). The first is primary; see [Several metrics](guide/models-and-scoring.md#several-metrics). |

## Models and ranking

| Flag | Default | Meaning |
|---|---|---|
| `--group` | `all` | Built-in model group: `linear`, `tree`, `all` or `fast`. |
| `--n-repeats` | `10` | Permutation repeats per feature group. |
| `--random-state` | none | Seed for reproducible permutations. Also seeds the `--max-train-samples` draw. |
| `--n-jobs` | sequential | Parallel workers over models; `-1` uses all cores. |
| `--evidence` | off | Add [evidence](guide/noise-and-evidence.md#evidence) to the report: train score, overfit gap, correlated groups and SHAP when `shap` is installed. |
| `--quiet` | off | Hide progress bars. Warnings and errors are still shown. |
| `--output` | stdout | Write the JSON report to this path. A `.json` extension is appended if missing. An existing report is replaced once the new one is complete (written in place only when its folder is read-only) and keeps its owner, group, mode and access rules; on Linux and macOS it is refused without write access to it. A new report gets your default permissions (umask). |

## Feature groups

| Flag | Default | Meaning |
|---|---|---|
| `--feature-groups` | none | A YAML file of `group: [columns]`, or the word `auto` to group correlated training columns with [`auto_groups()`](guide/selection.md#grouping-correlated-features-with-auto_groups). A file literally named `auto` must be written `./auto`. |
| `--auto-groups-threshold` | `0.7` | With `--feature-groups auto`, group columns whose absolute Spearman correlation is at least this. |

## Data options

| Flag | Meaning |
|---|---|
| `--missing` | `error` (default) rejects empty feature cells. `impute` gives every built-in model a median imputer fit on training data only. |
| `--categorical` | `auto` reads the training-CSV columns that have a non-numeric cell as text (a stray `NA` makes its whole column categorical; a column literally named `auto` cannot be listed). A comma-separated list such as `city,segment` reads those columns as text. Levels come from the training CSV only; an empty cell in such a column is a missing level. See [Categorical features](guide/data.md#categorical-features). |
| `--sample-weight` | A column present in both files, used as sample weights for fitting and scoring and not ranked as a feature ([Sample weights](guide/data.md#sample-weights)). |
| `--drop` | Comma-separated columns removed from both files before any feature is read, such as `id,created_at`, so they never become features (and a dropped text column is not read as categorical). An unknown name, the target, the `--sample-weight` column or a `--categorical` column raises an error, and so does dropping every feature. |

## Large data

| Flag | Meaning |
|---|---|
| `--max-samples` | Score on a subsample of the evaluation rows: an int (rows) such as `5000`, or a float with a decimal point in (0, 1] such as `0.25` (a share). |
| `--max-train-samples` | Fit the models on a subsample of the training rows, with the same forms. Drawn with `--random-state`. |
| `--max-categories` | At least 2. Keep at most this many one-hot columns per categorical column in the linear models; rarer levels share one column. |

See [Performance](guide/performance.md#large-data) for what each bounds.

## Ablation curve

| Flag | Meaning |
|---|---|
| `--ablate` | Add the [`ablation` block](guide/selection.md#ablation-curve) to the written report. It refits the voting models for each `k`, so the run takes longer. |
| `--ablate-ks` | With `--ablate`, comma-separated positive ints such as `1,2,5`. Default: 1, 2, 3, 5, 10, and so on. A `k` above the number of feature groups raises an error before any model is fit. |

> [!WARNING]
> The CLI curve is scored on the evaluation CSV that chose the groups, so it is optimistic. See the warning in [Ablation curve](guide/selection.md#ablation-curve). If no model beats the dummy there is no consensus: `--ablate` prints a warning to stderr and the report is written without an `ablation` block.

## Limits

The command line keeps array input and always shuffles rows freely (`shuffle="rows"`), because it has no group column. For time series, grouped rows, your own models or `rank_resampled()`, use the Python API.
