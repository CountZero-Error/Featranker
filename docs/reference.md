# Reference

English · [简体中文](zh-CN/reference.md)

This page lists every public name in FeatRanker with its signature, parameters, return value and errors. The same text is in `help()`.

[Manual](README.md) | [Getting started](getting-started.md) | [Report](report.md) | [Input rules](#input-rules)

Everything in `featranker.__all__` is public: `FeatureRanker`, `ConsensusSelector`, `Report`, `auto_groups` and `summarize`, plus `featranker.__version__`. See [API](api.md) for the stability policy.

## FeatureRanker

### `FeatureRanker`

```text
FeatureRanker(task, group="all", models=None, n_jobs=None, verbose=True, shadows=False,
              shadow_seed=None, evidence=False, shadow_mode="joint", missing="error",
              categorical=None, model_input="array", shuffle="rows", families=None,
              max_categories=None)
```

Ranks feature groups by permutation importance across many models. Creating it reads no data. Invalid options raise `ValueError` (or `TypeError` for a wrong type). Details of limits are in [Input rules](#input-rules).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `"clf"` or `"reg"` | required | Classification or regression. |
| `group` | `"all"`, `"linear"`, `"tree"`, `"fast"` | `"all"` | Built-in model group. `"fast"` runs one model per family. Cannot be combined with `models`. See [Models](models.md). |
| `models` | dict | `None` | Non-empty `{name: estimator}` that replaces the built-in models. Names are non-empty strings other than `dummy_baseline`. See [Custom models](guide/custom-models.md). |
| `n_jobs` | int | `None` | Parallel workers over models. `None` or `1` is sequential, `-1` uses all cores. |
| `verbose` | bool | `True` | Show progress bars. Warnings are always shown. |
| `shadows` | bool | `False` | Also train and rank a shuffled copy of every feature, to show which groups beat noise. See [Shadow features](guide/noise-and-evidence.md#shadow-features). |
| `shadow_seed` | int | `None` | Seed of the shadow shuffles. `None` gives different shadows on every run. |
| `evidence` | bool | `False` | Keep the training data and add train scores, overfit gaps, correlated groups and (with `featranker[shap]`) a SHAP summary. See [Evidence](guide/noise-and-evidence.md#evidence). |
| `shadow_mode` | `"joint"` or `"independent"` | `"joint"` | `"joint"` shuffles all shadow columns with one shared row order, `"independent"` shuffles each column on its own. Used only with `shadows=True`. |
| `missing` | `"error"` or `"impute"` | `"error"` | `"error"` rejects NaN. `"impute"` accepts NaN and gives every built-in model a median imputer fit on its training rows only; custom models get NaN unchanged. See [Missing values](guide/data.md#missing-values). |
| `categorical` | `"auto"` or list of str | `None` | `None` accepts numeric features only. `"auto"` treats object, string and category DataFrame columns as categorical; a list names the columns. See [Categorical features](guide/data.md#categorical-features). |
| `model_input` | `"array"` or `"dataframe"` | `"array"` | `"array"` gives models a float NumPy array. `"dataframe"` gives a DataFrame with the feature names (and `shadow__<name>` columns). Needs pandas. See [Columns by name](guide/custom-models.md#columns-by-name). |
| `shuffle` | `"rows"`, `"within_groups"`, `"circular"` | `"rows"` | How permutations and shadows shuffle rows: freely, only among rows with the same label in `shuffle_groups`, or by a cyclic shift (rows in time order). See [Shuffling dependent rows](guide/validation.md#shuffling-dependent-rows). |
| `families` | dict | `None` | `{model_name: family}` that sets the consensus family of custom or built-in models. Models in one family cast one vote; unmapped custom models stay separate. See [Families](guide/custom-models.md#families). |
| `max_categories` | int | `None` | At least 2. Built-in linear models one-hot encode at most this many levels per categorical column; the rest share one column. Trees and custom models are unaffected. See [Large data](guide/performance.md#large-data). |

Fitted attributes: `trained_models` (name to fitted model), `feature_names_`, `label_encoder` (classification: `classes_[i]` is the label of code `i`), `categorical_features_`, `categorical_levels_`, `fit_failures_` and `initialization_failures_` (lists of failures).

### `FeatureRanker.fit`

```text
fit(X, y, feature_names=None, sample_weight=None, shuffle_groups=None, max_samples=None,
    random_state=None)
```

Fits every configured model and the dummy baseline on the training data. Returns the ranker itself. Raises `ValueError` for invalid data or options and `RuntimeError` if no model fits.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | Training features, at least one row and one column. Numeric, plus columns named by `categorical`. Infinite values are rejected; NaN is rejected unless `missing="impute"`. |
| `y` | array | required | Training target, one value per row. Class labels may be numbers or strings; regression targets must be numeric. |
| `feature_names` | list of str | `None` | Unique, non-empty column names. Required for NumPy input; taken from the columns of a DataFrame. |
| `sample_weight` | array | `None` | Finite, non-negative weights, one per row, with a positive sum. Every model is fit with them; models that cannot take weights fail. See [Sample weights](guide/data.md#sample-weights). |
| `shuffle_groups` | array | `None` | One group label per training row. Required with `shuffle="within_groups"`; an error with another `shuffle`. |
| `max_samples` | int or float | `None` | Fit on a random subsample: an int (rows) or a float in (0, 1] (share). Shadows and the evidence copy use the sample too. See [Training subsample](guide/performance.md#training-subsample). |
| `random_state` | int | `None` | Seed of the `max_samples` draw. `None` draws fresh entropy. Unused without `max_samples`. |

### `FeatureRanker.rank_features`

```text
rank_features(X_eval, y_eval, *, scoring, feature_groups=None, n_repeats=10, random_state=None,
              sample_weight=None, max_samples=None, shuffle_groups=None)
```

Ranks feature groups on evaluation data that `fit()` did not see. Returns a [`Report`](report.md). Raises `RuntimeError` if `fit()` was not called or every model fails, and `ValueError` for invalid data or options.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X_eval` | array or DataFrame | required | Evaluation features with the same columns, in the same order, as `fit()`. Options changed since `fit()` raise `ValueError` ("call fit() again"); see [Input rules](#input-rules). |
| `y_eval` | array | required | Evaluation target. Class labels must have appeared in training. |
| `scoring` | str, callable or list | required | Keyword-only. A scikit-learn scorer name, a callable `(estimator, X, y) -> float` (higher is better), or a non-empty list of them; the first is primary. See [Scoring](guide/models-and-scoring.md). |
| `feature_groups` | dict | `None` | `{group_name: [feature, ...]}` to shuffle together. Ungrouped features are ranked on their own. See [Grouped permutation](guide/selection.md#grouped-permutation). |
| `n_repeats` | int | `10` | Shuffles per feature group; a positive integer. More repeats give a more precise mean and take longer. |
| `random_state` | int | `None` | Seed of the permutation shuffles. For reproducible reports also use seeded estimators, a deterministic scorer and, with shadows, a fixed `shadow_seed`. |
| `sample_weight` | array | `None` | Finite, non-negative evaluation weights, one per row, with a positive sum. Baseline, permuted and dummy scores are weighted. |
| `max_samples` | int or float | `None` | Score on one random subsample of the evaluation rows: an int (rows) or a float in (0, 1] (share). See [Fewer evaluation rows](guide/performance.md#fewer-evaluation-rows-max_samples). |
| `shuffle_groups` | array | `None` | One group label per evaluation row. Required with `shuffle="within_groups"`; an error with another `shuffle`. |

### `FeatureRanker.rank_resampled`

```text
rank_resampled(X, y, *, scoring, cv=5, groups=None, feature_names=None, feature_groups=None,
               n_repeats=10, random_state=None, top_k=5, sample_weight=None, max_samples=None,
               max_train_samples=None)
```

Fits and ranks on every cross-validation split and reports how stable the ranking is. Each split uses a new ranker with this ranker's settings, so this ranker is not changed. Returns a resampled [`Report`](report.md#resampled-reports). A split that fails alone is recorded in `failures`; it raises `RuntimeError` only if no split ranks.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | All development data, with the same rules as `fit()`. Each split takes its own rows. Keep your final test set out. |
| `y` | array | required | Target, one value per row. |
| `scoring` | str, callable or list | required | Keyword-only. As in `rank_features()`; stability uses the primary scorer. |
| `cv` | int, splitter or list | `5` | A number of folds (shuffled; stratified for classification), a scikit-learn splitter such as `RepeatedKFold` or `GroupKFold`, or a list of `(train_indices, eval_indices)`. Two identical splits raise `ValueError`. |
| `groups` | array | `None` | Passed to the splitter's `split()`. With `shuffle="within_groups"` they are also the shuffle groups, sliced per split, and required. |
| `feature_names` | list of str | `None` | As in `fit()`. |
| `feature_groups` | dict | `None` | As in `rank_features()`. |
| `n_repeats` | int | `10` | Shuffles per feature group in each split. |
| `random_state` | int | `None` | Seeds integer-`cv` shuffling. Split `i` uses seed `(random_state + i)` modulo 2**64. |
| `top_k` | int | `5` | A group is top in a split when its place is at most `top_k`; used for `top_k_share`. Ties share the better place, so a split can have more top groups. |
| `sample_weight` | array | `None` | Weights for all rows of `X`, sliced per split: each split trains and scores with its own rows' weights. |
| `max_samples` | int or float | `None` | As in `rank_features()`, applied to each split's evaluation rows. |
| `max_train_samples` | int or float | `None` | As `fit()`'s `max_samples`, applied to each split's training rows. The report gains `max_train_samples`. |

### `FeatureRanker.ablate`

```text
ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None,
       eval_sample_weight=None)
```

Refits the voting models of a held-out report on the top-`k` groups and scores them, giving a curve. Returns a new `Report`: a deep copy of `report` plus an [`ablation` block](report.md#ablation). `report` is not changed. Costs `len(ks)` refits of every voting model. See [Ablation curve](guide/selection.md#ablation-curve).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | The training data this ranker was fit on, checked against the report's `run["train_data_sha256"]`. |
| `y` | array | required | The training target. |
| `X_eval` | array or DataFrame | required | Data to score the refit models on, with the same columns as `fit()`. Every row is scored; the report's `max_samples` is ignored. |
| `y_eval` | array | required | Target of `X_eval`. |
| `report` | Report | required | A held-out report from this ranker's `rank_features()`. A resampled report or one with an empty consensus raises `ValueError`. See [Input rules](#input-rules). |
| `ks` | list of int | `None` | Keyword-only. Numbers of top groups, from 1 to the number of consensus groups. Default: 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000 and so on up to that number, plus that number. |
| `scoring` | str or callable | `None` | `None` uses the report's scorer name. Pass it when the report was ranked with a callable. |
| `sample_weight` | array | `None` | Training weights; pass the ones used for `fit()`. Checked against the report's `run["train_weights_sha256"]`. |
| `eval_sample_weight` | array | `None` | Evaluation weights; pass the ones used for `rank_features()`. Checked against `run["weights_sha256"]` when `X_eval` is the report's own data. |

## ConsensusSelector

### `ConsensusSelector`

```text
ConsensusSelector(task, scoring, *, k=None, min_shadow_share=None, max_shadow_q=None,
                  min_shadow_families=None, group="all", models=None, feature_groups=None,
                  auto_groups_threshold=0.7, cv=5, n_repeats=10, random_state=None,
                  max_samples=None, shadow_seed=None, shadow_mode="joint", missing="error",
                  categorical=None, model_input="array", shuffle="rows", families=None,
                  max_categories=None, max_train_samples=None, n_jobs=None, verbose=False)
```

A scikit-learn selector that ranks inside `fit()` and keeps the groups a rule selects. Use it as a `Pipeline` step. Set at least one of `k`, `min_shadow_share`, `max_shadow_q` and `min_shadow_families`: the shadow filters apply first, then the first `k` that pass. Invalid rules raise `ValueError` in `fit()`, not here.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `"clf"` or `"reg"` | required | As in `FeatureRanker`. |
| `scoring` | str, callable or list | required | As in `rank_features()`. With a list the first scorer selects and the rest are reported in `report_`. |
| `k` | int | `None` | Keep the first `k` groups in stability order (all if fewer). |
| `min_shadow_share` | float | `None` | Keep groups whose stability `above_shadow_share` is at least this (0 to 1). Turns shadows on. |
| `max_shadow_q` | float | `None` | Keep groups whose `shadow_q_value` is at most this (above 0, at most 1). Heuristic and experimental: `fit()` warns. Turns shadows on. Use `cv` with 10 or more splits. |
| `min_shadow_families` | int | `None` | Keep groups that at least this many model families put above their shadows in most splits. A positive integer; a heuristic. Turns shadows on. |
| `group` | `"all"`, `"linear"`, `"tree"`, `"fast"` | `"all"` | As in `FeatureRanker`. |
| `models` | dict | `None` | As in `FeatureRanker`. |
| `feature_groups` | dict or `"auto"` | `None` | As in `rank_resampled()`. `"auto"` groups correlated columns with `auto_groups()` on the data given to `fit()`. Any other string raises `ValueError`. |
| `auto_groups_threshold` | float | `0.7` | The correlation cut of `feature_groups="auto"`, above 0 and at most 1. Ignored for other `feature_groups`. |
| `cv` | int, splitter or list | `5` | As in `rank_resampled()`. |
| `n_repeats` | int | `10` | As in `rank_resampled()`. |
| `random_state` | int | `None` | As in `rank_resampled()`. |
| `max_samples` | int or float | `None` | As in `rank_resampled()`. |
| `shadow_seed` | int | `None` | As in `FeatureRanker`. |
| `shadow_mode` | `"joint"` or `"independent"` | `"joint"` | As in `FeatureRanker`. |
| `missing` | `"error"` or `"impute"` | `"error"` | As in `FeatureRanker`. With `"impute"` NaN passes `fit()` and `transform()`; put an imputer after the selector. |
| `categorical` | `"auto"` or list of str | `None` | As in `FeatureRanker`. Selected raw columns come back as given; put an encoder after the selector. |
| `model_input` | `"array"` or `"dataframe"` | `"array"` | As in `FeatureRanker`. |
| `shuffle` | `"rows"`, `"within_groups"`, `"circular"` | `"rows"` | As in `FeatureRanker`. With `"within_groups"` the `groups` given to `fit()` are required. |
| `families` | dict | `None` | As in `FeatureRanker`. |
| `max_categories` | int | `None` | As in `FeatureRanker`. |
| `max_train_samples` | int or float | `None` | As in `rank_resampled()`: each split's models are fit on a subsample of its training rows. |
| `n_jobs` | int | `None` | As in `FeatureRanker`. |
| `verbose` | bool | `False` | Show progress bars. Off by default because pipelines fit often. |

Fitted attributes: `selected_groups_`, `report_` (the resampled report of this fit), `n_features_in_`, `feature_names_in_` (DataFrames), and the standard `get_support()` and `get_feature_names_out()`.

### `ConsensusSelector.fit`

```text
fit(X, y, groups=None, sample_weight=None)
```

Ranks on `X` and `y` only (resampled) and keeps the selected groups. Returns the selector. A failed `fit()` clears earlier fitted state. Raises `ValueError` for an invalid rule or option, or if every model dropped the shadow columns while a shadow rule is set.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | Training features. Names come from DataFrame columns; for arrays they are `x0`, `x1`, and so on. |
| `y` | array | required | Training target. |
| `groups` | array | `None` | Row groups for a group-aware `cv` such as `GroupKFold`. In a pipeline pass `select__groups=...`. With `shuffle="within_groups"` they also confine the shuffles. |
| `sample_weight` | array | `None` | Row weights, as in `rank_resampled()`. In a pipeline enable metadata routing and call `set_fit_request(sample_weight=True)` on the selector. |

### `ConsensusSelector.transform`

```text
transform(X)
```

Keeps the columns of the selected groups, in their original order. If no group was selected it returns zero columns (and `fit()` warned).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | Data with the same columns as `fit()`. DataFrame column labels are compared as strings. |

## Report

A `Report` is a `dict` subclass: indexing, `json.dumps`, pickling and dict equality work as for any dict. The keys are described in [Report](report.md). Its three helpers import pandas or matplotlib only when called.

### `Report.summary`

```text
summary()
```

Returns the [`summarize()`](#summarize) findings for this report, for single-split and resampled reports. It takes no parameters.

### `Report.to_frame`

```text
to_frame()
```

Returns the consensus table (or, for a resampled report, the stability table) as a pandas DataFrame, one row per feature group. The nested per-family and per-split rank columns are left out. Raises `ImportError` without pandas. It takes no parameters.

### `Report.plot`

```text
plot(ax=None)
```

Returns a matplotlib `Axes`. A single-split report shows each voting family's rank per group with the median marked; a resampled report shows each group's range of places across splits with the median marked. The most important group is on top. Raises `ImportError` without matplotlib (`pip install 'featranker[plot]'`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ax` | matplotlib `Axes` | `None` | Axes to draw on. `None` creates a new figure. |

## Functions

### `auto_groups`

```text
auto_groups(X, *, threshold=0.7, feature_names=None, categorical=None)
```

Proposes `feature_groups` from `X` alone, by Spearman correlation with complete linkage. It never sees a target. Returns `{group_name: [column, ...]}` for groups of two or more columns, named by their members joined with `+`. Raises `ValueError` for an invalid `threshold` or data, or a generated name that equals a column or another group. See [Grouping correlated features](guide/selection.md#grouping-correlated-features-with-auto_groups).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | Features, validated like `fit()` input. Use enough rows: with few rows the sample correlation is inflated. |
| `threshold` | float | `0.7` | Keyword-only. Every pair inside a group has absolute correlation of at least this (above 0, at most 1). Lower gives bigger groups. |
| `feature_names` | list of str | `None` | Keyword-only. Column names for non-DataFrame input; default `x0`, `x1`, and so on. |
| `categorical` | `"auto"` or list of str | `None` | Keyword-only. As in `FeatureRanker`. These columns are accepted as text and never grouped. |

### `summarize`

```text
summarize(report)
```

Returns structured findings that apply the [reading guide](report.md#reading-the-report) to a report. The report is not changed. Each finding has `code`, `severity`, `message`, `evidence`, `basis` and `scoring`. Raises `TypeError` if `report` is not a dict and `ValueError` if it lacks the keys of a report. See [Summarize a report](guide/selection.md#summarize-a-report).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `report` | dict | required | A report from `rank_features()` or `rank_resampled()`, or one loaded from the CLI's JSON. |

## Input rules

These are the exact limits behind the tables above. They are checked before any model is fit unless noted.

- **Seeds.** `random_state` and `shadow_seed` are integers from 0 to 2**64 - 1. NumPy integers are accepted. Bools, NumPy durations and generators raise `TypeError`; out-of-range values raise `ValueError`. For integer CV, seeds below 2**32 go directly to the splitter. Wider seeds deterministically generate a 32-bit splitter seed using NumPy SeedSequence. A seed is validated before anything changes, so a refused `fit(random_state=...)` leaves a fitted ranker fitted. Any other refused `fit()` (for example `max_samples=0`) leaves the ranker unfitted, so an earlier fit cannot be ranked by mistake. `rank_resampled()` adds the split index modulo 2**64.
- **Counts.** `n_repeats`, `top_k`, `k`, `max_samples` (as an int), `max_train_samples` and `min_shadow_families` are positive integers below 2**63. `max_categories` is an int of at least 2 and below 2**63. Bools, floats and strings are refused.
- **`n_jobs`.** `None` or any non-zero int; negative counts back from all cores, as in joblib (`-1` = all). `0`, a bool, a float, a string or a NumPy duration raises an error naming `n_jobs`.
- **`max_samples` and `max_train_samples`.** An int is a number of rows (more than there are uses all). A float must be in (0, 1] and is a share, rounded down, at least 1. A subsample whose `sample_weight` values are all zero raises `ValueError`. A `fit()` sample that misses a class raises `ValueError` naming it.
- **`models`.** A non-empty dict. Names are non-empty strings other than `dummy_baseline`. It cannot be combined with a `group` other than `"all"`.
- **`families`.** A mapping of non-empty strings to non-empty strings, copied to a dict. A name that is not a configured model raises `ValueError` in `fit()`, `rank_resampled()` and `ConsensusSelector.fit()`. The name of a built-in model that failed to initialise is accepted; a built-in name outside the chosen `group` or `task` raises.
- **`missing`.** With built-in models, changing `missing` after construction raises `ValueError`; make a new `FeatureRanker`.
- **Options changed after `fit()`.** `missing`, `model_input`, `max_categories`, `shadows`, `shadow_seed`, `shadow_mode` (and with shadows also `shuffle`) raise `ValueError` in `rank_features()` ("call fit() again").
- **`X` and `y`.** `X` is 2-D with at least one row and one column. NaN is rejected unless `missing="impute"`; infinite values always. Targets must be numeric for regression. Evaluation columns must match `fit()` in names and order. Date and time columns are rejected.
- **`feature_names`.** Unique, non-empty. Required for NumPy input.
- **`sample_weight`.** 1-D, finite, non-negative, one per row, positive sum. Anything else raises `ValueError`.
- **`shuffle_groups` and `groups`.** One label per row. With `"within_groups"` labels must be sortable (`None`, NaN and mixed types raise `ValueError`), and if every group has one row the call raises `ValueError`.
- **`scoring`.** A list needs unique names. An empty list, a repeated name or an invalid entry raises before any model is ranked.
- **`cv`.** Two splits with identical train and evaluation indices raise `ValueError`.
- **`ks` in `ablate()`.** A non-empty list of ints from 1 to the number of consensus groups; de-duplicated and sorted. Anything else raises `ValueError`.
- **`feature_groups`.** Empty groups, unknown features, duplicate membership, overlaps and group-name collisions raise errors.
- **`ablate()` and the report.** It raises `ValueError` for an unfitted ranker, a resampled report, an empty consensus, other training data than `run["train_data_sha256"]` (skipped when that is `null`), other weights than the recorded fingerprints, a voting model that the ranker's latest `fit()` did not fit, a voting model of another class (`models[m]["class"]`), another `model_input` or other categorical columns than the ranker's, or other parameters than the ranker's model of that name. When training fingerprints are available, the latest fit must match the report's training data and weights. Non-scalar parameters are compared through `models[m]["param_hashes"]`, and nested component classes through `models[m]["components"]`. A report without those keys, without `params` or without a `run` block is not checked on them, and the ranker must still have used every training row. A report made under another scikit-learn version is compared on the parameters both versions have and only warns (`UserWarning`) when they differ. A ranker with user models pickled before 1.22 refits a clone of its fitted model, not the current template.

Training evidence requires a fit with `evidence=True`. Refitting discards evidence retained from the previous training set.
