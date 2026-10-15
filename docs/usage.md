# FEATRANKER ▍ USAGE

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <b>Usage</b> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <a href="models.md">Models</a></sub>

## ▍ API

The API is one class: `FeatureRanker(...)`, then `.fit(...)`, then `.rank_features(...)`. [`.rank_resampled(...)`](#resampled-ranking) repeats that over cross-validation splits, and [`summarize(report)`](#summarize) reads the result.

### `FeatureRanker(task, group="all", models=None, n_jobs=None, verbose=True)`

| Parameter | Default | Meaning |
|---|---|---|
| `task` | required | `"clf"` for classification or `"reg"` for regression. |
| `group` | `"all"` | Built-in model group: `"linear"`, `"tree"`, or `"all"`. Groups select estimators; model families determine consensus votes. See [Models](models.md). |
| `models` | `None` | Non-empty dict of `{name: estimator}` to use instead of the built-in models. Names must be non-empty strings other than `dummy_baseline`. Cannot be combined with a `group` other than `"all"`. See [Your own models](#your-own-models). |
| `n_jobs` | `None` | Parallel workers over models. `None` or `1` is sequential, `-1` uses all cores, `0` is invalid. See [Parallel models](#parallel-models). |
| `verbose` | `True` | Show progress bars during model setup, `fit()`, and `rank_features()`. `False` hides them; warnings are still shown. |

### `.fit(X, y, feature_names=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X` | required | Training features: 2-D NumPy array or pandas DataFrame of finite numbers, with at least one row and one column. NaN and infinite values are rejected. |
| `y` | required | Training target, one value per row. Class labels may be numbers or strings; regression targets must be numeric. |
| `feature_names` | `None` | Unique, non-empty column names. Required for NumPy input; taken from the columns for a DataFrame. |

Returns the ranker itself. Raises if no model fits.

### `.rank_features(X_eval, y_eval, *, scoring, feature_groups=None, n_repeats=10, random_state=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X_eval` | required | Evaluation features with the same columns, in the same order, as `fit()`. Keeping them separate from the training data is up to you. |
| `y_eval` | required | Evaluation target. Class labels must have appeared in training. |
| `scoring` | required, keyword-only | scikit-learn scorer name or a callable `(estimator, X, y) -> float` where higher is better. See [Choosing a scorer](#choosing-a-scorer). |
| `feature_groups` | `None` | Dict of `{group_name: [feature, ...]}` to shuffle together. Ungrouped features are ranked on their own. See [Grouped permutation](#grouped-permutation). |
| `n_repeats` | `10` | Shuffles per feature group; a positive integer. More repeats give a more precise average importance and take longer. |
| `random_state` | `None` | Integer seed for the shuffles. Set it for reproducible reports. |

Returns the report dict described in [Report](report.md). Raises if every model fails ranking.

<a id="choosing-a-scorer"></a>

## ▍ Choosing task, group, and scorer

**Task.** `"clf"` when the target is a set of classes, `"reg"` when it is a number.

**Group.** `"all"` (the default) gives the broadest consensus. `"tree"` holds tree ensembles, which capture non-linear effects and interactions. `"linear"` holds standardized linear, discriminant, naive Bayes, and kernel models (including RBF `SVR`), so it is not purely additive. See [Models](models.md).

**Scorer.** Pick one that matches the problem. scikit-learn scorers are higher-is-better, so error metrics come with a `neg_` prefix.

| Problem | Good scorer names | Watch out for |
|---|---|---|
| Binary classification | `"accuracy"`, `"balanced_accuracy"`, `"f1"`, `"roc_auc"` | `"f1"` treats the second class in sorted order as positive. See [Labels](#labels). |
| Multiclass classification | `"accuracy"`, `"balanced_accuracy"`, `"f1_macro"`, `"roc_auc_ovr"` | `"f1"` and `"roc_auc"` are binary-only and fail every model, including the dummy. `"roc_auc_ovr"` needs `predict_proba`, so `linear_svm`, `sgd_classifier`, `ridge_classifier`, `perceptron`, and `passive_aggressive` fail. |
| Regression | `"r2"`, `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"` | |

A model that cannot produce what the scorer needs is listed in the report's `failures` and the others still rank. If no model can, `rank_features()` raises.

<a id="your-own-models"></a>

## ▍ Your own models

Pass `models` to rank with your own estimators instead of the built-in registry. Estimators are cloned at `fit()`, so yours are never modified, and a `Pipeline` keeps its preprocessing fit on training data only.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker

ranker = FeatureRanker(
    task="reg",
    models={
        "scaled_ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "forest": RandomForestRegressor(n_estimators=200, random_state=0),
    },
)
```

> [!IMPORTANT]
> Your estimators receive a float NumPy array, even when you pass a DataFrame. Inside a pipeline, select columns by position (for example `ColumnTransformer([("scale", StandardScaler(), [0, 1])])`), not by name. Categorical values and missing values must be encoded or removed before the data reaches FeatRanker.

<a id="parallel-models"></a>

## ▍ Parallel models

`FeatureRanker(..., n_jobs=2)` runs the per-model work in `fit()` and `rank_features()` in parallel with joblib. `None` runs sequentially and `-1` uses all cores.

- Reports match `n_jobs=None`, with models in the same order, when every estimator is deterministic or seeded, the scorer is deterministic, and `random_state` is a fixed integer. Every built-in model with randomness is seeded (`random_state=42`) since 0.4.1. An unseeded custom estimator, such as `RandomForestRegressor()` without `random_state`, can give different reports on every run, parallel or not.
- Per-model failures are recorded exactly as in the sequential path.
- With any parallel setting (`n_jobs` other than `None` or `1`, including `-1`), custom scorers and estimators are sent to worker processes, so everything they hold must be serializable. Lambdas and local functions are fine.

## ▍ Command line

The CLI takes separate numeric training and evaluation CSV files.

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

Both CSV files need headers, the same feature columns in the same order, and the target column. Feature values must be numeric; targets may be numeric or strings. Add `--quiet` to hide progress bars. Run `featranker --help` for every flag.

<details>
<summary><b>Example input files</b></summary>

`train.csv`

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.2,8.1,1,0,12.4
2.1,7.3,0,1,14.8
3.4,6.8,1,0,18.2
4.0,5.9,0,1,20.1
```

`eval.csv`

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.7,7.8,1,0,13.6
3.8,6.2,0,1,19.4
```

`groups.yaml`

```yaml
segment:
  - segment_red
  - segment_blue
```

</details>

<a id="resampled-ranking"></a>

## ▍ Resampled ranking

One split gives one ranking. `rank_resampled()` fits and ranks on every cross-validation split of the data you pass, then reports how much each feature group's rank moves between splits.

```python
from sklearn.datasets import load_diabetes

from featranker import FeatureRanker

data = load_diabetes()
ranker = FeatureRanker(task="reg", verbose=False)
report = ranker.rank_resampled(
    data.data,
    data.target,
    feature_names=list(data.feature_names),
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=0,
    top_k=3,
)

for row in report["stability"][:4]:
    print(
        f"{row['feature_group']:>4}  median place {row['median_rank']:g}  "
        f"range {row['min_rank']}-{row['max_rank']}  "
        f"top 3 in {row['top_k_share']:.0%} of splits"
    )
```

```text
  s5  median place 1  range 1-2  top 3 in 100% of splits
 bmi  median place 2  range 1-2  top 3 in 100% of splits
  bp  median place 3  range 3-3  top 3 in 100% of splits
  s3  median place 4  range 4-7  top 3 in 0% of splits
```

Output shown with the `xgboost` and `catboost` extras installed. A group's *place* in a split is its position in that split's sorted consensus. `s5` and `bmi` swap first and second place, `bp` is third in every split, and `s3` moves between fourth and seventh.

### `.rank_resampled(X, y, *, scoring, cv=5, groups=None, feature_names=None, feature_groups=None, n_repeats=10, random_state=None, top_k=5)`

| Parameter | Default | Meaning |
|---|---|---|
| `X`, `y`, `feature_names` | | All development data, with the same rules as `fit()`. Each split takes its own training and evaluation rows from it. |
| `scoring`, `feature_groups`, `n_repeats` | | Same as `rank_features()`. |
| `cv` | `5` | An integer number of folds (shuffled; stratified for `task="clf"`), any scikit-learn splitter such as `RepeatedKFold` or `GroupKFold`, or a list of `(train_indices, eval_indices)` pairs. |
| `groups` | `None` | Passed to the splitter's `split()`, for example row groups for `GroupKFold`. |
| `random_state` | `None` | Seeds integer-`cv` shuffling. Split `i` uses permutation seed `random_state + i`. |
| `top_k` | `5` | For `top_k_share`: a group counts as top in a split when its place in that split's consensus is at most `top_k`. Groups tied on both `median_rank` and `mean_rank` share the better place, so a split can have more than `top_k` top groups. |

Each split uses a new ranker with this ranker's settings, so the ranker you call it on is not changed. A split that fails entirely is recorded in `failures` with `stage: "split"` and the run continues; it raises only if no split ranks. Invalid `cv`, `top_k`, scorer, `n_repeats`, or feature groups raise before any model is fit. See [Resampled reports](report.md#resampled-reports) for the output.

**Cost.** One call costs about `n_splits` times a normal `fit()` plus `rank_features()`. With `group="all"` and `cv=10`, that is ten full runs; start with `cv=5`, a smaller `group`, or fewer `n_repeats`.

> [!IMPORTANT]
> Pass development data only and keep your final test set out. The splits here are internal validation. Choosing features from these results and then scoring a model on the same data overstates its performance.
>
> Fit learned preprocessing within each training split. Put transformations compatible with FeatRanker's finite numeric input contract inside custom model pipelines. For raw missing or categorical data requiring learned preprocessing, use a manual split workflow; do not fit preprocessing on all development rows before this call.

<a id="summarize"></a>

## ▍ Summarize a report

`summarize(report)` applies the [reading guide](report.md#reading-the-report) for you. It returns a list of findings and does not change the report. It works on the dict from `rank_features()` or on a report loaded from the CLI's JSON. For a resampled report, summarize each `report["splits"][i]["report"]`; passing the resampled report itself raises `ValueError`.

```python
from featranker import summarize

for finding in summarize(report):
    print(finding["severity"], finding["code"], "-", finding["message"])
```

Each finding is a dict with `code`, `severity` (`"info"` or `"warning"`), `message`, and JSON-safe `evidence`. Findings come in this order:

| Code | Severity | When |
|---|---|---|
| `gate_off` | warning | The dummy baseline failed, so every model voted. |
| `empty_consensus` | warning | No model beat the dummy. |
| `failures` | warning for `fit` and `ranking`; info for `initialization` (usually a missing optional library) | One finding per failed stage, listing the models. |
| `few_families` | warning | Fewer than 3 model families voted. |
| `tied_groups` | info | Groups share the same `median_rank`; one finding per tie. `mean_rank` still orders them, and the finding does not measure ranking uncertainty. |
| `within_noise` | info | In most voting models, the group's mean importance is no greater than its standard deviation across shuffles (`mean <= std`). This flags large relative spread or non-positive importance; it does not prove shuffling left the score unchanged. With `n_repeats=1` the spread is 0, so it only catches importances of 0 or less. |

An empty list means nothing stood out, not that the ranking is proven. The checks cover one evaluation set.

<a id="grouped-permutation"></a>

## ▍ Grouped permutation

Related columns can be shuffled as one unit:

```python
feature_groups = {
    "segment": ["segment_red", "segment_blue", "segment_other"],
    "channel": ["channel_web", "channel_store", "channel_partner"],
}
```

Every column in a multi-column group receives the same row permutation during a repeat. This keeps the relationships inside the block while breaking its relationship with the target.

Features left out of `feature_groups` become singleton groups. Empty groups, unknown features, duplicate membership, overlaps, and group-name collisions raise errors.

<a id="labels"></a>

## ▍ Data handling and labels

FeatRanker does not impute, encode features, or select features. It changes data values in two places only:

- **Linear scaling.** Built-in linear models sit behind a `StandardScaler` fit on training data only. Tree models and your own models get the data as given.
- **Classification labels.** For `task="clf"`, `fit()` encodes the labels to `0..n_classes-1` with a `LabelEncoder`, because some estimators (XGBoost) require it. Every estimator and scorer, including your own, sees the encoded labels. `ranker.label_encoder.classes_` maps each code back to the original label; code `i` is `classes_[i]`.

So a custom scorer must refer to classes by code. For example, to score F1 for the original class `"case"`:

```python
from sklearn.metrics import f1_score, make_scorer

case_code = list(ranker.label_encoder.classes_).index("case")
scorer = make_scorer(f1_score, pos_label=case_code)
report = ranker.rank_features(X_eval, y_eval, scoring=scorer)
```

Non-finite baseline, shuffled, or importance values fail that model's ranking. They never enter successful reports or the consensus.

## ▍ Troubleshooting

| Error or warning | Cause | Fix |
|---|---|---|
| `X must contain only finite values; impute or drop missing ...` | NaN or infinite values in the features. | Impute or drop them before calling FeatRanker. |
| `feature_names are required for NumPy input.` | `X` is a NumPy array and no names were given. | Pass `feature_names=[...]`, or pass a DataFrame. |
| `Evaluation feature names and order must match fit().` | `X_eval` columns differ from training. | Use the same columns in the same order. |
| `y contains labels not seen during fit(): ...` | The evaluation set has a class the training set lacks. | Stratify the split so every class appears in training. |
| `Regression targets must be numeric.` | `task="reg"` with a text target. | Convert the target to numbers, or use `task="clf"`. |
| `n_repeats must be a positive integer.` | `n_repeats` is 0, negative, or not an `int`. | Use 1 or more. |
| `No models fit successfully.` | Every model failed in `fit()`. The message lists each failure. | Read the listed estimator errors; check the data and any custom estimators. |
| `No models ranked successfully.` | Every model failed in `rank_features()`. The message lists each failure. | Check that the scorer suits the task and the models' outputs (see [Choosing a scorer](#choosing-a-scorer)), and look for non-finite scores. |
| `No model beat the dummy baseline, so the consensus is empty.` | No model did better than a model that ignores the features. | See [Reading the report](report.md#reading-the-report). |
| `Skipping model '<name>': ...` | An optional library (XGBoost, CatBoost) is missing or broken. | Install the extra, or ignore it; other models still run. |
| Pickling error with `n_jobs` | A custom estimator or scorer cannot be sent to worker processes. | Make sure everything it holds can be serialized by joblib (locks and connections, for example, cannot), or use `n_jobs=None`. |
