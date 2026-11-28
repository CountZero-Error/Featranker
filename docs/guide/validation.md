# Validation

English · [简体中文](../zh-CN/guide/validation.md)

This page shows how to check how stable a ranking is, how to rank time-ordered and grouped rows, and how to keep selection honest with nested validation.

[Manual](../README.md) | [Data](data.md) | [Noise and evidence](noise-and-evidence.md) | [Selection](selection.md) | [Reference](../reference.md)

## Resampled ranking

One split gives one ranking. `rank_resampled()` fits and ranks on every cross-validation split of the data you pass, then reports how much each group's rank moves between splits. Use it when you want to know whether a ranking is stable.

```python
from sklearn.datasets import load_diabetes

from featranker import FeatureRanker

data = load_diabetes()
ranker = FeatureRanker(task="reg", group="fast", verbose=False)
report = ranker.rank_resampled(
    data.data,
    data.target,
    feature_names=list(data.feature_names),
    scoring="neg_mean_absolute_error",
    cv=5,
    n_repeats=3,
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
  s5  median place 1  range 1-1  top 3 in 100% of splits
 bmi  median place 2  range 1-2  top 3 in 100% of splits
  bp  median place 3  range 3-8  top 3 in 60% of splits
 sex  median place 4  range 3-9  top 3 in 20% of splits
```

A group's place in a split is its position in that split's sorted consensus. Each stability row gives the median place, the range across splits, and `top_k_share`, the share of splits in which the group was in the top `top_k`. Groups tied on both `median_rank` and `mean_rank` share the better place, so a split can have more than `top_k` top groups.

- `cv` is a number of folds (shuffled; stratified for `task="clf"`), any scikit-learn splitter such as `RepeatedKFold` or `GroupKFold`, or a list of `(train_indices, eval_indices)` pairs. Two splits with identical train and eval indices raise `ValueError`, since a repeated split would count as an independent trial in the shadow test.
- Each split uses a new ranker with this ranker's settings, so the ranker you call it on is not changed.
- A split that fails entirely is recorded in `failures` with `stage: "split"` and the run continues. It raises only if no split ranks.
- Invalid `cv`, `top_k`, scorer, `n_repeats` or feature groups raise before any model is fit.
- `sample_weight`, `max_samples` and `max_train_samples` are applied per split. Split `i` uses seed `(random_state + i)` modulo 2**64.
- See [Report](../report.md#resampled-reports) for the output.

> [!WARNING]
> Pass development data only and keep your final test set out. The splits here are internal validation. Choosing features from these results and then scoring a model on the same data overstates its performance. Fit learned preprocessing within each training split: put it inside custom model pipelines, use `missing="impute"` ([Data](data.md#missing-values)) and `categorical=` ([Data](data.md#categorical-features)), or use a manual split workflow. Do not fit it on all development rows before this call.

> [!NOTE]
> One call costs about `n_splits` times a normal `fit()` plus `rank_features()`. With `group="all"` and `cv=10` that is ten full runs. Start with `cv=5`, `group="fast"` or fewer `n_repeats`.

## Time series and grouped data

For ordered or clustered rows, pass the splitter that matches how the model will be used. `TimeSeriesSplit` is for rows in time order, and `GroupKFold` with `groups=` is for rows that belong together, such as several rows per patient. A model is then never evaluated on rows from before its training rows or from the same group.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

rng = np.random.default_rng(0)
n_rows = 300
X = rng.normal(size=(n_rows, 4))
y = 2.0 * X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n_rows)
names = ["strong", "weak", "noise_1", "noise_2"]
patients = np.repeat(np.arange(30), 10)  # 10 rows per patient

ordered = FeatureRanker(task="reg", group="fast", shuffle="circular", verbose=False)  # see "Shuffling dependent rows"
grouped = FeatureRanker(task="reg", group="fast", shuffle="within_groups", verbose=False)

by_time = ordered.rank_resampled(  # rows in time order: train on the past, evaluate on the future
    X, y, feature_names=names, scoring="r2", cv=TimeSeriesSplit(n_splits=4),
    n_repeats=3, random_state=0,
)
print("time:", [row["feature_group"] for row in by_time["stability"]])

by_patient = grouped.rank_resampled(  # rows of one patient never sit on both sides of a split
    X, y, feature_names=names, scoring="r2", cv=GroupKFold(n_splits=5), groups=patients,
    n_repeats=3, random_state=0,
)
print("patient:", [row["feature_group"] for row in by_patient["stability"]])

pipeline = Pipeline([  # the same splitters in a pipeline: selection never sees the rows it is scored on
    ("select", ConsensusSelector(
        "reg", "r2", k=2, group="fast", cv=GroupKFold(n_splits=5), n_repeats=3, random_state=0,
        shuffle="within_groups")),
    ("model", Ridge()),
])
scores = cross_val_score(
    pipeline, X, y, groups=patients, cv=GroupKFold(n_splits=3),
    params={"select__groups": patients},
)
print(f"pipeline R2 per outer fold: {np.round(scores, 2)}")
```

```text
time: ['strong', 'weak', 'noise_1', 'noise_2']
patient: ['strong', 'weak', 'noise_2', 'noise_1']
pipeline R2 per outer fold: [0.98 0.98 0.98]
```

The splitter only controls how rows are divided. By default, shadows and permutations shuffle rows freely, which ignores dependence between neighbouring rows or between rows of one group. The `shuffle` option makes both respect that structure. Without it, treat shadows and the hits test as less reliable on such data.

## Shuffling dependent rows

`FeatureRanker(shuffle=...)` and `ConsensusSelector(shuffle=...)` set how permutation importance and shadow columns shuffle rows.

| `shuffle` | Behaviour |
|---|---|
| `"rows"` (default) | Rows are shuffled freely. |
| `"within_groups"` | A column's values move only among rows with the same group label. Pass one label per row as `shuffle_groups` to `fit()` (shadows) and to `rank_features()` (permutations). `rank_resampled()` and `ConsensusSelector.fit()` use their `groups`, sliced per split. A row in a group of one keeps its value; if every group has one row the call raises `ValueError`. With few rows per group there are few distinct shuffles, so importances are noisier. Labels must be sortable: `None`, NaN and mixed types raise `ValueError`. |
| `"circular"` | Each shuffle is a cyclic shift by a random offset from 1 to n-1, which keeps a series' autocorrelation. It assumes rows in time order (as `TimeSeriesSplit` does); on unordered rows it is just a shift. `rank_resampled()` sorts each split's rows into their original order first. |

`shuffle_groups` is required with `"within_groups"` (also when `shadows=False`, in both `fit()` and `rank_features()`, though `fit()` then ignores it) and is an error with another mode. A report records `shuffle` when it is not `"rows"`. The CLI has no group column and always uses `"rows"`.

Joint shadows share one offset (or one within-group order); `shadow_mode="independent"` draws one per column. With `max_samples`, permutations shuffle the sampled rows only. Under `"within_groups"` the subsample is made of whole groups, drawn until the target row count is reached, so it can slightly exceed it (`n_eval_rows` records the count). Under `"circular"` the sampled rows keep their order, but subsampling thins the series, so a shift no longer moves values by whole time steps. Correlation and SHAP evidence do not shuffle and are unchanged.

For single fits:

```python
ranker = FeatureRanker(task="reg", group="fast", shuffle="within_groups", verbose=False)
ranker.fit(X[:150], y[:150], feature_names=names, shuffle_groups=patients[:150])
report = ranker.rank_features(
    X[150:], y[150:], scoring="r2", shuffle_groups=patients[150:], random_state=0,
)
print(report["shuffle"], [row["feature_group"] for row in report["consensus"]])
```

```text
within_groups ['strong', 'weak', 'noise_1', 'noise_2']
```

### What within_groups measures

Moving values only among rows of the same group tests how much the model relies on variation within groups. It cannot test reliance on variation between groups. A predictor that is constant within each group (a patient-level value repeated on every row) is not changed by the shuffle, so its importance is 0 by construction, however strongly it drives the target. In a check with 40 patients of 4 rows, a linear model and a patient-level `x` with `y = 3x` plus small noise, the importance of `x` was about 2 under free shuffling and exactly 0.0 under `"within_groups"`.

To measure group-level effects use `shuffle="rows"` with a grouped splitter (`GroupKFold`): the shuffle is free, but rows of one group never sit on both sides of a split.

With `"within_groups"`, `rank_features()` records `constant_within_groups`: the sorted feature groups whose every column has one value (a missing value counts as a value) within every shuffle group on the scored rows, after `max_samples`. The key is absent with the other modes. A non-empty list also raises a `UserWarning`.

- Each split report of `rank_resampled()` carries its own list. The resampled report has a top-level `constant_within_groups` with the union, and `summarize()` adds a warning that lists, per group, the splits in which it was constant (evidence `splits`).
- A group with a column that varies is not listed even if other columns are constant. Near-constant predictors, which vary in only a few groups, are not flagged either. The list catches only the exact case.

## Nested validation

Choosing features from a ranking of all your data and then cross-validating a model on them overstates its performance. Rank inside the training part of each fold instead.

- Put a [`ConsensusSelector`](selection.md#consensusselector-in-a-pipeline) in a `Pipeline` under `cross_val_score`. It reranks on each training fold.
- To choose `k` honestly, put `GridSearchCV(pipe, {"select__k": [2, 3, 5]})` inside `cross_val_score` (nested cross-validation). A fixed `k` picked from an [ablation curve](selection.md#ablation-curve) and then cross-validated is still optimistic, because the same rows chose it.
- Report final performance on a test set that no step of the procedure touched.
