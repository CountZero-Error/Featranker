# FEATRANKER ▍ USAGE

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <b>Usage</b> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <a href="models.md">Models</a> &nbsp;│&nbsp; <a href="api.md">API</a></sub>

## ▍ API

The API is one class: `FeatureRanker(...)`, then `.fit(...)`, then `.rank_features(...)`, which returns a [`Report`](#working-with-reports). [`.rank_resampled(...)`](#resampled-ranking) repeats that over cross-validation splits, [`ConsensusSelector`](#feature-selection) turns the ranking into a scikit-learn pipeline step, [`summarize(report)`](#summarize) reads the result, and [`to_prompt` / `explain`](#llm-interpretation) hand it to an LLM with checked citations (deprecated since 1.1). `evidence=True` adds [mechanism evidence](#evidence) to the report.

### `FeatureRanker(task, group="all", models=None, n_jobs=None, verbose=True, shadows=False, shadow_seed=None, evidence=False)`

| Parameter | Default | Meaning |
|---|---|---|
| `task` | required | `"clf"` for classification or `"reg"` for regression. |
| `group` | `"all"` | Built-in model group: `"linear"`, `"tree"`, `"fast"`, or `"all"`. Groups select estimators; model families determine consensus votes. See [Models](models.md). |
| `models` | `None` | Non-empty dict of `{name: estimator}` to use instead of the built-in models. Names must be non-empty strings other than `dummy_baseline`. Cannot be combined with a `group` other than `"all"`. See [Your own models](#your-own-models). |
| `n_jobs` | `None` | Parallel workers over models. `None` or `1` is sequential, `-1` uses all cores, `0` is invalid. See [Parallel models](#parallel-models). |
| `verbose` | `True` | Show progress bars during model setup, `fit()`, and `rank_features()`. `False` hides them; warnings are still shown. |
| `shadows` | `False` | Train and rank with a shuffled copy of every feature, to show which groups beat noise. See [Shadow features](#shadow-features). |
| `shadow_seed` | `None` | Non-negative integer seed for the shadow shuffles, or `None` for different shadows on every run. |
| `evidence` | `False` | Keep a copy of the training data and add train scores, overfit gaps, correlated groups, and (with `featranker[shap]`) a SHAP summary to the report. See [Evidence](#evidence). |

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
| `random_state` | `None` | Integer seed for permutation-importance shuffles. Reproducible reports also require deterministic or seeded estimators and deterministic scoring; with shadows enabled, fix `shadow_seed` too. |

Returns the report dict described in [Report](report.md). Raises if every model fails ranking.

<a id="working-with-reports"></a>

## ▍ Working with reports

`rank_features()` and `rank_resampled()` return a `Report`. It is a plain `dict` (indexing, `json.dumps`, pickling, and comparison with dicts all work as before) with three helpers:

- `report.summary()`: the [`summarize()`](#summarize) findings (single-split reports).
- `report.to_frame()`: the consensus table, or the stability table for a resampled report, as a pandas DataFrame (needs pandas).
- `report.plot(ax=None)`: returns a matplotlib `Axes` (needs `pip install 'featranker[plot]'`). For a single-split report it shows each voting family's rank per group with the median marked; for a resampled report, each group's range of places across splits with the median marked. The most important group is at the top.

```python
import matplotlib

matplotlib.use("Agg")  # no window; remove in a notebook

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
ranker = FeatureRanker(task="reg", group="tree", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)

print(report.to_frame()[["feature_group", "median_rank", "n_families"]].head(4).to_string(index=False))
print([finding["code"] for finding in report.summary()])
report.plot().figure.savefig("ranking.png", bbox_inches="tight")
```

```text
feature_group  median_rank  n_families
          bmi         1.50           2
           s5         1.50           2
           bp         3.25           2
           s2         3.75           2
['few_families', 'tied_groups', 'tied_groups', 'within_noise']
```

Output shown with the `xgboost` and `catboost` extras installed. `ranking.png` shows `bmi` and `s5` sharing the top, with the bagging and boosting families' ranks as dots around each median.

<a id="choosing-a-scorer"></a>

## ▍ Choosing task, group, and scorer

**Task.** `"clf"` when the target is a set of classes, `"reg"` when it is a number.

**Group.** `"all"` (the default) gives the broadest consensus. `"tree"` holds tree ensembles, which capture non-linear effects and interactions. `"linear"` holds standardized linear, discriminant, naive Bayes, and kernel models (including RBF `SVR`), so it is not purely additive. `"fast"` keeps one representative per family from both. See [Speed](#speed) and [Models](models.md).

**Scorer.** Pick one that matches the problem. scikit-learn scorers are higher-is-better, so error metrics come with a `neg_` prefix.

| Problem | Good scorer names | Watch out for |
|---|---|---|
| Binary classification | `"accuracy"`, `"balanced_accuracy"`, `"f1"`, `"roc_auc"` | `"f1"` treats the second class in sorted order as positive. See [Labels](#labels). |
| Multiclass classification | `"accuracy"`, `"balanced_accuracy"`, `"f1_macro"`, `"roc_auc_ovr"` | `"f1"` and `"roc_auc"` are binary-only and fail every model, including the dummy. `"roc_auc_ovr"` needs `predict_proba`, so `linear_svm`, `sgd_classifier`, `ridge_classifier`, `perceptron`, and `passive_aggressive` fail. |
| Regression | `"r2"`, `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"` | |

A model that cannot produce what the scorer needs is listed in the report's `failures` and the others still rank. If no model can, `rank_features()` raises.

<a id="speed"></a>

## ▍ Speed

`group="fast"` fits one built-in model per family: 7 for classification (22 in `"all"`) and 5 for regression (18 in `"all"`), all from scikit-learn. Every candidate family has a representative, so the consensus is built from about a third of the models. A representative can fail or score below the dummy baseline, in which case its family does not vote, so the set of voting families can differ from `"all"`, where a family's other members may still vote. Check the families in `report["models"]` and the report's `failures`. Which models run is marked in the **Fast** column of [Models](models.md). It works the same in `rank_resampled()`, `ConsensusSelector(group="fast")`, and the CLI (`--group fast`).

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

for group in ("all", "fast"):
    ranker = FeatureRanker(task="reg", group=group, verbose=False)
    ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
    report = ranker.rank_features(X_eval, y_eval, scoring="r2", random_state=42)
    families = {m["family"] for m in report["models"].values()}
    top = [row["feature_group"] for row in report["consensus"][:3]]
    print(f"{group:>4}: {len(report['models'])} models, {len(families)} families, top 3 {top}")
```

```text
 all: 18 models, 5 families, top 3 ['s5', 'bmi', 'bp']
fast: 5 models, 5 families, top 3 ['s5', 'bmi', 'bp']
```

Output shown with the `xgboost` and `catboost` extras installed; without them `"all"` runs 16 models. The trade-off is a thinner vote: each family is one model instead of the median of several, so one unlucky model moves its family's rank more. `"all"` stays the default. `svr_regression`, the regression kernel family's only model, takes time quadratic in the number of rows; on large data pass your own `models=` instead (`group="fast"` cannot be combined with `models=`).

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

- Reports match `n_jobs=None`, with models in the same order, when every estimator is deterministic or seeded, the scorer is deterministic, and `random_state` is a fixed integer. Every built-in model with randomness is seeded (`random_state=42`) since 0.4.1. An unseeded custom estimator, such as `RandomForestRegressor()` without `random_state`, can give different reports on every run, parallel or not. With `shadows=True`, reproducibility also requires a fixed non-negative integer `shadow_seed`; fixing `random_state` alone does not seed shadow construction.
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

Both CSV files need headers, the same feature columns in the same order, and the target column. Feature values must be numeric; targets may be numeric or strings. Add `--quiet` to hide progress bars and `--evidence` to add [evidence](#evidence) to the report. `--feature-groups` takes a YAML file, or the word `auto` to group correlated columns of the training CSV with [`auto_groups()`](#auto-groups) (a file literally named `auto` must be written `./auto`). Run `featranker --help` for every flag.

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

**Cost.** One call costs about `n_splits` times a normal `fit()` plus `rank_features()`. With `group="all"` and `cv=10`, that is ten full runs; start with `cv=5`, a smaller `group` (such as `"fast"`), or fewer `n_repeats`.

> [!IMPORTANT]
> Pass development data only and keep your final test set out. The splits here are internal validation. Choosing features from these results and then scoring a model on the same data overstates its performance.
>
> Fit learned preprocessing within each training split. Put transformations compatible with FeatRanker's finite numeric input contract inside custom model pipelines. For raw missing or categorical data requiring learned preprocessing, use a manual split workflow; do not fit preprocessing on all development rows before this call.

### Time series and grouped data

For ordered or clustered rows, pass the splitter that matches how the model will be used. `rank_resampled()` and `ConsensusSelector` accept any scikit-learn splitter (`TimeSeriesSplit` for rows in time order, `GroupKFold` with `groups=` for rows that belong together, such as several rows per patient), so a model is never evaluated on rows from before its training rows or from the same group.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector, FeatureRanker

rng = np.random.default_rng(0)
n_rows = 300
X = rng.normal(size=(n_rows, 4))
y = 2.0 * X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n_rows)
names = ["strong", "weak", "noise_1", "noise_2"]
patients = np.repeat(np.arange(30), 10)  # 10 rows per patient

ranker = FeatureRanker(task="reg", group="fast", verbose=False)

# Rows are in time order: every split trains on the past and evaluates on the future.
by_time = ranker.rank_resampled(
    X, y, feature_names=names, scoring="r2", cv=TimeSeriesSplit(n_splits=4),
    n_repeats=3, random_state=0,
)
print("time:", [row["feature_group"] for row in by_time["stability"]])

# Rows from one patient never sit on both sides of a split.
by_patient = ranker.rank_resampled(
    X, y, feature_names=names, scoring="r2", cv=GroupKFold(n_splits=5), groups=patients,
    n_repeats=3, random_state=0,
)
print("patient:", [row["feature_group"] for row in by_patient["stability"]])

# The same splitters in a pipeline, so selection never sees the rows it is scored on.
pipeline = Pipeline([
    ("select", ConsensusSelector(
        "reg", "r2", k=2, group="fast", cv=GroupKFold(n_splits=5), n_repeats=3, random_state=0)),
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

The splitter only controls how rows are divided. Shadow features shuffle rows within each training and evaluation set, and permutation importance shuffles rows within the evaluation set, which ignores any dependence between neighbouring rows (autocorrelation). Treat shadows and the hits test as less reliable on such data. This is a known limitation listed in the roadmap (`openspec/config.yaml`).

<a id="shadow-features"></a>

## ▍ Shadow features

A rank says a group is more important than others, not that it carries any signal. With `shadows=True`, every model also gets a *shadow* copy of all features: the same rows in a shuffled order. The shadow block preserves feature values and within-row relationships while randomly reassigning rows. It provides a randomized reference; a particular shuffle can retain chance association with the target. A real group is `above_shadow` in a model when its mean importance is greater than the strongest shadow's.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker, summarize

data = load_diabetes()
rng = np.random.default_rng(0)
X = np.hstack([data.data, rng.normal(size=(len(data.data), 2))])  # two pure-noise columns
names = list(data.feature_names) + ["noise_1", "noise_2"]
X_train, X_eval, y_train, y_eval = train_test_split(
    X, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", shadows=True, shadow_seed=0, verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)

for row in report["consensus"]:
    print(
        f"{row['feature_group']:>7}  median rank {row['median_rank']:4.1f}  "
        f"family-weighted shadow-hit share {row['above_shadow_share']:.0%}"
    )
```

```text
    bmi  median rank  1.2  family-weighted shadow-hit share 97%
     s5  median rank  2.0  family-weighted shadow-hit share 100%
     bp  median rank  3.2  family-weighted shadow-hit share 86%
     s4  median rank  5.8  family-weighted shadow-hit share 47%
     s3  median rank  6.0  family-weighted shadow-hit share 41%
     s6  median rank  6.5  family-weighted shadow-hit share 31%
     s1  median rank  7.0  family-weighted shadow-hit share 38%
noise_2  median rank  7.0  family-weighted shadow-hit share 12%
    sex  median rank  7.0  family-weighted shadow-hit share 12%
     s2  median rank  8.5  family-weighted shadow-hit share 38%
    age  median rank  9.5  family-weighted shadow-hit share 25%
noise_1  median rank 10.2  family-weighted shadow-hit share 0%
```

Output shown with the `xgboost` and `catboost` extras installed. The two pure-noise columns have family-weighted shadow-hit shares of at most 12%; `bmi`, `s5`, and `bp` are at 86% or more.

- `above_shadow_share` is family-weighted: each voting family contributes the fraction of its members in which the group beat the strongest shadow.
- `above_shadow` is a relative comparison, not proof of useful signal. If `shadow_max` is negative, even negative real importance can beat it. Inspect the real mean, shuffle spread, and evaluation score alongside the share.
- It is descriptive, not a p-value. Families are not independent trials, and a share below 0.5 does not prove a group is useless. For a test across splits, see [the shadow hits test](#shadow-test).
- Ranks still compare real groups only. Extra noise columns can still change how some models fit (for example, forests that sample a fraction of features per split), so ranks with shadows on can differ slightly from a run without them.
- **Cost.** Models train on twice the columns and every group is permuted twice, so expect about double the time and memory.
- `rank_resampled()` uses the same setting; split `i` uses `shadow_seed + i`, and each stability row gains the mean `above_shadow_share` over ranked splits, plus the [hits test](#shadow-test) keys.

> [!WARNING]
> Shadows only work if every model sees every column. A custom pipeline that drops columns, such as a positional `ColumnTransformer` with the default `remainder="drop"`, never sees the shadows, so every group trivially beats them. Shadow columns are appended after the real ones, so positional selectors of real columns still work.

<a id="shadow-test"></a>

### Shadow hits test

With shadows on, `rank_resampled()` also tests, for every group, whether it beats its shadows across splits more often than chance, at no extra model fits. This is Boruta's hits test applied to your splits. In each split, the group has a *hit* when its consensus `above_shadow_share` is strictly greater than 0.5: a majority of the family-weighted vote put it above the strongest shadow. Each stability row gains:

- `shadow_hits`: the number of splits that were hits. A split where the group has no consensus row (a failed or empty split) is a non-hit, and the number of trials is always `n_splits`.
- `shadow_p_value`: the one-sided binomial probability of at least `shadow_hits` hits in `n_splits` trials with success probability 0.5.
- `shadow_q_value`: the Benjamini-Hochberg adjusted p-value across all groups.

A 0.5 null probability and false-discovery-rate control are not established for arbitrary estimators or overlapping splits; these binomial reference values should not be interpreted as calibrated significance. The report says so: it has a `shadow_test` block (method, hit rule, `null_probability`, correction, `n_trials`, `min_p_value`, `calibrated: false`, and a `note`). Studying calibration is planned for 1.6.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import RepeatedKFold

from featranker import FeatureRanker

data = load_diabetes()
rng = np.random.default_rng(0)
X = np.hstack([data.data, rng.normal(size=(len(data.data), 2))])  # two pure-noise columns
names = list(data.feature_names) + ["noise_1", "noise_2"]

ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
report = ranker.rank_resampled(
    X, data.target, feature_names=names, scoring="r2", n_repeats=3, random_state=0,
    cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
)

test = report["shadow_test"]
print(f"{test['n_trials']} trials, smallest possible p-value {test['min_p_value']:.4f}")
for row in report["stability"]:
    print(
        f"{row['feature_group']:>7}  hits {row['shadow_hits']:>2}/{test['n_trials']}  "
        f"p {row['shadow_p_value']:.4f}  q {row['shadow_q_value']:.4f}"
    )
```

```text
10 trials, smallest possible p-value 0.0010
     s5  hits 10/10  p 0.0010  q 0.0059
    bmi  hits 10/10  p 0.0010  q 0.0059
     bp  hits  7/10  p 0.1719  q 0.6875
     s3  hits  4/10  p 0.8281  q 1.0000
     s6  hits  3/10  p 0.9453  q 1.0000
     s4  hits  0/10  p 1.0000  q 1.0000
    sex  hits  0/10  p 1.0000  q 1.0000
    age  hits  0/10  p 1.0000  q 1.0000
     s2  hits  0/10  p 1.0000  q 1.0000
     s1  hits  1/10  p 0.9990  q 1.0000
noise_2  hits  0/10  p 1.0000  q 1.0000
noise_1  hits  0/10  p 1.0000  q 1.0000
```

Output from scikit-learn models only (`group="fast"`), on 10 splits.

Read it with these limits in mind:

- **Uncalibrated null.** The test compares against 0.5, which is Boruta's choice, but nothing here shows that a group with no effect hits in half of the splits or fewer. An accepted custom estimator can break it: for example, a model whose importance for a pure-noise column is reliably above its shadow's gives small q-values on noise. Treat `shadow_p_value` and `shadow_q_value` as reference values for ordering groups, not as probabilities or error rates.
- **Dependent splits.** Cross-validation training sets overlap, so the splits are not independent trials and the binomial tail is not exact even for well-behaved models.
- **Low power with few splits.** The smallest p-value `n_splits` allows is `0.5 ** n_splits` (`shadow_test["min_p_value"]`): 0.031 for the default `cv=5`. At `cv=5` only a group that hits in all 5 splits reaches it (4 of 5 already gives p = 0.19), and Benjamini-Hochberg multiplies it by the number of groups over its rank, so it stays above 0.05 unless most groups hit 5 of 5. With 12 groups of which `s5` and `bmi` hit 5 of 5, both get q = 0.19. Use `RepeatedKFold` or another splitter with 10 or more splits when you want the test to say something; no warning is raised for fewer, except that a `ConsensusSelector` with `max_shadow_q` that selects nothing adds `n_splits` and the smallest possible p-value to its empty-selection warning.
- **What a small q means.** The group beats shuffled noise across splits. It is not proof of a causal effect or of usefulness for a new target.
- **Not for single-split reports.** `rank_features()` has one trial, so it gets no test.

`ConsensusSelector(max_shadow_q=0.05)` keeps the groups with `shadow_q_value <= 0.05` (see [Feature selection](#feature-selection)). Because the values are uncalibrated, `max_shadow_q` is a heuristic filter, not a significance level.

<a id="evidence"></a>

## ▍ Evidence

A rank says which groups matter, not why. With `evidence=True`, `fit()` also keeps the training data (as its own copy, so changing your arrays afterwards does not change the evidence) and `rank_features()` adds evidence about the mechanism behind the ranking:

- **Train score and overfit gap.** Each model gets `train_score` (the same scorer on the data it was fit on) and `overfit_gap = train_score - evaluation_score`. A large gap means the model scores much better on its training data than on the evaluation data. Memorization is one cause; a shift between the training and evaluation distributions, or a small evaluation sample, can produce the same gap, so check those before concluding that the model does not generalize.
- **Correlated groups.** `evidence.correlation` lists pairs of feature groups whose strongest member-column Spearman correlation on the **training** data is at least 0.5, with its sign. Permutation importance can split credit between correlated groups; consider putting them in one `feature_groups` entry.
- **SHAP summary** (needs `pip install 'featranker[shap]'`). For each linear model (`LinearExplainer`) and each tree model SHAP's tree explainer supports, including a `Pipeline` whose earlier steps are all `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, or `RobustScaler`, or identity steps (`"passthrough"`, `None`) (any other pipeline step lists the model as unsupported), `models[name]["shap"]` gives every group's `mean_abs` (mean absolute SHAP value, summed over the group's columns) and `direction` (Spearman correlation between a single-column group's value and its SHAP value; positive means higher values push the explained output up). Other models appear in `evidence.shap.unsupported`; there is no model-agnostic fallback. Without `shap`, a `UserWarning` is raised and the rest of the evidence is still computed.

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(
    task="reg",
    models={
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "forest": RandomForestRegressor(n_estimators=50, random_state=0),
        "tree": DecisionTreeRegressor(random_state=0),
    },
    evidence=True,
    verbose=False,
)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(X_eval, y_eval, scoring="r2", random_state=0)

for name, model in report["models"].items():
    print(f"{name:>6}  train {model['train_score']:.2f}  eval {model['evaluation_score']:.2f}"
          f"  overfit gap {model['overfit_gap']:.2f}")

for pair in report["evidence"]["correlation"]["pairs"][:3]:
    print(pair["groups"], f"rho {pair['rho']:.2f}")

for group, effect in report["models"]["ridge"]["shap"]["groups"].items():
    print(f"{group:>4}  mean|SHAP| {effect['mean_abs']:5.1f}  direction {effect['direction']:+.2f}")

print([f["code"] for f in report.summary()])
```

```text
 ridge  train 0.55  eval 0.39  overfit gap 0.16
forest  train 0.92  eval 0.29  overfit gap 0.63
  tree  train 1.00  eval -0.13  overfit gap 1.13
['s1', 's2'] rho 0.87
['s3', 's4'] rho -0.81
['s2', 's4'] rho 0.65
 age  mean|SHAP|   1.9  direction -1.00
 sex  mean|SHAP|   9.1  direction -1.00
 bmi  mean|SHAP|  21.4  direction +1.00
  bp  mean|SHAP|  10.7  direction +1.00
  s1  mean|SHAP|  14.3  direction -1.00
  s2  mean|SHAP|   5.4  direction +1.00
  s3  mean|SHAP|   4.5  direction -1.00
  s4  mean|SHAP|   2.5  direction +1.00
  s5  mean|SHAP|  24.5  direction +1.00
  s6  mean|SHAP|   0.8  direction +1.00
['few_families', 'tied_groups', 'tied_groups', 'within_noise', 'overfit', 'correlated_groups']
```

The unpruned tree fits its training data perfectly (`train 1.00`) and does worse than the dummy on the evaluation data, and the forest also lost most of its training advantage, so `summarize()` adds an `overfit` finding for both. `s1`/`s2` and `s3`/`s4` are correlated, so their permutation importances share credit (`correlated_groups`). For the ridge model, `s5` and `bmi` push predictions up and carry the largest `mean_abs`.

- `mean_abs` and `direction` are computed on at most 1000 evaluation rows, evenly spaced, so they do not depend on `random_state`. For linear models the background is at most 1000 evenly spaced **training** rows (the training distribution, like the tree explainer's own reference), so a linear `mean_abs` is the mean absolute distance from the training mean times the coefficient.
- `mean_abs` is on the explainer's output scale, which is not always the scale of `predict()`. For linear models it is the linear predictor, so for a model with a link function (for example `PoissonRegressor`, whose predictions are the exponential of it) it is in link units, such as log units. For tree models it is the raw output: the prediction for regression; log-odds, margin, or probability for classifiers, depending on the model. Compare groups within one model, not between models.
- For multiclass models `mean_abs` is averaged over classes and `direction` is `null`; `direction` is also `null` for multi-column groups and constant features. For binary classification the explained output is the positive class, the second label in sorted order (`label_encoder.classes_[1]`).
- For linear models `direction` is the sign of the coefficient (+1 or -1) when the coefficient is not zero, even when `mean_abs` is near 0, so read it together with `mean_abs`. A zero coefficient (for example a `Lasso` that dropped the column) gives `null`.

```python
import numpy as np
from sklearn.linear_model import Lasso, PoissonRegressor
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3))
y = rng.poisson(np.exp(4.0 + 1.0 * X[:, 0] + 0.5 * X[:, 1]))
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3, random_state=0)

ranker = FeatureRanker(
    task="reg",
    models={"poisson": PoissonRegressor(), "lasso": Lasso(alpha=1e3)},
    evidence=True,
    verbose=False,
)
ranker.fit(X_train, y_train, feature_names=["a", "b", "c"])
report = ranker.rank_features(X_eval, y_eval, scoring="r2", random_state=0)

prediction = ranker.trained_models["poisson"].predict(X_eval)
print(f"Poisson predictions span {prediction.min():.1f} to {prediction.max():.1f}")
for name in ("poisson", "lasso"):
    for group, effect in report["models"][name]["shap"]["groups"].items():
        direction = effect["direction"]
        shown = "null" if direction is None else f"{direction:+.2f}"
        print(f"{name:>7} {group}  mean|SHAP| {effect['mean_abs']:6.2f}  direction {shown}")
```

```text
Poisson predictions span 4.9 to 1429.9
poisson a  mean|SHAP|   0.69  direction +1.00
poisson b  mean|SHAP|   0.44  direction +1.00
poisson c  mean|SHAP|   0.00  direction -1.00
  lasso a  mean|SHAP|   0.00  direction null
  lasso b  mean|SHAP|   0.00  direction null
  lasso c  mean|SHAP|   0.00  direction null
```

The Poisson `mean_abs` values (at most 0.69) are in log units; the predictions span 4.9 to 1429.9. The `Lasso` with a large penalty has all coefficients at zero, so its `mean_abs` is 0 and its `direction` is `null`.
- Evidence costs one extra scoring pass per model, plus SHAP when installed (seconds per model for a large forest). It is off by default, and reports made without it are unchanged.
- A ranker fitted with `evidence=True` keeps its own copy of the training matrix, so pickling it stores that data too. With `shadows=True` the stored matrix includes the shadow block, about twice the size.
- `rank_resampled()` passes the setting to every split, so each `report["splits"][i]["report"]` carries its own evidence; nothing is aggregated across splits.
- On the command line, add `--evidence`.

<a id="feature-selection"></a>

## ▍ Feature selection in a pipeline

Choosing features from a ranking of all your data and then cross-validating a model on them overstates its performance. `ConsensusSelector` avoids that by ranking inside `fit()`: as a `Pipeline` step under `cross_val_score` or `GridSearchCV`, it reranks on each training fold and never sees the rows it is scored on.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

X, y = load_diabetes(return_X_y=True, as_frame=True)
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=4, group="tree", cv=3, n_repeats=5,
        random_state=0,
    )),
    ("model", Ridge()),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"nested-CV MAE: {-scores.mean():.1f} +/- {scores.std():.1f}")

pipe.fit(X, y)  # final model on all development data
print("selected:", pipe.named_steps["select"].selected_groups_)
```

```text
nested-CV MAE: 49.9 +/- 2.3
selected: ['s5', 'bmi', 'bp', 'sex']
```

Output shown with the `xgboost` and `catboost` extras installed. The score is honest because every outer fold chose its own four groups from its own training rows; the last line shows the groups chosen from all the data for the final model.

### `ConsensusSelector(task, scoring, *, k=None, min_shadow_share=None, max_shadow_q=None, ...)`

| Parameter | Default | Meaning |
|---|---|---|
| `task`, `scoring` | required | As in `FeatureRanker` and `rank_features()`. |
| `k` | `None` | Keep the first `k` groups in the [stability](report.md#resampled-reports) order (or all groups if there are fewer). Python or NumPy integer. |
| `min_shadow_share` | `None` | Keep groups whose stability `above_shadow_share` is at least this (0 to 1). Turns on [shadow features](#shadow-features). |
| `max_shadow_q` | `None` | Keep groups whose stability `shadow_q_value` (the [shadow hits test](#shadow-test)) is at most this (greater than 0, at most 1); a heuristic filter, since the values are [uncalibrated](#shadow-test). Turns on shadow features. Use `cv` with 10 or more splits, such as `RepeatedKFold`; with 5 the test has little power. |
| `group`, `models`, `n_jobs`, `shadow_seed` | as `FeatureRanker` | Passed to the ranker built in `fit()`. |
| `feature_groups`, `cv`, `n_repeats`, `random_state` | as `rank_resampled()` | Passed to `rank_resampled()` on the data given to `fit()`. |
| `feature_groups="auto"` | | Groups correlated columns with [`auto_groups()`](#auto-groups) on the data given to `fit()`, so inside a pipeline from the training fold only. The groups used are in `report_["feature_groups"]`. Any other string raises `ValueError` in `fit()`. |
| `verbose` | `False` | Progress bars; off by default because pipelines fit many times. |

Set at least one of `k`, `min_shadow_share`, and `max_shadow_q`; the shadow filters apply first, then the first `k` that pass. Invalid rules raise `ValueError` in `fit()`, before any ranking, following scikit-learn's convention that `__init__` only stores parameters (so `clone`, `set_params`, and `GridSearchCV` over `select__k` work).

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import RepeatedKFold

from featranker import ConsensusSelector

data = load_diabetes()
rng = np.random.default_rng(0)
values = np.hstack([data.data, rng.normal(size=(len(data.data), 2))])  # two pure-noise columns
names = list(data.feature_names) + ["noise_1", "noise_2"]
X = pd.DataFrame(values, columns=names)

selector = ConsensusSelector(
    "reg", "r2", max_shadow_q=0.05, group="fast", n_repeats=3, random_state=0,
    shadow_seed=0, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
)
selector.fit(X, data.target)
print(selector.selected_groups_)
```

```text
['s5', 'bmi']
```

A selected group keeps all of its columns, in their original order. Feature names come from DataFrame columns; for arrays they are `x0`, `x1`, … and `feature_groups` must use those names. After `fit()`: `selected_groups_`, `report_` (the resampled report from this fit), `n_features_in_`, `feature_names_in_` (DataFrames), and the standard `get_support()` and `get_feature_names_out()`.

- `report_` reflects only the data that particular fit saw. For the ranking of your whole development set, call `rank_resampled()` directly.
- If no group passes, `fit()` warns and the selector keeps no columns, as scikit-learn's own selectors do. Most downstream estimators (such as `Ridge`) then fail on zero columns, so that pipeline fit errors out. Loosen `min_shadow_share` or `max_shadow_q`, or use only the top-k rule by setting both to `None` and a positive `k`.
- For a group-aware inner `cv` such as `GroupKFold`, pass row groups to `fit(X, y, groups=...)`; in a `Pipeline`, pass `select__groups=...` (for example `cross_val_score(..., params={"select__groups": groups})`, which slices them to each training fold).
- DataFrame column labels are compared as strings, in `fit()` and `transform()`. A failed `fit()` clears any earlier fitted state.
- **Cost.** Each `fit()` is one `rank_resampled()` run: `cv` rankings, doubled with shadows. Inside `cross_val_score(cv=n)` that is `n × cv` rankings; fitting the final pipeline afterwards adds one more run. `GridSearchCV` multiplies the cost by the number of candidates, plus its final refit. Use `group="fast"` or `"tree"`, a small `cv`, and fewer `n_repeats` while searching.

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
| `failures` | warning for `fit` and `ranking`; info for `initialization` (usually a missing optional library) and `evidence` (a model's train score could not be computed) | One finding per failed stage, listing the models. |
| `few_families` | warning | Fewer than 3 model families voted. |
| `tied_groups` | info | Groups share the same `median_rank`; one finding per tie. `mean_rank` still orders them, and the finding does not measure ranking uncertainty. |
| `within_noise` | info | In most voting models, the group's mean importance is no greater than its standard deviation across shuffles (`mean <= std`). This flags large relative spread or non-positive importance; it does not prove shuffling left the score unchanged. With `n_repeats=1` the spread is 0, so it only catches importances of 0 or less. |
| `below_shadow` | info | With [shadows](#shadow-features) on, groups whose `above_shadow_share` is below 0.5. Descriptive, not a significance test. |
| `overfit` | info | With [evidence](#evidence) on: models that lost at least half of their training advantage over the dummy on the evaluation data, that is `(train_score - evaluation_score) / (train_score - baseline_score) >= 0.5`. Each is listed with its scores and that share. Skipped when `baseline_score` is `None`. |
| `correlated_groups` | info | With [evidence](#evidence) on: feature groups correlated on the training data (absolute Spearman rho of at least 0.5), whose permutation importance may be split between them. |

An empty list means nothing stood out, not that the ranking is proven. The checks cover one evaluation set.

<a id="llm-interpretation"></a>

## ▍ LLM interpretation

> [!WARNING]
> **Deprecated since 1.1; removal in 2.0.** `to_prompt()` and `explain()` emit a `FutureWarning` on every call and otherwise work as before through 1.x. The investigator study behind this feature found that the mechanism [evidence](#evidence), not an LLM, carried the interpretive gains, so FeatRanker is not extending LLM features.

FeatRanker can hand a report to any LLM you choose and check the numbers in the reply. It never calls a model, holds an API key, or imports an LLM library itself.

- `to_prompt(report, *, anonymize=False)` returns a prompt string: FeatRanker's interpretation rules (no causal claims, equal `median_rank` ties only the primary score, low importance is not proof of non-use, votes are not independent confirmations, shares are not p-values), the [`summarize()`](#summarize) findings, a compact JSON view of the report (without per-repeat values or fitted params), and a required JSON reply format in which every claim cites the data it uses. Paste it into any chat, or send it yourself. It works for single-split and [resampled](#resampled-ranking) reports.
- `explain(report, llm, *, anonymize=False)` builds the same prompt, calls `llm(prompt)` once, where `llm` is any function you write that takes the prompt string and returns the reply string, and checks every cited value.

```python
import json

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker, explain

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
ranker = FeatureRanker(task="reg", group="tree", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)


def stand_in_llm(prompt: str) -> str:
    # A real LLM reads the prompt; this fixed reply just shows the checking.
    return json.dumps({
        "summary": "bmi and s5 share first place; bp follows.",
        "claims": [
            {"text": "bmi and s5 are tied at median rank 1.5.", "evidence": [
                {"path": "consensus[0].median_rank", "value": 1.5},
                {"path": "consensus[1].median_rank", "value": 1.5},
            ]},
            {"text": "bp is third with median rank 3.25.", "evidence": [
                {"path": "consensus[2].feature_group", "value": "bp"},
                {"path": "consensus[2].median_rank", "value": 3.25},
            ]},
            {"text": "Nine models vote.", "evidence": [
                {"path": "consensus[0].n_models", "value": 9},
            ]},
        ],
    })


result = explain(report, stand_in_llm)
for claim in result["claims"]:
    print("verified    " if claim["verified"] else "NOT verified", claim["text"])
print("all_verified:", result["all_verified"])
```

```text
verified     bmi and s5 are tied at median rank 1.5.
verified     bp is third with median rank 3.25.
NOT verified Nine models vote.
all_verified: False
```

The stand-in returns a fixed reply so the example runs without a model. The third claim cites `n_models` as 9; the report says 7, so it is flagged.

`explain()` returns `{"summary": ..., "claims": [{"text": ..., "evidence": [{"path", "value", "verified"}], "verified": ...}], "all_verified": ...}`. A cited value is verified when its path into the prompt's DATA section exists and the value matches. Paths use dot keys and `[i]` indices (`consensus[0].median_rank`), or a JSON Pointer for keys containing `.`, `[`, or `/` (`/models/model.v1/importance/lab[0]/mean`, with `~1` for `/`). Numbers may differ by up to 0.01 or 1% (so rounding to two decimals passes); other values must match exactly and in type, so `true` never matches `1`. Values must be a number, string, boolean, or null. Verification uses a copy taken when the prompt was built, so changing the report during the `llm` call cannot affect it. A claim is verified only if it cites at least one value and every cited value checks out. A reply that is not the requested JSON raises `ValueError` with its first 200 characters.

> [!WARNING]
> Verified numbers do not make a reading correct. A claim can cite true values and still overstate what they mean; the prompt's rules guide the wording, but only the numbers are checked. Read the claims against [Reading the report](report.md#reading-the-report).

**With Claude.** A minimal `llm` function using Anthropic's official Python SDK (`pip install anthropic`, your own API key):

```python
import anthropic

from featranker import explain

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or an `ant auth login` profile


def claude(prompt: str) -> str:
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # a declined request is retried on Anthropic's recommended fallback model
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Request declined: {response.stop_details}")
    return next(block.text for block in response.content if block.type == "text")


result = explain(report, claude, anonymize=True)
```

This recipe is not run by FeatRanker's tests, because it needs your key and network access. It opts into server-side fallback (`fallbacks="default"`), so a declined request is retried on Anthropic's recommended fallback model, and it raises if the request is still declined. For a local model, write the same kind of function around your model's client; FeatRanker only needs a string back.

**Privacy.** Only the compact report view and the `summarize()` findings are sent: scores, ranks, importances, family and shadow fields, and failure model, stage, and error type. Failure messages are never sent, because exceptions can quote feature names or data values, and a custom exception's class name is sent as `CustomError` (built-in names such as `ValueError` stay); no data rows are sent. Identifiers are included unless you pass `anonymize=True`, which replaces feature groups with `g1`, `g2`, …, columns with `c1`, `c2`, …, your own model and family names with `m1`, `f1`, …, and a custom scorer's name with `custom_scorer`. Built-in model, family, and scorer names stay readable. The prompt then asks the model to wrap those aliases as `[[g1]]`-style markers and keep built-in names as they are, and only the markers are mapped back in the returned text; evidence paths and values are returned as cited. Anonymization covers identifiers, not fixed vocabulary: a feature named `fit` still shares its spelling with the failure stage `"fit"`.

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

<a id="auto-groups"></a>

### Grouping correlated features: `auto_groups()`

Permutation importance splits credit between near-copies: shuffling one of two correlated columns barely hurts a model that can use the other, so both look unimportant. `auto_groups(X, *, threshold=0.7, feature_names=None)` proposes the groups from `X` alone (it never sees a target) and returns a dict you can pass as `feature_groups`.

```python
from sklearn.datasets import load_diabetes

from featranker import auto_groups

X, y = load_diabetes(return_X_y=True, as_frame=True)

print(auto_groups(X))
print(auto_groups(X, threshold=0.4))
print(auto_groups(X, threshold=0.9))
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{'bmi+s5': ['bmi', 's5'], 's1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{}
```

- Columns are clustered by Spearman correlation with complete linkage, so every pair of columns inside a group has absolute correlation of at least `threshold` (`0 < threshold <= 1`), up to a floating-point tolerance of `1e-12`, which lets exact copies group at `threshold=1`. Lowering it makes bigger groups. A constant column is never grouped, whatever the threshold.
- Use enough rows: with few rows the sample correlation is inflated (with two rows every non-constant pair has absolute correlation 1), so near-unrelated columns can be grouped.
- Only groups of two or more columns are returned. A group is named by its members joined with `+`, in column order; a name that equals an existing column or another generated group's name raises `ValueError`.
- `X` is validated like `FeatureRanker.fit()` input. Column names come from a DataFrame, else `feature_names`, else `x0`, `x1`, ….
- Call it on your training or development data and pass the result to `rank_features()` or `rank_resampled()`; there is no `"auto"` option there, so the grouping stays visible in your code. The [correlation evidence](#evidence) lists group pairs from 0.5, which shows what sits near the cut.
- The correlation matrix and clustering are O(p²) in the number of columns, which is fine for hundreds of columns.

Where you cannot compute the groups yourself, use `"auto"`: `ConsensusSelector(feature_groups="auto")` in a pipeline, or `--feature-groups auto` on the command line. Both use the default threshold.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

X, y = load_diabetes(return_X_y=True, as_frame=True)
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=3, group="fast", cv=3, n_repeats=3,
        feature_groups="auto", random_state=0,
    )),
    ("model", Ridge()),
])
pipe.fit(X, y)

select = pipe.named_steps["select"]
print({name: cols for name, cols in select.report_["feature_groups"].items() if len(cols) > 1})
print("selected:", select.selected_groups_)
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
selected: ['s5', 'bmi', 's3+s4']
```

Under `cross_val_score` each outer training fold computes its own groups, so evaluation rows never shape them.

<a id="labels"></a>

## ▍ Data handling and labels

`FeatureRanker` does not impute, encode features, or select features; `ConsensusSelector` selects feature groups. The ranker changes data values in two places only:

- **Linear scaling.** Built-in linear models sit behind a `StandardScaler` fit on training data only. Tree models and your own models get the data as given.
- **Classification labels.** For `task="clf"`, `fit()` encodes the labels to `0..n_classes-1` with a `LabelEncoder`, because some estimators (XGBoost) require it. Every estimator and scorer, including your own, sees the encoded labels. `ranker.label_encoder.classes_` maps each code back to the original label; code `i` is `classes_[i]`.

`fit()`, `rank_features()` and `rank_resampled()` work on their own float copies of `X` and `y`, so changing your arrays afterwards changes nothing in the models, the evidence or later reports, even for estimators that store their training data (such as k-nearest neighbours). The cost is one extra copy of the data per call.

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
