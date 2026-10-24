# FEATRANKER ▍ USAGE

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <b>Usage</b> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <a href="models.md">Models</a> &nbsp;│&nbsp; <a href="api.md">API</a></sub>

## ▍ API

The API is one class: `FeatureRanker(...)`, then `.fit(...)`, then `.rank_features(...)`, which returns a [`Report`](#working-with-reports). [`.rank_resampled(...)`](#resampled-ranking) repeats that over cross-validation splits, [`.ablate(...)`](#ablation-curve) refits the voting models on the top-k groups to check the ranking, [`ConsensusSelector`](#feature-selection) turns the ranking into a scikit-learn pipeline step, [`summarize(report)`](#summarize) reads the result, and [`to_prompt` / `explain`](#llm-interpretation) hand it to an LLM with checked citations (deprecated since 1.1). `evidence=True` adds [mechanism evidence](#evidence) to the report.

### `FeatureRanker(task, group="all", models=None, n_jobs=None, verbose=True, shadows=False, shadow_seed=None, evidence=False, shadow_mode="joint", missing="error", categorical=None, model_input="array", shuffle="rows", families=None)`

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
| `shadow_mode` | `"joint"` | `"joint"` shuffles all shadow columns with one shared row order; `"independent"` shuffles each shadow column on its own. Only used with `shadows=True`. See [Shadow mode](#shadow-mode). |
| `missing` | `"error"` | `"error"` rejects NaN in features. `"impute"` accepts NaN and gives every built-in model a median imputer fit on its training rows only; custom models receive NaN unchanged. See [Missing values](#missing-values). |
| `categorical` | `None` | `None` accepts numeric features only. `"auto"` treats DataFrame columns of object, string, or category dtype as categorical; a list of feature names marks those columns. Levels are learned in each `fit()` from the training rows only and each column stays one feature. See [Categorical features](#categorical-features). |
| `model_input` | `"array"` | `"array"` gives models a float NumPy array. `"dataframe"` gives every model a pandas DataFrame with the feature names (and `shadow__<name>` columns with shadows), so custom pipelines can select columns by name. Needs pandas. See [Columns by name](#columns-by-name). |
| `shuffle` | `"rows"` | How permutations and shadows shuffle rows: `"rows"` (free), `"within_groups"` (only among rows with the same label in `shuffle_groups`), or `"circular"` (cyclic shift, for rows in time order). Anything else raises `ValueError`. See [Shuffling dependent rows](#shuffling-dependent-rows). |
| `families` | `None` | Dict of `{model_name: family}` (non-empty strings) that sets the consensus family of custom or built-in models; models mapped to one family cast one vote. Unmapped custom models stay separate families. A name that is not a configured model raises `ValueError` in `fit()` (and in `rank_resampled()` and `ConsensusSelector.fit()`, before any split); the name of a built-in model that failed to initialise is accepted, a built-in name outside the chosen `group` or `task` raises. See [Your own models](#your-own-models). |

### `.fit(X, y, feature_names=None, sample_weight=None, shuffle_groups=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X` | required | Training features: 2-D NumPy array or pandas DataFrame of numbers (plus the columns named by `categorical`), with at least one row and one column. Infinite values are rejected, and so is NaN unless `missing="impute"`. |
| `y` | required | Training target, one value per row. Class labels may be numbers or strings; regression targets must be numeric. |
| `feature_names` | `None` | Unique, non-empty column names. Required for NumPy input; taken from the columns for a DataFrame. |
| `sample_weight` | `None` | 1-D array-like of finite, non-negative training weights, one per row, with a positive sum. Every model is fit with them; see [Sample weights](#sample-weights). |
| `shuffle_groups` | `None` | One group label per training row; required with `shuffle="within_groups"` (shadows are shuffled within groups), an error with another `shuffle`. See [Shuffling dependent rows](#shuffling-dependent-rows). |

Returns the ranker itself. Raises if no model fits.

### `.rank_features(X_eval, y_eval, *, scoring, feature_groups=None, n_repeats=10, random_state=None, sample_weight=None, max_samples=None, shuffle_groups=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X_eval` | required | Evaluation features with the same columns, in the same order, as `fit()`. Keeping them separate from the training data is up to you. |
| `y_eval` | required | Evaluation target. Class labels must have appeared in training. |
| `scoring` | required, keyword-only | scikit-learn scorer name or a callable `(estimator, X, y) -> float` where higher is better, or a non-empty list or tuple of them (the first is primary). See [Choosing a scorer](#choosing-a-scorer) and [Several metrics](#several-metrics). |
| `feature_groups` | `None` | Dict of `{group_name: [feature, ...]}` to shuffle together. Ungrouped features are ranked on their own. See [Grouped permutation](#grouped-permutation). |
| `n_repeats` | `10` | Shuffles per feature group; a positive integer. More repeats give a more precise average importance and take longer. |
| `random_state` | `None` | Integer seed for permutation-importance shuffles. Reproducible reports also require deterministic or seeded estimators and deterministic scoring; with shadows enabled, fix `shadow_seed` too. |
| `sample_weight` | `None` | 1-D array-like of finite, non-negative evaluation weights, one per evaluation row, with a positive sum. The baseline, permuted, and dummy scores are weighted with them. See [Sample weights](#sample-weights). |
| `max_samples` | `None` | Score on a random subsample of the evaluation rows: an int (rows; more than there are uses all) or a float in (0, 1] (share of rows, rounded down, at least 1). Anything else raises `ValueError` before scoring, as does a subsample whose `sample_weight` values are all zero. See [Speed](#speed). |
| `shuffle_groups` | `None` | One group label per evaluation row; required with `shuffle="within_groups"` (permutations and shadows are shuffled within groups), an error with another `shuffle`. See [Shuffling dependent rows](#shuffling-dependent-rows). |

Returns the report dict described in [Report](report.md). Raises if every model fails ranking.

### `.ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None, eval_sample_weight=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X`, `y` | required | The training data this ranker was fit on. They are checked against the report's `run["train_data_sha256"]` (skipped when that is `null`, as for a ranker pickled before 1.12); other data raises `ValueError`. |
| `X_eval`, `y_eval` | required | Data to score the refit models on, with the same columns as `fit()`. Usually the evaluation data of `report`; see the caveat in [Ablation curve](#ablation-curve). Every row is scored: the report's `max_samples` is ignored. |
| `report` | required | A held-out report from this ranker's `rank_features()`. A resampled report, or one with an empty consensus, raises `ValueError`. |
| `ks` | `None` | How many top consensus groups to keep: a non-empty list of ints from 1 to the number of consensus groups, de-duplicated and sorted; anything else raises `ValueError`. The default is 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, ... up to that number, plus that number. |
| `scoring` | `None` | A scorer name or callable. `None` uses the report's `scoring` name, which a report ranked with a callable cannot resolve; pass `scoring=` then. |
| `sample_weight`, `eval_sample_weight` | `None` | Training and evaluation weights for the refits and the scores, as in `fit()` and `rank_features()`. The ranker does not remember them, so pass the ones used for the ranking again: the training weights are checked against the report's `run["train_weights_sha256"]` (none given when the key is absent; skipped with the training-data check when `train_data_sha256` is `null`), and the evaluation weights against `run["weights_sha256"]` when `X_eval` and `y_eval` are the report's own evaluation data (a third data set takes any weights). A mismatch raises `ValueError`. A report without a `run` block is not checked. |

Returns a new `Report`: a deep copy of `report` plus an [`ablation` block](report.md#ablation). `report` is not modified. An unfitted ranker also raises `ValueError`.

<a id="working-with-reports"></a>

## ▍ Working with reports

`rank_features()` and `rank_resampled()` return a `Report`. It is a plain `dict` (indexing, `json.dumps`, pickling, and comparison with dicts all work as before) with three helpers:

- `report.summary()`: the [`summarize()`](#summarize) findings (single-split and resampled reports).
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

### Fewer evaluation rows: `max_samples`

Permutation importance predicts over every evaluation row once per model, feature group, and repeat, so on a large evaluation set most of that time buys little precision. `max_samples` scores on a random subsample instead: an int is a number of rows, a float in (0, 1] a share of them (`int(max_samples * n)`, rounded down as in scikit-learn: 0.29 of 200 rows is 57). One subsample, drawn without replacement with `random_state`, is used for the dummy gate, every baseline, and every permuted score, so they stay comparable; the training scores and the [evidence](#evidence) are unchanged (the SHAP values and shadows still use every row; `overfit_gap` is the train score minus the model's score on the subsample). With `sample_weight`, the weights follow their rows; if the drawn rows all have zero weight, `rank_features()` raises `ValueError` naming `max_samples` and `sample_weight` before scoring (in `rank_resampled()` that split fails). The report gains `max_samples` and `n_eval_rows`, and `run["data_sha256"]` still fingerprints all the evaluation rows you passed. The sample is plain random, not stratified: a small one can miss a rare class, and a scorer such as `roc_auc` can then fail (if every model fails, `rank_features()` raises a `RuntimeError` listing the failures), so keep `max_samples` large enough to hold every class; there is no stratified option. It works the same in `rank_resampled()` (each split's evaluation rows), `ConsensusSelector(max_samples=...)`, and the CLI (`--max-samples`). To cut cost further, combine it with `group="fast"` and [`n_jobs`](#parallel-models).

```python
ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
for max_samples in (None, 0.5, 40):
    report = ranker.rank_features(
        X_eval, y_eval, scoring="r2", random_state=42, max_samples=max_samples
    )
    top = [row["feature_group"] for row in report["consensus"][:3]]
    print(f"max_samples={max_samples}: {report.get('n_eval_rows', len(y_eval))} rows, top 3 {top}")
```

```text
max_samples=None: 133 rows, top 3 ['s5', 'bmi', 'bp']
max_samples=0.5: 66 rows, top 3 ['s5', 'bmi', 's6']
max_samples=40: 40 rows, top 3 ['bmi', 's5', 'sex']
```

Same data and split as the example above. A half-size sample already swaps `bp` out of the top three, so judge the ranking's stability before you trade rows for speed.

### Several metrics

`scoring` also takes a list or tuple, such as `["accuracy", "neg_log_loss"]`, to check a ranking under more than one metric without refitting the models. Names must be unique (a callable is named by its `__name__`, a `make_scorer` object by its score function's, so wrap one of two `make_scorer(f1_score, ...)` scorers in a function with a distinct `__name__`; an empty list, a repeated name, or an invalid entry raises before any model is ranked). The first scorer is primary: the dummy gate, consensus, shadow evidence, stability, `summarize()`, and `ConsensusSelector` use it alone, so the ranking is exactly what a single-scorer run gives (a one-element list changes nothing). Every other scorer is secondary: on the same permuted copies of the data, each model also gets its score, and its importance per feature group (`mean`, `std`), under that scorer. There is no consensus or stability table per secondary scorer; read `report["models"][model]["secondary"][scorer]` (see [Report](report.md#several-scorers)). Each secondary scorer predicts again on every permuted copy, so time grows with the number of scorers; `max_samples` offsets it, and applies the same subsample, and `sample_weight` the same weights, to every scorer. If a secondary scorer raises or returns a non-finite value for a model, that model stays ranked, its entry is `null`, and `failures` gets an `"evidence"` entry. It works the same in `rank_resampled()`, `ConsensusSelector(scoring=[...])`, and the CLI (`--scoring accuracy,neg_log_loss`).

```python
report = ranker.rank_features(
    X_eval, y_eval, scoring=["r2", "neg_mean_absolute_error"], random_state=42
)

print(report["scoring"], report["secondary_scoring"])
for name, model in report["models"].items():
    mae = model["secondary"]["neg_mean_absolute_error"]
    top = max(mae["importance"], key=lambda group: mae["importance"][group]["mean"])
    print(f"{name}: r2 {model['evaluation_score']:.2f}, MAE {-mae['evaluation_score']:.1f}, top under MAE {top}")
print([row["feature_group"] for row in report["consensus"][:3]])
```

```text
r2 ['neg_mean_absolute_error']
ridge_regression: r2 0.39, MAE 44.7, top under MAE s5
svr_regression: r2 0.14, MAE 53.4, top under MAE bmi
decision_tree_regressor: r2 -0.30, MAE 64.0, top under MAE s5
random_forest_regressor: r2 0.30, MAE 47.9, top under MAE bmi
hist_gradient_boosting_regressor: r2 0.28, MAE 46.6, top under MAE s5
['s5', 'bmi', 'bp']
```

Same data and split as the example above, with `group="fast"`. The consensus is the `r2` ranking; the MAE columns show whether each model would rank the top group the same way under the second metric.

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

**Families.** Each custom model is its own family by default, and each family casts one vote in the consensus. Several near-identical custom models (three ridge variants, say) therefore count as three votes and can outvote a different kind of model, so the claim that near-duplicates cannot outvote the rest holds only for the built-in families. Say which of your models belong together with `families={model_name: family}`; every model mapped to the same string votes once, as the median of its members, and the report's `family` field shows the mapping. A name that is not one of the configured models raises `ValueError` in `fit()` (and in `rank_resampled()` and `ConsensusSelector.fit()`, before any split), and an empty or non-string name or family, or a value that is not a mapping, raises `ValueError` in the constructor; any mapping is accepted and copied to a dict. `families` also accepts built-in model names, to regroup them (the name of a built-in model that failed to initialise is still accepted, while a built-in name outside the chosen `group` or `task` raises), and `ConsensusSelector(families=...)` passes it on.

```python
ranker = FeatureRanker(
    task="reg",
    models={
        "ridge_small": Ridge(alpha=0.1),
        "ridge_large": Ridge(alpha=10.0),
        "forest": RandomForestRegressor(n_estimators=200, random_state=0),
    },
    families={"ridge_small": "ridge", "ridge_large": "ridge"},  # forest stays its own
)
```

> [!IMPORTANT]
> By default your estimators receive a float NumPy array, even when you pass a DataFrame. Inside a pipeline, select columns by position (for example `ColumnTransformer([("scale", StandardScaler(), [0, 1])])`), or pass `model_input="dataframe"` to select by name (see [Columns by name](#columns-by-name)). Categorical values must be numbers by then: either encode them yourself, or pass `categorical=` and your estimators receive the integer codes (see [Categorical features](#categorical-features)). With `missing="impute"` your estimators receive NaN unchanged; see [Missing values](#missing-values).

<a id="columns-by-name"></a>

## ▍ Columns by name

`FeatureRanker(..., model_input="dataframe")` fits, scores, and permutes every model (built-in and custom) on a pandas DataFrame instead of a NumPy array. Its columns are the feature names in order, then, with `shadows=True`, `shadow__<name>` for each shadow column. This is the normal scikit-learn way to preprocess mixed data: `ColumnTransformer([("num", StandardScaler(), ["age", "income"])])`. Importances, scores, train scores, and the SHAP summary equal those of the default `"array"` mode for models that do not depend on column names; the DataFrame is rebuilt from the permuted array for each scoring call, which costs a little time on wide data. The report records `model_input: "dataframe"` (the key is absent in the default mode). `rank_resampled()` and `ConsensusSelector(model_input=...)` use it in every split; the CLI keeps array input. Without pandas, `model_input="dataframe"` raises `ImportError` at construction.

**The shadow contract.** With shadows on, a custom pipeline must keep the shadow columns, because a model that never sees them has shadow importances of zero, so every real feature "beats" the shadows. After fitting, FeatRanker works out which input columns each model uses, in both input modes: it follows a `Pipeline` into its first step, a `ColumnTransformer` into each branch (a `ColumnTransformer`, `Pipeline` or `FeatureUnion` inside a branch too), a `FeatureUnion` into each part, and a fitted feature selector (`SelectKBest`, `SelectFromModel`, ...) to the columns it keeps, mapping every branch's columns back to the model input. A later `Pipeline` step that is or contains a `ColumnTransformer` is checked only when every earlier step is known to keep the columns one for one (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, a `SimpleImputer` that adds no indicator and drops no column); after any other step it cannot be mapped and counts as unverified. If the transformers and `remainder` together leave a shadow column unused, or if the columns cannot be worked out (scikit-learn's private attributes are not as expected, a meta-estimator such as `TransformedTargetRegressor` wraps a `ColumnTransformer`, or a later `ColumnTransformer` follows a step that may change the columns), the model is left out of every shadow comparison in that report (no `above_shadow` flags, no `shadow_max`, not counted in `family_above_shadow` or `above_shadow_share`) and `failures` gets an entry with stage `"shadows"` and `error_type` `"ShadowColumnsDropped"`. Its importances and its ranking vote are unaffected. The failure message says which of the two it was ("does not use every shadow column" or "could not be verified"). Use `remainder="passthrough"`, or list the `shadow__` columns, to keep it in the shadow evidence. The check cannot see other ways of dropping columns: a `FunctionTransformer`, a custom transformer, or any other estimator that ignores some inputs is assumed to use all of its columns, and is your responsibility.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({"age": rng.uniform(20, 70, n), "income": rng.normal(50, 10, n), "noise": rng.normal(size=n)})
y = 0.3 * df["age"] + 0.1 * df["income"] + rng.normal(0, 1, n)
train, evaluation = df.iloc[:200], df.iloc[200:]

models = {
    # Lists only the real columns and drops the rest, shadows included.
    "drops_shadows": make_pipeline(
        ColumnTransformer([("num", StandardScaler(), ["age", "income", "noise"])]), Ridge()
    ),
    "keeps_shadows": make_pipeline(
        ColumnTransformer(
            [("num", StandardScaler(), ["age", "income"])], remainder="passthrough"
        ),
        Ridge(),
    ),
}
ranker = FeatureRanker(
    task="reg", models=models, shadows=True, shadow_seed=0,
    model_input="dataframe", verbose=False,
)
ranker.fit(train, y[:200])
report = ranker.rank_features(evaluation, y[200:], scoring="r2", random_state=0)

print(report["model_input"])
print([(f["model"], f["stage"], f["error_type"]) for f in report["failures"]])
print([m for m, r in report["models"].items() if "shadow_max" in r])
for row in report["consensus"]:
    print(f"{row['feature_group']:>6}  median rank {row['median_rank']:.1f}  "
          f"n_models {row['n_models']}  above_shadow_share {row['above_shadow_share']:.1f}")
```

```text
dataframe
[('drops_shadows', 'shadows', 'ShadowColumnsDropped')]
['keeps_shadows']
   age  median rank 1.0  n_models 2  above_shadow_share 1.0
income  median rank 2.0  n_models 2  above_shadow_share 1.0
 noise  median rank 3.0  n_models 2  above_shadow_share 0.0
```

Both models vote in the consensus, but only `keeps_shadows` supplies shadow evidence; `summarize()` reports the `"shadows"` stage as a warning.

Limitations of named input:

- XGBoost rejects feature names containing `[`, `]` or `<` in dataframe mode; those fit failures appear in `failures`. Rename the columns or use array mode.
- The shadow check follows `Pipeline`s, `ColumnTransformer` branches, `FeatureUnion` parts and fitted feature selectors. A `ColumnTransformer` wrapped in a meta-estimator (`TransformedTargetRegressor`, `GridSearchCV`, ...) is reported as unverified; a `FunctionTransformer` or custom transformer that drops columns is not seen at all (it is assumed to use every column).
- Columns are mapped through `ColumnTransformer` branches and the first step of a `Pipeline`. A later `Pipeline` step sees derived features, so a later `ColumnTransformer` is checked only after steps known to keep the columns one for one (the scalers and a plain `SimpleImputer`); after any other step it is unverified.

<a id="parallel-models"></a>

## ▍ Parallel models

`FeatureRanker(..., n_jobs=2)` runs the per-model work in `fit()` and `rank_features()` in parallel with joblib. `None` runs sequentially and `-1` uses all cores.

- Reports match `n_jobs=None`, with models in the same order, when every estimator is deterministic or seeded, the scorer is deterministic, and `random_state` is a fixed integer. Every built-in model with randomness is seeded (`random_state=42`) since 0.4.1. An unseeded custom estimator, such as `RandomForestRegressor()` without `random_state`, can give different reports on every run, parallel or not. With `shadows=True`, reproducibility also requires a fixed non-negative integer `shadow_seed`; fixing `random_state` alone does not seed shadow construction.
- Per-model failures are recorded exactly as in the sequential path.
- With any parallel setting (`n_jobs` other than `None` or `1`, including `-1`), custom scorers and estimators are sent to worker processes, so everything they hold must be serializable. Lambdas and local functions are fine.

<a id="missing-values"></a>

## ▍ Missing values

By default NaN in the features raises `ValueError`. With `missing="impute"` FeatRanker accepts NaN (infinite values are still rejected) and imputes inside each model, so no evaluation row shapes an imputation:

- Every built-in model becomes a pipeline that starts with `SimpleImputer(strategy="median")`, followed by the usual scaler for linear models. The imputer is fit on the rows the model is fit on and nothing else, so under `rank_resampled()` and `ConsensusSelector` each training split learns its own medians. This applies to every built-in model, including those that handle NaN natively (such as `HistGradientBoosting`), so one rule holds across models and scikit-learn versions; native handling is not used. A column that is missing in every training row becomes a constant 0 in training and keeps its position; its importance is meaningless, and `train_counts` in the report shows it.
- Your own `models=` receive the NaN values unchanged and must handle them, for example with `make_pipeline(SimpleImputer(...), ...)` or an estimator that supports NaN.
- The dummy baseline ignores feature values and is unaffected.
- Evidence works with NaN: correlated groups and `auto_groups()` use, for each pair that involves a column with NaN, the rows where both are present (a pair with fewer than 30 such rows counts as uncorrelated, so sparse columns that rarely overlap are not grouped by chance; pairs of complete columns are computed as without NaN), and SHAP treats the leading imputer as a per-column step. Pairwise correlation costs O(columns squared times rows) when NaN is present; complete data takes the faster path with identical numbers to earlier versions. `auto_groups()` needs no argument for this.
- The report gains `missing` (see [Report](report.md#structure)) with the number of missing training values per feature, so a mostly-missing feature is visible.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X = data.data.copy()
rng = np.random.default_rng(0)
X[rng.random(len(X)) < 0.2, 2] = np.nan  # 20% of bmi missing
X[rng.random(len(X)) < 0.1, 3] = np.nan  # 10% of bp missing
X_train, X_eval, y_train, y_eval = train_test_split(
    X, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", missing="impute", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)

print(report["missing"]["train_counts"])
for row in report["consensus"][:3]:
    print(f"{row['feature_group']:>5}  median rank {row['median_rank']:.1f}")

# The imputer holds the training median of bmi, not the evaluation one.
imputer = ranker.trained_models["ridge_regression"][0]
print(round(imputer.statistics_[2], 4), round(np.nanmedian(X_train[:, 2]), 4),
      round(np.nanmedian(X_eval[:, 2]), 4))
```

```text
{'age': 0, 'sex': 0, 'bmi': 67, 'bp': 27, 's1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0}
   s5  median rank 1.0
  bmi  median rank 2.0
   bp  median rank 3.5
-0.0105 -0.0105 0.0013
```

The ranking is from the five `group="fast"` built-in models. The last line shows the fitted imputer's median for `bmi` equal to the training median (-0.0105), not the evaluation one (0.0013).

`ConsensusSelector` and the command line take the same policy: `ConsensusSelector("reg", "r2", k=2, missing="impute")` accepts NaN in `fit()` and `transform()` (it only chooses columns, so put an imputer after it in a `Pipeline` for the final model), and `featranker ... --missing impute` reads empty CSV cells as missing.

**Limitations.** Median imputation mixes missingness with values. A lab test ordered only for sick patients is missing for a reason, and imputation leaves a spike at the median, and tree and kernel models can learn it, so a feature's importance can mix its values with its missingness (in a probe where only missingness drove the target, tree importance was about 1.9 and SVR about 1.05, while ridge, which can hardly use a spike, gave 0.006). To separate values from missingness, use custom models with `SimpleImputer(add_indicator=True)` or native NaN handling, and read `train_counts` to see which features are mostly missing. Missing values in the target are not supported. For string columns see [Categorical features](#categorical-features).

<a id="categorical-features"></a>

## ▍ Categorical features

By default every feature must be numeric. With `categorical`, FeatRanker accepts text and category columns as they are, learns how to encode them inside each `fit()`, and still ranks each raw column as **one** feature:

- `categorical="auto"` treats DataFrame columns of object, string, or category dtype as categorical (bool and numeric dtypes stay numeric). NumPy arrays have no per-column dtypes, so `"auto"` finds nothing there. `categorical=["city", ...]` marks those columns, whatever their dtype, and works for NumPy object arrays too (give `feature_names`). A name that is not a feature raises `ValueError` in `fit()`.
- **Levels come from the training rows only.** Each categorical column's levels are the sorted string forms of its training values, coded `0..k-1`; a missing value (`None` or NaN) is its own level, last. Values that were not in training, including a missing value when training had none, share one extra "unseen" code `k` at evaluation time. Under `rank_resampled()` and `ConsensusSelector` every split learns its own levels. The fitted ranker exposes `categorical_features_` (the names) and `categorical_levels_` (name to levels, `None` for the missing level).
- **One feature per raw column.** The ranker works on a matrix where each categorical column holds its codes, so shuffling the codes is shuffling the raw values: `city` is one feature group in `feature_groups`, in the consensus, and in the shadows, however many levels it has.
- **Built-in linear models** one-hot encode the categorical columns (a code never seen in training is ignored, giving an all-zero row) and scale the numeric ones; under `missing="impute"` the numeric columns are imputed first, and a missing categorical value is simply a level. **Built-in tree models** use the codes as numbers, which gives the levels an arbitrary order; for high-cardinality columns or a different encoding, pass your own `models=`. **Your own models** receive the code matrix (floats) and can read `ranker.categorical_features_` to know which columns hold codes.
- The report gains `categorical` (see [Report](report.md#structure)): the columns, their number of training levels, and how many evaluation values were unseen.
- Evidence: categorical columns are left out of the Spearman correlations (`evidence.correlation`, and `auto_groups(X, categorical=...)`, which never groups them). The SHAP summary explains tree models on the codes (the `direction` of a categorical group is `null`, since codes have no order); linear models with a one-hot step are listed in `evidence.shap.unsupported`.

```python
import numpy as np
import pandas as pd

from featranker import FeatureRanker

rng = np.random.default_rng(0)
n = 400
city = rng.choice(["Oslo", "Rome", "Lima", "Kyiv", "Cairo"], n)
premium = {"Oslo": 30.0, "Rome": -20.0, "Lima": 25.0, "Kyiv": -25.0, "Cairo": 0.0}
size = rng.uniform(30, 120, n)
age = rng.uniform(0, 60, n)
price = size * 2 + np.array([premium[c] for c in city]) + rng.normal(0, 8, n)
df = pd.DataFrame({"city": city, "size": size, "age": age})
train, evaluation = df.iloc[:280], df.iloc[280:]
# Cairo never appears in training; its rows are mapped to Rome here.
train = train.assign(city=train["city"].replace("Cairo", "Rome"))

ranker = FeatureRanker(task="reg", group="fast", categorical="auto", verbose=False)
ranker.fit(train, price[:280])
report = ranker.rank_features(
    evaluation, price[280:], scoring="neg_mean_absolute_error", random_state=42
)

print(ranker.categorical_features_, ranker.categorical_levels_)
print(report["categorical"])
print(list(report["feature_groups"]))
for row in report["consensus"]:
    print(f"{row['feature_group']:>5}  median rank {row['median_rank']:.1f}")
```

```text
['city'] {'city': ['Kyiv', 'Lima', 'Oslo', 'Rome']}
{'columns': ['city'], 'train_levels': {'city': 4}, 'unseen_eval': {'city': 28}}
['city', 'size', 'age']
 size  median rank 1.0
 city  median rank 2.0
  age  median rank 3.0
```

`city` is one feature although it has four training levels; the 28 evaluation rows from Cairo got the shared unseen code instead of shaping the encoding. `ConsensusSelector(..., categorical="auto")` takes the same option and returns the selected raw columns from `transform()` (put an encoder after it in a `Pipeline` for the final model), and `featranker ... --categorical auto` (or `--categorical city,segment`) reads those CSV columns as text; an empty cell in such a column is the missing level.

**Limitations.** Levels are compared by their string forms, so `1` and `"1"` are the same level, and so are `1` and `1.0` (a column that pandas upcast to float because of a missing value still matches its integer training levels); clean such columns first. Target encoding and other supervised encoders are not offered, because they leak the target; build one into a custom model pipeline if you need it. Date and time columns are still rejected ("X must be numeric"), also when listed as categorical; convert them to numbers or to labels first. To give custom models column names, see [Columns by name](#columns-by-name).

## ▍ Sample weights

Survey data, imbalanced classes, and aggregated rows come with sample weights. Pass them as `sample_weight` to `fit()` (training rows), `rank_features()` (evaluation rows), or `rank_resampled()` (all rows; each split takes its slices), and the ranking answers the question your weighted model asks. Each is a 1-D array-like of finite, non-negative numbers, one per row, with a positive sum; anything else raises `ValueError` before any model is fit. The weights are copied.

**What is weighted.** With training weights, every model is fit with them: for a built-in pipeline they go to the final estimator only (the imputer, scaler, and one-hot encoder stay unweighted, as in scikit-learn's own pipelines), and the dummy baseline is fit with them too, so a weighted majority class is what it predicts. With evaluation weights, the baseline score, every permuted score, and the dummy's score are weighted; with `evidence=True` and training weights, the train score uses them. You can pass either set alone. The report records which were used as `sample_weight: {"train": ..., "eval": ...}` (see [Report](report.md#structure)); without weights the key is absent and reports are unchanged.

**Models that cannot take weights.** A model whose final estimator has no `sample_weight` in `fit()` is not fit unweighted, because its importances would answer a different question than the others'. It is recorded in `failures` with stage `"fit"` and `error_type` `"SampleWeightNotSupported"`. Among the built-in models these are `lda` and `qda` (classification) and `pls_regression`, so under weights the other built-in models vote without them. The same holds for your own models, bare or as the last step of a `Pipeline`; wrap or replace them with a weight-aware estimator if they must vote. An estimator whose `fit` takes `**kwargs` counts as weight-capable (a real error is then an ordinary fit failure), and a nested `Pipeline` is followed to its last step. FeatureRanker runs its own fits and scorers with scikit-learn metadata routing off, since it passes weights explicitly.

**Shuffle donors are not weighted.** The donors are the scored rows, that is, the evaluation rows after [`max_samples`](#fewer-evaluation-rows-max_samples), zero-weight rows included, and weights play no part in who donates. With `shuffle="rows"` each permutation is a random reordering of the scored rows, so every row donates exactly once whatever its weight; `"within_groups"` reorders rows only among rows with the same label; `"circular"` shifts the rows by a random offset instead of drawing. Shadow columns are shuffled copies of every row, also unweighted: the evaluation shadows are made from all evaluation rows before `max_samples`, and the training shadows from all training rows, zero-weight ones included. Only the scores are weighted. The permuted data therefore follows the unweighted distribution of the evaluation rows, while the baseline it is compared with is weighted. That suits cost or importance weights ("a mistake on these rows costs more"). For survey or frequency weights, where the weighted rows are meant to stand for a population, the permuted distribution is not the weighted population: a row with a large weight is no more likely to donate its value than a row with a small one. If the population is your target, drop zero-weight rows, or resample the evaluation rows by weight before ranking (then pass no `sample_weight` for them), and compare the results. FeatRanker has no option that draws donors by weight.

**Unweighted evidence.** Correlation evidence and the SHAP summary describe the data and the fitted models' behaviour, not the weighted objective, and stay unweighted. Custom scorers must accept a `sample_weight=` keyword when weights are given; scikit-learn scorer names do.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from featranker import FeatureRanker

rng = np.random.default_rng(0)
n = 600
survey = np.arange(n) % 2 == 0  # rows that carry almost all of the weight
x1, x2 = rng.normal(size=n), rng.normal(size=n)
X = np.column_stack([x1, x2, survey.astype(float)])
y = np.where(survey, x1, x2) + 0.1 * rng.normal(size=n)
w = np.where(survey, 1.0, 0.001)
names = ["x1", "x2", "is_survey"]
models = {"forest": RandomForestRegressor(n_estimators=50, random_state=0)}

for label, weights in [("unweighted", None), ("weighted", w)]:
    ranker = FeatureRanker(task="reg", models=models, verbose=False)
    ranker.fit(X[:400], y[:400], feature_names=names,
               sample_weight=None if weights is None else weights[:400])
    report = ranker.rank_features(
        X[400:], y[400:], scoring="r2", n_repeats=5, random_state=0,
        sample_weight=None if weights is None else weights[400:],
    )
    importance = report["models"]["forest"]["importance"]
    print(label, report.get("sample_weight"),
          {name: round(importance[name]["mean"], 2) for name in names[:2]})

builtin = FeatureRanker(task="clf", group="all", verbose=False)
builtin.fit(X[:400], (y[:400] > 0).astype(int), feature_names=names, sample_weight=w[:400])
print([(f["model"], f["error_type"]) for f in builtin.fit_failures_])
```

```text
unweighted None {'x1': 0.85, 'x2': 0.8}
weighted {'train': True, 'eval': True} {'x1': 1.44, 'x2': 0.02}
[('lda', 'SampleWeightNotSupported'), ('qda', 'SampleWeightNotSupported')]
```

`x1` drives the target in the heavily weighted rows and `x2` in the nearly weightless ones. Unweighted, the two look alike; weighted, `x2` all but disappears.

**Selector and pipelines.** `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)` passes the weights to its resampled ranking. Inside a `Pipeline`, use scikit-learn metadata routing: `sklearn.set_config(enable_metadata_routing=True)`, `ConsensusSelector(...).set_fit_request(sample_weight=True)`, then `pipeline.fit(X, y, sample_weight=w)`. Without routing enabled, scikit-learn raises its usual error for `sample_weight` on a pipeline. **The command line** takes `--sample-weight COLUMN`: the column must exist in both CSV files, is read as the weights (training and evaluation), and is not ranked as a feature.

<a id="ablation-curve"></a>

## ▍ Ablation curve

Permutation importance shows how much a fitted model loses when a group becomes noise. It does not show whether the top groups are enough: correlated groups share credit, so each can look small while removing both hurts. `ablate()` checks the ranking by refitting: for each `k`, every model that voted in the consensus is fit again on the columns of the top-`k` consensus groups and scored on the evaluation data. The curve answers "how many groups do I need": where it flattens, the later groups add little.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
names = list(data.feature_names)
X_dev, X_test, y_dev, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.4, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(X_val, y_val, scoring="r2", random_state=42)

# The same refits scored on the validation rows that produced the ranking
# and on the untouched test rows, for every k.
ks = [1, 2, 3, 5, 10]
validation = ranker.ablate(X_train, y_train, X_val, y_val, report, ks=ks)
test = ranker.ablate(X_train, y_train, X_test, y_test, report, ks=ks)
for on_val, on_test in zip(validation["ablation"]["curve"], test["ablation"]["curve"]):
    print(f"k={on_val['k']:>2}  validation {on_val['median_score']:.2f}  "
          f"test {on_test['median_score']:.2f}  added {on_val['feature_groups'][-1]}")
```

```text
k= 1  validation 0.26  test 0.04  added bmi
k= 2  validation 0.43  test 0.16  added s5
k= 3  validation 0.41  test 0.17  added bp
k= 5  validation 0.46  test 0.13  added s3
k=10  validation 0.44  test 0.15  added sex
```

The example is self-contained and uses only scikit-learn models (`group="fast"`), so the output does not depend on optional extras. Both curves refit the same models on the same training rows and the same top-`k` groups; only the scored rows differ. The validation curve rises from 0.26 at `k=1` to 0.43 at `k=2` and then stays within 0.41 to 0.46, so the first two or three groups carry most of it. The test rows score lower at every `k`, including `k=10`, where no groups were left out and nothing was selected: 0.15 against 0.44. So most of the gap between the validation and test scores is the difference between two small splits, not selection. Suppose you pick `k=3` from the validation curve: the validation rows chose `k`, so 0.41 is a validation score, and the untouched test rows (0.17 for the same groups) are the number to report. The validation score is still not that number.

How it works:

- **Groups and columns.** The top `k` rows of `report["consensus"]` are kept, in list order; their columns are the union of their `feature_groups` entries, in training column order. Groups are never split.
- **Models.** Only the voting models (`in_consensus: true`) are refit; models that lost to the dummy are skipped. An empty consensus raises `ValueError`.
- **Same preprocessing.** Each `k` builds a fresh `FeatureRanker` with the ranker's `task`, `group` or `models`, `n_jobs`, `missing`, `model_input`, and `categorical` columns (those among the kept ones), without shadows or evidence, and fits and scores it with the normal code path. Imputation, categorical coding, named model input, and sample weights therefore behave as in the ranking run. That ranker also takes one permutation per model, which is unused; its cost is small next to a refit.
- **Cost.** `len(ks)` refits of every voting model. A shorter `ks` list is the way to save time.
- **Failures.** A model that fails to fit or score at some `k` gets `null` there and an entry in `ablation["failures"]`; other models and ks continue. A model that cannot fit few columns (the built-in `pls_regression` with one column, for example) fails at small `k` this way. `median_score` is the median over the models that did not fail, and `null` if all did.
- **Paired median.** Because the models that fail can change from one `k` to the next, `median_score` can move although no model's score did. Each row therefore also has `n_models_scored` and `paired_median_score`, the median over `paired_models`, the models with a score at every `k` (`null` when there are none). Prefer `paired_median_score` when `n_models_scored` varies; the cohort is model-level, so a family with many models counts for more.
- **Provenance.** The block records `data_sha256` (a [fingerprint](#comparing-runs) of `X_eval` and `y_eval`), `weights_sha256` (of `eval_sample_weight`, `null` without), `n_eval_rows`, and `same_holdout`: `true` when that data and weights are the ones the report was ranked on, `false` for any other data set.
- **Summary.** `summarize()` adds an `ablation_curve` finding and, when the set of models with a score differs between rows, an `ablation_cohort_changed` warning; see [Summarize a report](#summarize).

> [!WARNING]
> By default the curve is scored on the evaluation data that also produced the ranking, so it is optimistic: the top groups were chosen because they scored well there. Passing a third data set as `X_eval` and `y_eval` removes that optimism for a `k` you fixed in advance. Once you use the curve to choose `k` (the highest score, or where it flattens), that data set has been used for selection and its best score is a validation score, which is still optimistic as an estimate of final performance. Report final performance on another untouched test set, or with outer cross-validation around the whole procedure (a `ConsensusSelector` in a `Pipeline` under `cross_val_score`; see [Feature selection in a pipeline](#feature-selection)). Choose `k` inside that outer loop, for example `GridSearchCV(pipe, {"select__k": [2, 3, 5]})` as the estimator passed to `cross_val_score` (nested cross-validation); a fixed `k` picked from the curve and then cross-validated is still optimistic, because the same rows chose it. The block's `baseline_score` is the dummy's score from the refits themselves (same training rows and weights, every `X_eval` row, the scorer used), so it is comparable to the curve, also with another `scoring`; it can differ from the report's `baseline_score`, which used the report's own rows and scorer and `max_samples`. It is `null` only if no refit scored.

The curve is descriptive. FeatRanker does not choose `k` for you, and it does not drop one group at a time (leave-one-group-out). For a resampled report there is no ablation; ablate a held-out report.

## ▍ Command line

The CLI takes separate training and evaluation CSV files; the features are numeric unless named by `--categorical`. `--scoring` takes one scorer name or several separated by commas (`--scoring accuracy,neg_log_loss`; the first is primary, see [Several metrics](#several-metrics)).

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

Both CSV files need headers, the same feature columns in the same order, and the target column. Feature values must be numeric; targets may be numeric or strings. With `--missing impute`, empty feature cells are read as missing values and imputed per training split (see [Missing values](#missing-values)); by default they are rejected. Only empty cells and the text `nan` (any case) are read as missing; `NA`, `N/A`, and `null` are not, and raise the numeric-values error. Add `--quiet` to hide progress bars and `--evidence` to add [evidence](#evidence) to the report. `--feature-groups` takes a YAML file, or the word `auto` to group correlated columns of the training CSV with [`auto_groups()`](#auto-groups) (a file literally named `auto` must be written `./auto`); `--auto-groups-threshold 0.5` changes its default 0.7 cut. `--categorical auto` reads the training-CSV columns that have a non-numeric cell as text (so a stray `NA` makes its whole column categorical; a column literally named `auto` cannot be listed, since `auto` always means detection), and `--categorical city,segment` reads the named columns as text (see [Categorical features](#categorical-features)); levels are learned from the training CSV only, and an empty cell in such a column is a missing level. `--sample-weight w` reads column `w` of both files as [sample weights](#sample-weights) and leaves it out of the features. `--max-samples 5000` (rows) or `--max-samples 0.25` (a share; a decimal point makes it a float) scores on a [subsample](#speed) of the evaluation rows. `--drop id,created_at` removes those columns from both CSV files before any feature is read, so an ID or timestamp column never becomes a feature (and a dropped text column is not read as categorical); an unknown name, the target, the `--sample-weight` column, or a `--categorical` column raises an error, and so does dropping every feature. `--ablate` adds the [`ablation` block](#ablation-curve) to the written report (it refits the voting models for each `k`, so the run takes longer); `--ablate-ks 1,2,5` sets the ks (comma-separated positive ints, and it needs `--ablate`), and a `k` above the number of feature groups raises an error before any model is fit. The CLI curve is scored on the evaluation CSV that chose the groups, so it is optimistic; see the warning in [Ablation curve](#ablation-curve). If no model beats the dummy, there is no consensus: `--ablate` prints a warning to stderr and the report is written without an `ablation` block. Run `featranker --help` for every flag.

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

### `.rank_resampled(X, y, *, scoring, cv=5, groups=None, feature_names=None, feature_groups=None, n_repeats=10, random_state=None, top_k=5, sample_weight=None, max_samples=None)`

| Parameter | Default | Meaning |
|---|---|---|
| `X`, `y`, `feature_names` | | All development data, with the same rules as `fit()`. Each split takes its own training and evaluation rows from it. |
| `scoring`, `feature_groups`, `n_repeats` | | Same as `rank_features()`; `scoring` may be a list, and the split reports then carry the secondary results (stability uses the primary scorer). |
| `cv` | `5` | An integer number of folds (shuffled; stratified for `task="clf"`), any scikit-learn splitter such as `RepeatedKFold` or `GroupKFold`, or a list of `(train_indices, eval_indices)` pairs. |
| `groups` | `None` | Passed to the splitter's `split()`, for example row groups for `GroupKFold`. With `shuffle="within_groups"` they are also the shuffle groups, sliced per split, and required. |
| `random_state` | `None` | Seeds integer-`cv` shuffling. Split `i` uses permutation seed `random_state + i`. |
| `sample_weight` | `None` | Row weights for all of `X`, sliced per split: each split trains with its training rows' weights and scores with its evaluation rows' weights. See [Sample weights](#sample-weights). |
| `max_samples` | `None` | As in `rank_features()`, applied to each split's evaluation rows (the split's own subsample, drawn with its seed `random_state + i`). |
| `top_k` | `5` | For `top_k_share`: a group counts as top in a split when its place in that split's consensus is at most `top_k`. Groups tied on both `median_rank` and `mean_rank` share the better place, so a split can have more than `top_k` top groups. |

Each split uses a new ranker with this ranker's settings, so the ranker you call it on is not changed. A split that fails entirely is recorded in `failures` with `stage: "split"` and the run continues; it raises only if no split ranks. Invalid `cv`, `top_k`, scorer, `n_repeats`, or feature groups raise before any model is fit. See [Resampled reports](report.md#resampled-reports) for the output.

**Cost.** One call costs about `n_splits` times a normal `fit()` plus `rank_features()`. With `group="all"` and `cv=10`, that is ten full runs; start with `cv=5`, a smaller `group` (such as `"fast"`), or fewer `n_repeats`.

> [!IMPORTANT]
> Pass development data only and keep your final test set out. The splits here are internal validation. Choosing features from these results and then scoring a model on the same data overstates its performance.
>
> Fit learned preprocessing within each training split. Put transformations compatible with FeatRanker's finite numeric input contract inside custom model pipelines. Missing values can be left in with `missing="impute"` (see [Missing values](#missing-values)). Raw categorical columns can be left in with `categorical=` (see [Categorical features](#categorical-features)): each split learns its own levels. For other preprocessing that needs learning, use a manual split workflow; do not fit it on all development rows before this call.

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
# Shuffles that respect the structure (see "Shuffling dependent rows" below).
ordered = FeatureRanker(task="reg", group="fast", shuffle="circular", verbose=False)
grouped = FeatureRanker(task="reg", group="fast", shuffle="within_groups", verbose=False)

# Rows are in time order: every split trains on the past and evaluates on the future.
by_time = ordered.rank_resampled(
    X, y, feature_names=names, scoring="r2", cv=TimeSeriesSplit(n_splits=4),
    n_repeats=3, random_state=0,
)
print("time:", [row["feature_group"] for row in by_time["stability"]])

# Rows from one patient never sit on both sides of a split.
by_patient = grouped.rank_resampled(
    X, y, feature_names=names, scoring="r2", cv=GroupKFold(n_splits=5), groups=patients,
    n_repeats=3, random_state=0,
)
print("patient:", [row["feature_group"] for row in by_patient["stability"]])

# The same splitters in a pipeline, so selection never sees the rows it is scored on.
pipeline = Pipeline([
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

**Within-group shuffling answers a narrower question.** `shuffle="within_groups"` measures how much the model relies on variation within groups only. A predictor that varies only between groups (a patient-level value repeated on each of the patient's rows) gets an importance of about 0 by construction, even if it drives the target. To measure group-level effects use `shuffle="rows"` (free shuffles, with `GroupKFold` or another grouped splitter for the splits). Such predictors are listed in the report's `constant_within_groups`, with a `UserWarning`; see [Shuffling dependent rows](#shuffling-dependent-rows).

The splitter only controls how rows are divided. By default, shadow features shuffle rows within each training and evaluation set, and permutation importance shuffles rows within the evaluation set, which ignores any dependence between neighbouring rows (autocorrelation) or between rows of one group. The `shuffle` option, used for `ordered` and `grouped` above, makes both shuffles respect that structure; see [Shuffling dependent rows](#shuffling-dependent-rows). Without it, treat shadows and the hits test as less reliable on such data.

### Shuffling dependent rows

`FeatureRanker(shuffle=...)` (and `ConsensusSelector(shuffle=...)`) sets how permutation importance and shadow columns shuffle rows:

| `shuffle` | Behaviour |
|---|---|
| `"rows"` (default) | Rows are shuffled freely, as in earlier versions; reports are unchanged. |
| `"within_groups"` | A column's values move only among rows with the same group label. Pass one label per row as `shuffle_groups` to `fit()` (shadows) and to `rank_features()` (permutations); `rank_resampled()` and `ConsensusSelector.fit()` use their `groups` and slice them per split. A row in a group of one keeps its value; if every group has one row the call raises `ValueError`. With few rows per group there are few distinct shuffles, so importances are noisier. Labels must be sortable: `None`, NaN and mixed types raise `ValueError`. |
| `"circular"` | Each shuffle is a cyclic shift of the rows by a random offset from 1 to n-1, which keeps a series' autocorrelation. It assumes the rows are in time order (as `TimeSeriesSplit` does); on unordered rows it is just a shift and means nothing. `rank_resampled()` sorts each split's train and evaluation rows into their original order first. |

**What `"within_groups"` measures.** Moving values only among rows of the same group tests how much the model relies on variation *within* groups. It cannot test reliance on variation *between* groups: a predictor that is constant within each group (a patient-level measurement repeated on every row of the patient) is not changed by the shuffle, so its importance is 0 by construction, however strongly it drives the target. In a check with 40 patients of 4 rows, a linear model, and a patient-level `x` that sets `y` (`y = 3x` plus small noise), the importance of `x` was about 2 under free shuffling (1.7 to 2.1 across seeds and fit/evaluation splits) and exactly 0.0 under `shuffle="within_groups"`. To measure group-level effects use `shuffle="rows"` with a grouped splitter (`GroupKFold`) for the data split, so that the shuffle is free but rows of one group still never sit on both sides of a split. After `rank_features()` the report lists such feature groups in `constant_within_groups` (see below), and `summarize()` warns.

**`constant_within_groups`.** With `shuffle="within_groups"`, `rank_features()` records the feature groups whose every column has one value (a missing value counts as a value) within every shuffle group on the scored rows, that is, after `max_samples`. The key is a sorted list placed after `shuffle` (empty when no group qualifies) and is absent with the other modes. A non-empty list also raises a `UserWarning`. Each split report of `rank_resampled()` carries its own list, and the resampled report has a top-level `constant_within_groups` with the sorted union of the split lists (`[]` when none). `summarize()` of a held-out or a resampled report adds a `constant_within_groups` warning. A group that is constant on the scored rows but varies elsewhere is also listed; the list describes the scored rows. A group with a column that varies is not listed even if other columns are constant, so constant columns inside a multi-column group are not flagged. Near-constant predictors, which vary in only a few groups, are not flagged either. The list only catches the exact case; the shuffle still barely moves such columns, and their between-group effect is still not measured.

Joint shadows share one offset (or one within-group order); `shadow_mode="independent"` draws one per column. With `max_samples`, permutations shuffle the sampled rows only (their labels, or a shift within the sample). Under `"within_groups"` the subsample is made of whole groups, drawn in random order until the target row count is reached, so it can slightly exceed it (`n_eval_rows` records the actual count). Under `"circular"` the sampled rows keep their order, but subsampling thins the series, so a shift no longer moves values by whole time steps. Correlation and SHAP evidence do not shuffle and are unchanged. `shuffle_groups` is required with `"within_groups"` (also when `shadows=False`, in both `fit()` and `rank_features()`, though `fit()` then ignores it and accepts all-singleton labels) and is an error with another mode. A report records `shuffle` when it is not `"rows"`. The CLI has no group column and always uses `"rows"`.

The recipe above ranks the time-ordered rows with `shuffle="circular"` and the patient rows with `shuffle="within_groups"`; for single fits:

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
- `rank_resampled()` uses the same setting; split `i` uses `shadow_seed + i`, and each stability row gains `above_shadow_share` (the sum of the per-split values divided by `n_splits`: a split that failed, had an empty consensus, or lacks the group counts as 0, so a group with a consensus row in only one split of ten has a share of at most 0.1; `ConsensusSelector(min_shadow_share=...)` filters on this value), plus the [hits test](#shadow-test) keys and the [family evidence](#family-evidence) keys.

> [!WARNING]
> Shadows only work if every model sees every column. A custom pipeline that drops columns, such as a positional `ColumnTransformer` with the default `remainder="drop"`, never sees the shadows, so every group trivially beats them. Shadow columns are appended after the real ones, so positional selectors of real columns still work.

<a id="shadow-test"></a>

### Shadow hits test

With shadows on, `rank_resampled()` also tests, for every group, whether it beats its shadows across splits more often than chance, at no extra model fits. This is Boruta's hits test applied to your splits. In each split, the group has a *hit* when its consensus `above_shadow_share` is strictly greater than 0.5: a majority of the family-weighted vote put it above the strongest shadow. Each stability row gains:

- `shadow_hits`: the number of splits that were hits. A split where the group has no consensus row (a failed or empty split) is a non-hit, and the number of trials is always `n_splits`.
- `shadow_p_value`: the one-sided binomial probability of at least `shadow_hits` hits in `n_splits` trials with success probability 0.5.
- `shadow_q_value`: the Benjamini-Hochberg adjusted p-value across all groups.

A 0.5 null probability and false-discovery-rate control are not established for arbitrary estimators or overlapping splits; these binomial reference values should not be interpreted as calibrated significance. The report says so: it has a `shadow_test` block (method, hit rule, `null_probability`, correction, `n_trials`, `min_p_value`, `calibrated: false`, and a `note`). Calibration study ([details](calibration.md)): on null data (1000 rows, 10 independent features, 10 `RepeatedKFold` splits, built-in `fast` and `tree` models) no group had `shadow_p_value <= 0.05`, so the binomial reference values were conservative there. That is partly built in: with 10 splits the test's real level is 0.0107 (at least 9 hits), and in regression most pure-noise runs had too few splits with a consensus for the test to fire at all, so classification is the informative case. The values remain not calibrated for custom estimators, other data, or dependent rows. The test also has low power: `max_shadow_q=0.05` recovered the `c·d` interaction in 0 to 75% of seeds.

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

<a id="family-evidence"></a>

### Family evidence

`above_shadow_share` averages one vote per model family, so it measures agreement across families. A group that only some families can model, such as a pure interaction that additive models cannot see, can be found by those families and still average below 0.5. With shadows on, the report therefore also says which families found each group:

- Each consensus row gains `family_above_shadow`: for each voting family, the fraction of its voting members whose importance beat that model's strongest shadow. `above_shadow_share` is the mean of these values; keys are sorted by family name.
- In a resampled report, each stability row gains `family_above_shadow_share` (for each family that voted in at least one split, the sum of its per-split fractions divided by `n_splits`: a split where the family did not vote, failed, or was empty counts as 0, as for `shadow_hits`, so a family that passes the dummy baseline in only a few splits cannot reach a majority) and `shadow_families`, the number of families whose share is strictly above 0.5.

The example plants an interaction (`x1` and `x2` matter only together, through a logistic link, plus one linear feature `x0` and two noise columns) and keeps groups with `ConsensusSelector`:

```python
import numpy as np
from sklearn.model_selection import RepeatedKFold

from featranker import ConsensusSelector

# x0 is linear, x1 and x2 matter only together (an interaction), x3 and x4 are noise.
rng = np.random.default_rng(0)
X = rng.normal(size=(800, 5))
score = 1.5 * X[:, 0] + 2.0 * X[:, 1] * X[:, 2]
y = rng.binomial(1, 1 / (1 + np.exp(-score)))  # logistic link, sampled 0/1 labels

options = dict(
    group="fast", cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
    n_repeats=3, random_state=0, shadow_seed=0,
)
by_share = ConsensusSelector("clf", "roc_auc", min_shadow_share=0.5, **options).fit(X, y)
by_family = ConsensusSelector("clf", "roc_auc", min_shadow_families=2, **options).fit(X, y)

for row in by_family.report_["stability"]:
    print(
        f"{row['feature_group']}  above_shadow_share {row['above_shadow_share']:.2f}  "
        f"shadow_families {row['shadow_families']}"
    )
row = next(r for r in by_family.report_["stability"] if r["feature_group"] == "x2")
print({family: round(share, 2) for family, share in row["family_above_shadow_share"].items()})
print("min_shadow_share=0.5   keeps", by_share.selected_groups_)
print("min_shadow_families=2  keeps", by_family.selected_groups_)
```

```text
x0  above_shadow_share 1.00  shadow_families 7
x1  above_shadow_share 0.53  shadow_families 3
x2  above_shadow_share 0.41  shadow_families 3
x4  above_shadow_share 0.19  shadow_families 1
x3  above_shadow_share 0.06  shadow_families 0
{'bagging': 1.0, 'boosting': 1.0, 'discriminant': 0.0, 'linear_margin': 0.0, 'logistic': 0.0, 'naive_bayes': 0.3, 'single_tree': 0.6}
min_shadow_share=0.5   keeps ['x0', 'x1']
min_shadow_families=2  keeps ['x0', 'x1', 'x2']
```

Output from scikit-learn models only (`group="fast"`, seven classification families). The four additive families (logistic, linear margin, discriminant, naive Bayes) do not put `x2` above their shadows in most splits (shares 0.0 to 0.3), so its share stays below 0.5 and the share rule drops it (it keeps `x1` only because the share is 0.53), while bagging, boosting and the single tree, which can represent the interaction, put `x2` above its shadows in most splits, so three families pass.

`ConsensusSelector(min_shadow_families=N)` keeps groups with `shadow_families >= N`. It answers "do at least `N` families find signal?" and leaves `min_shadow_share`, `max_shadow_q`, and their meanings unchanged; it turns shadows on and combines with the other rules (all filters apply, then `k`).

- **When to use it.** When you suspect interactions or effects that only some model types capture, and a majority of families cannot be expected to find them.
- **Choosing `N`.** `N=1` keeps a group any single family finds, so one family fitting noise (for example a single decision tree) can pass noise on its own; in the table above `x4`, a noise column, already has `shadow_families` of 1. Requiring two families means two kinds of model have to agree. A family that passes the dummy baseline in only a few splits is counted as 0 in the others, so it cannot reach a majority from those few splits. The [calibration study](calibration.md) measured this on synthetic data. Its preregistered rule recommends `N=1` for regression and classification: `N=1` and `N=2` each selected at most 1% of noise groups on null data, and with `group="fast"` both found the `c·d` interaction in every seed. Post hoc, `N=1` kept some noise in the signal study (up to 0.4 of 4 noise features per run) and `N=2` at most 0.1, with no loss of `c·d` recall; the regression null cannot separate them. Practical guidance: use `N=1` if recall matters more, `N=2` if false selections matter more, and avoid `N=3` with `group="fast"` in classification (it found `c·d` in only 0 to 15% of seeds). One synthetic design with built-in models; treat `min_shadow_families` as a heuristic filter, like `max_shadow_q`.
- **Which families count.** Every voting family counts equally, including families that cannot represent the effect you care about. FeatRanker does not choose "suitable" families for you; restrict them with `group` if you want to. With your own `models=`, each model is its own family unless `families=` maps it, so `N` counts models (or the families you defined). With the built-in `group="tree"`, regression and classification each have three families (bagging, boosting, single tree), so `N=2` is a majority there; `"fast"` has seven classification and five regression families.

<a id="shadow-mode"></a>

### Shadow mode

By default (`shadow_mode="joint"`) the shadow block is the rows in one shuffled order, applied to every column, so two collinear real columns stay exactly as collinear in the shadows. `shadow_mode="independent"` shuffles each shadow column with its own row order instead: every shadow column is still a permutation of its real column, but relationships between columns are broken. The orders are drawn column by column from one generator seeded with `shadow_seed` (the evaluation shadows use their own seed, as before), so `shadow_seed` still makes runs reproducible, and `"joint"` produces exactly the shadows of earlier versions. The report records the mode as `shadow_mode` next to `shadows` and `shadow_seed`; reports without shadows have no `shadow_mode`.

The [calibration study](calibration.md) found no evidence for or against `independent` (small pooled gains in recall of the collinear source, within seed noise); the default stays `joint`. The mode changes which shadow is strongest, so results can differ between modes. Both `FeatureRanker` and `ConsensusSelector` accept `shadow_mode`.

```python
import numpy as np
from sklearn.model_selection import RepeatedKFold

from featranker import FeatureRanker

# x0 and x1 are near copies of each other; x2 is noise.
rng = np.random.default_rng(0)
x0 = rng.normal(size=400)
X = np.column_stack([x0, x0 + 0.05 * rng.normal(size=400), rng.normal(size=400)])
y = x0 + 0.5 * rng.normal(size=400)
names = ["x0", "x1", "x2"]

for mode in ("joint", "independent"):
    ranker = FeatureRanker(
        task="reg", group="fast", shadows=True, shadow_seed=0, shadow_mode=mode, verbose=False
    )
    report = ranker.rank_resampled(
        X, y, feature_names=names, scoring="r2", n_repeats=3, random_state=0,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
    )
    shares = {r["feature_group"]: round(r["above_shadow_share"], 2) for r in report["stability"]}
    print(report["shadow_mode"], shares)
```

```text
joint {'x0': 1.0, 'x1': 1.0, 'x2': 0.1}
independent {'x0': 0.96, 'x1': 1.0, 'x2': 0.1}
```

Output from scikit-learn models only (`group="fast"`). The two modes agree on this data apart from a small shift in `x0`.

<a id="evidence"></a>

## ▍ Evidence

A rank says which groups matter, not why. With `evidence=True`, `fit()` also keeps the training data (as its own copy, so changing your arrays afterwards does not change the evidence) and `rank_features()` adds evidence about the mechanism behind the ranking. Because of that copy, a pickled ranker with `evidence=True` contains its training data: expect the file to grow by roughly the size of the training arrays (about twice that with shadows on, which adds the shadow columns), and treat it with the same care as the data.

- **Train score and overfit gap.** Each model gets `train_score` (the same scorer on the data it was fit on) and `overfit_gap = train_score - evaluation_score`. A large gap means the model scores much better on its training data than on the evaluation data. Memorization is one cause; a shift between the training and evaluation distributions, or a small evaluation sample, can produce the same gap, so check those before concluding that the model does not generalize.
- **Correlated groups.** `evidence.correlation` lists pairs of feature groups whose strongest member-column Spearman correlation on the **training** data is at least 0.5, with its sign. Permutation importance can split credit between correlated groups; consider putting them in one `feature_groups` entry. Categorical columns are left out.
- **SHAP summary** (needs `pip install 'featranker[shap]'`). For each linear model (`LinearExplainer`) and each tree model SHAP's tree explainer supports, including a `Pipeline` whose earlier steps are all `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, or `RobustScaler`, or identity steps (`"passthrough"`, `None`) (any other pipeline step lists the model as unsupported), `models[name]["shap"]` gives every group's `mean_abs` (mean absolute SHAP value, summed over the group's columns) and `direction` (Spearman correlation between a single-column group's value and its SHAP value; positive means higher values push the explained output up). Other models appear in `evidence.shap.unsupported`, including linear models with a one-hot step for [categorical features](#categorical-features); there is no model-agnostic fallback. Without `shap`, a `UserWarning` is raised and the rest of the evidence is still computed.

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

### `ConsensusSelector(task, scoring, *, k=None, min_shadow_share=None, max_shadow_q=None, min_shadow_families=None, ...)`

| Parameter | Default | Meaning |
|---|---|---|
| `task`, `scoring` | required | As in `FeatureRanker` and `rank_features()`; `scoring` may be a list (the first scorer selects, the rest are reported in `report_`). |
| `k` | `None` | Keep the first `k` groups in the [stability](report.md#resampled-reports) order (or all groups if there are fewer). Python or NumPy integer. |
| `min_shadow_share` | `None` | Keep groups whose stability `above_shadow_share` is at least this (0 to 1). Turns on [shadow features](#shadow-features). |
| `max_shadow_q` | `None` | Keep groups whose stability `shadow_q_value` (the [shadow hits test](#shadow-test)) is at most this (greater than 0, at most 1); a heuristic filter, since the values are [uncalibrated](#shadow-test). Turns on shadow features. Use `cv` with 10 or more splits, such as `RepeatedKFold`; with 5 the test has little power. |
| `min_shadow_families` | `None` | Keep groups whose stability `shadow_families` (see [family evidence](#family-evidence)) is at least this (a positive integer). Turns on shadow features. `1` and `2` both selected at most 1% of noise groups on null data; `1` kept more noise than `2` in the signal study (see the [calibration study](calibration.md) for how to choose); a heuristic. |
| `group`, `models`, `families`, `n_jobs`, `shadow_seed`, `shadow_mode`, `missing`, `categorical`, `model_input`, `shuffle` | as `FeatureRanker` | Passed to the ranker built in `fit()`. |
| `feature_groups`, `cv`, `n_repeats`, `random_state`, `max_samples` | as `rank_resampled()` | Passed to `rank_resampled()` on the data given to `fit()`. |
| `feature_groups="auto"` | | Groups correlated columns with [`auto_groups()`](#auto-groups) on the data given to `fit()`, so inside a pipeline from the training fold only. The groups used are in `report_["feature_groups"]`. Any other string raises `ValueError` in `fit()`. |
| `auto_groups_threshold` | `0.7` | The |rho| cut of `feature_groups="auto"`, as `auto_groups(threshold=...)` (`0 < threshold <= 1`, else `ValueError` in `fit()`). Recorded in `report_["run"]["auto_groups"]`; ignored for other `feature_groups`. |
| `verbose` | `False` | Progress bars; off by default because pipelines fit many times. |

Set at least one of `k`, `min_shadow_share`, `max_shadow_q`, and `min_shadow_families`; the shadow filters apply first, then the first `k` that pass. Invalid rules raise `ValueError` in `fit()`, before any ranking, following scikit-learn's convention that `__init__` only stores parameters (so `clone`, `set_params`, and `GridSearchCV` over `select__k` work). To choose `k` honestly, put that `GridSearchCV` inside `cross_val_score` (nested cross-validation); see the warning in [Ablation curve](#ablation-curve).

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
- If any split failed or produced no consensus, `fit()` warns how many of the splits did not vote. Those splits already count as misses in the shares, so the selection still happens; look at `report_["failures"]` and `report_["splits"]` to see why.
- If no group passes, `fit()` warns and the selector keeps no columns, as scikit-learn's own selectors do. Most downstream estimators (such as `Ridge`) then fail on zero columns, so that pipeline fit errors out. Loosen `min_shadow_share`, `max_shadow_q`, or `min_shadow_families`, or use only the top-k rule by setting the shadow rules to `None` and a positive `k`.
- For a group-aware inner `cv` such as `GroupKFold`, pass row groups to `fit(X, y, groups=...)`; in a `Pipeline`, pass `select__groups=...` (for example `cross_val_score(..., params={"select__groups": groups})`, which slices them to each training fold). With `shuffle="within_groups"` the same groups also confine the permutation and shadow shuffles (see [Shuffling dependent rows](#shuffling-dependent-rows)).
- DataFrame column labels are compared as strings, in `fit()` and `transform()`. A failed `fit()` clears any earlier fitted state.
- **Cost.** Each `fit()` is one `rank_resampled()` run: `cv` rankings, doubled with shadows. Inside `cross_val_score(cv=n)` that is `n × cv` rankings; fitting the final pipeline afterwards adds one more run. `GridSearchCV` multiplies the cost by the number of candidates, plus its final refit. Use `group="fast"` or `"tree"`, a small `cv`, and fewer `n_repeats` while searching.

<a id="summarize"></a>

## ▍ Summarize a report

`summarize(report)` applies the [reading guide](report.md#reading-the-report) for you. It returns a list of findings and does not change the report. It works on the dict from `rank_features()`, on a report loaded from the CLI's JSON, and on a [resampled report](#resampled-ranking). A resampled report has its own findings (below); to read one split, summarize `report["splits"][i]["report"]`.

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
| `constant_within_groups` | warning | Held-out or resampled report ranked with `shuffle="within_groups"`: feature groups constant within every shuffle group; their importance is zero by construction. Evidence: `groups`. |
| `ablation_curve` | info | The report has an [`ablation`](#ablation-curve) block with a score that is not `null`: the `k` with the highest `paired_median_score` (the smallest on ties) and its score at the largest `k` with one; when `paired_models` is empty, the same on the unpaired `median_score`, and the message says so. Evidence: `best_k`, `best_median_score`, `max_k`, `max_k_median_score`. |
| `ablation_cohort_changed` | warning | The `ablation` curve's rows have different sets of scored models, so the unpaired `median_score` compares different model sets. Evidence: `rows`, a list of `k`, `n_models_scored` and `models`. |

An empty list means nothing stood out, not that the ranking is proven. The checks cover one evaluation set.

<a id="summarize-resampled"></a>

### Summarizing a resampled report

For `rank_resampled()` reports, `summarize()` returns these findings, in this order, each only when its condition holds:

| Code | Severity | When |
|---|---|---|
| `split_failures` | warning | Splits that failed outright (`failures` with stage `"split"`), with their indices and error types. They count as misses in every share. |
| `failures` | as for a single split | Model failures aggregated over splits, one finding per stage, each model listed with the splits it failed in. |
| `gate_off` | warning | Splits whose `baseline_score` is `None`. |
| `empty_consensus` | warning | Splits whose consensus is empty. |
| `below_shadow` | info | With shadows on, groups whose stability `above_shadow_share` is below 0.5. |
| `insufficient_splits` | info | With shadows on, the [shadow hits test](#shadow-test) cannot reach a q-value of 0.05 with this many splits (`shadow_test.min_p_value > 0.05`, which is the case below 5 splits). |

```python
from sklearn.datasets import load_diabetes

from featranker import FeatureRanker, summarize

data = load_diabetes()
ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
report = ranker.rank_resampled(
    data.data, data.target, feature_names=list(data.feature_names),
    scoring="r2", cv=3, n_repeats=3, random_state=0,
)
for finding in summarize(report):
    print(finding["severity"], finding["code"], "-", finding["message"])
```

```text
info below_shadow - s3, sex, s6, age, s4 have a family-weighted above-shadow share below 0.5. Each family contributes its fraction of voting members beating that model's strongest shadow group. This is descriptive, not a significance test.
info insufficient_splits - With 3 splits the smallest possible shadow p-value is 0.125, so no group can reach a q-value of 0.05; use more splits, e.g. RepeatedKFold.
```

Output shown with the `xgboost` and `catboost` extras installed. The per-split findings (`few_families`, `tied_groups`, and the rest) are not repeated here; summarize a split's own report for those.

<a id="comparing-runs"></a>

## ▍ Comparing runs

Every report has a `run` block that says what it was computed from, so two reports can be told apart without keeping the data:

- `task`, `group`: the preset.
- `fingerprint_scheme`: `2` since 1.17: integer and boolean columns (for example class labels) are hashed exactly, tagged by dtype kind (not size), where scheme 1 converted them to float64 and so could not tell integers above 2**53 apart. Booleans hash as the integers 0 and 1 and `int32` equals `int64` for the same values; an integer column and a float column holding the same numbers still differ. Float and text columns hash as before. Fingerprints of integer or boolean data from earlier versions do not equal 1.17 ones; `ablate()` reads a report without this key as scheme 1.
- `data_sha256`: a fingerprint of the evaluation data for `rank_features()`, and of all the data passed to `rank_resampled()`.
- `train_data_sha256`: a fingerprint of the data passed to `fit()` (`rank_features()` reports only; `null` for a ranker pickled before 1.12).
- `weights_sha256` (and `train_weights_sha256` for `rank_features()`): a fingerprint of the sample weights, present only when weights were given.
- `shuffle_groups_sha256` (and `train_shuffle_groups_sha256` for `rank_features()`): a fingerprint of the shuffle group labels, present only when labels were given.
- `rank_resampled()` only: `cv` (the integer given, the splitter's `repr` without memory addresses, such as `KFold(n_splits=5, random_state=0, shuffle=True)`, or `"iterable of splits"`) and `splits_sha256`, a fingerprint of every split's train and evaluation row indices. Each split's own report has its `run` as well.
- `auto_groups`: `{"threshold": 0.7, "method": "spearman-complete-linkage"}`, only when [`feature_groups="auto"`](#auto-groups) chose the groups (`ConsensusSelector`, `--feature-groups auto`).

A fingerprint is the SHA-256 digest of the feature names, the feature values, and the target values (for this `fingerprint_scheme`). Equal inputs under the same library versions give equal fingerprints; changing any value, a name, or the row order changes it. Fingerprints may differ across NumPy or pandas versions, notably for text and other object columns (the versions are in `provenance`); a categorical cell `None` in NumPy input and a NaN from pandas hash differently. They do not cover model settings.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
names = list(data.feature_names)


def rank(X_eval, y_eval):
    ranker = FeatureRanker(task="reg", group="fast", verbose=False)
    ranker.fit(X_train, y_train, feature_names=names)
    return ranker.rank_features(X_eval, y_eval, scoring="r2", n_repeats=3, random_state=0)


first = rank(X_eval, y_eval)
print("same inputs:", first["run"] == rank(X_eval, y_eval)["run"])

changed = y_eval.copy()
changed[0] += 1.0
other = rank(X_eval, changed)
print("evaluation data differs:", first["run"]["data_sha256"] != other["run"]["data_sha256"])
print("training data differs:", first["run"]["train_data_sha256"] != other["run"]["train_data_sha256"])
```

```text
same inputs: True
evaluation data differs: True
training data differs: False
```

To compare two resampled reports, check `run["data_sha256"]`, `run["cv"]`, and `run["splits_sha256"]` first: with all three equal, the stability tables came from the same rows in the same splits.

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

Permutation importance splits credit between near-copies: shuffling one of two correlated columns barely hurts a model that can use the other, so both look unimportant. `auto_groups(X, *, threshold=0.7, feature_names=None, categorical=None)` proposes the groups from `X` alone (it never sees a target) and returns a dict you can pass as `feature_groups`.

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
- `X` is validated like `FeatureRanker.fit()` input. Column names come from a DataFrame, else `feature_names`, else `x0`, `x1`, …. With `categorical` (as in `FeatureRanker`: `"auto"` or a list of names) those columns are accepted as text and never grouped; `ConsensusSelector` and the CLI pass it on.
- Call it on your training or development data and pass the result to `rank_features()` or `rank_resampled()`; there is no `"auto"` option there, so the grouping stays visible in your code. The [correlation evidence](#evidence) lists group pairs from 0.5, which shows what sits near the cut.
- The correlation matrix and clustering are O(p²) in the number of columns, which is fine for hundreds of columns.

Where you cannot compute the groups yourself, use `"auto"`: `ConsensusSelector(feature_groups="auto")` in a pipeline, or `--feature-groups auto` on the command line. Both use the default threshold of 0.7; set `ConsensusSelector(auto_groups_threshold=...)` or `--auto-groups-threshold` to change it. The report records the threshold in `run["auto_groups"]`, and only when the grouping was automatic.

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
| `X must contain only finite values; impute or drop missing ...` | NaN or infinite values in the features. | Impute or drop them before calling FeatRanker, or pass `missing="impute"` for NaN (infinite values are always rejected). |
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
