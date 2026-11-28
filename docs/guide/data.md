# Data

English · [简体中文](../zh-CN/guide/data.md)

This page covers missing values, categorical features, sample weights, and how FeatRanker treats labels and your arrays.

[Manual](../README.md) | [Custom models](custom-models.md) | [Validation](validation.md) | [Reference](../reference.md)

## Missing values

By default NaN in the features raises `ValueError`. With `missing="impute"` FeatRanker accepts NaN (infinite values are still rejected) and imputes inside each model, so no evaluation row shapes an imputation.

- Every built-in model becomes a pipeline that starts with `SimpleImputer(strategy="median")`, followed by the usual scaler for linear models. The imputer is fit on the rows the model is fit on and nothing else. Under `rank_resampled()` and `ConsensusSelector` each training split learns its own medians.
- This holds for every built-in model, including those that handle NaN natively (such as `HistGradientBoosting`), so one rule applies across models and scikit-learn versions.
- A column that is missing in every training row becomes a constant 0 and keeps its position. Its importance is meaningless; `train_counts` shows it.
- Your own `models=` receive the NaN unchanged and must handle it, for example with `make_pipeline(SimpleImputer(...), ...)`.
- The dummy baseline ignores feature values and is unaffected.
- Evidence works with NaN. Correlated groups and `auto_groups()` use, for each pair that involves a column with NaN, the rows where both are present. A pair with fewer than 30 such rows counts as uncorrelated. SHAP treats the leading imputer as a per-column step. Pairwise correlation costs O(columns squared times rows) when NaN is present.
- The report gains `missing` with the number of missing training values per feature ([Report](../report.md#structure)).

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

imputer = ranker.trained_models["ridge_regression"][0]  # holds the training median of bmi
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

The last line shows the fitted imputer's median for `bmi` equal to the training median (-0.0105), not the evaluation one (0.0013).

`ConsensusSelector("reg", "r2", k=2, missing="impute")` accepts NaN in `fit()` and `transform()`. It only chooses columns, so put an imputer after it in a `Pipeline`. On the command line, `--missing impute` reads empty CSV cells as missing.

> [!WARNING]
> Median imputation mixes missingness with values. A lab test ordered only for sick patients is missing for a reason, and imputation leaves a spike at the median that tree and kernel models can learn. A feature's importance can then mix its values with its missingness (in a probe where only missingness drove the target, tree importance was about 1.9 and SVR about 1.05, while ridge gave 0.006). To separate the two, use custom models with `SimpleImputer(add_indicator=True)` or native NaN handling, and read `train_counts`. Missing values in the target are not supported.

## Categorical features

By default every feature must be numeric. With `categorical`, FeatRanker accepts text and category columns as they are, learns how to encode them inside each `fit()`, and ranks each raw column as one feature.

- `categorical="auto"` treats DataFrame columns of object, string or category dtype as categorical (bool and numeric dtypes stay numeric). A NumPy array has no per-column dtypes, so `"auto"` finds nothing there.
- `categorical=["city", ...]` marks those columns whatever their dtype, and works for NumPy object arrays too (give `feature_names`). A name that is not a feature raises `ValueError` in `fit()`.
- **Levels come from the training rows only.** Each column's levels are the sorted string forms of its training values, coded `0..k-1`; a missing value (`None` or NaN) is its own level, last. Values not seen in training, including a missing value when training had none, share one extra "unseen" code `k` at evaluation time. Every split of `rank_resampled()` and `ConsensusSelector` learns its own levels. The fitted ranker exposes `categorical_features_` and `categorical_levels_` (name to levels, `None` for the missing level).
- **One feature per raw column.** `city` is one group in `feature_groups`, in the consensus and in the shadows, however many levels it has.
- **Built-in linear models** one-hot encode categorical columns (a code never seen in training gives an all-zero row) and scale the numeric ones. **Built-in tree models** use the codes as numbers, which gives the levels an arbitrary order. **Your own models** receive the code matrix (floats); read `ranker.categorical_features_` to know which columns hold codes. For high-cardinality columns or another encoding, pass your own `models=`.
- Under `missing="impute"` the numeric columns are imputed first, and a missing categorical value is a level.
- The report gains `categorical`: the columns, their number of training levels, and how many evaluation values were unseen.
- Evidence: categorical columns are left out of the Spearman correlations (`auto_groups(X, categorical=...)` never groups them). SHAP explains tree models on the codes (the `direction` of a categorical group is `null`) and built-in linear models with the one-hot columns summed per source column.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
city = rng.choice(["Oslo", "Rome", "Lima", "Kyiv", "Cairo"], n)
premium = {"Oslo": 30.0, "Rome": -20.0, "Lima": 25.0, "Kyiv": -25.0, "Cairo": 0.0}
size = rng.uniform(30, 120, n)
age = rng.uniform(0, 60, n)
price = size * 2 + np.array([premium[c] for c in city]) + rng.normal(0, 8, n)
df = pd.DataFrame({"city": city, "size": size, "age": age})
train, evaluation = df.iloc[:280], df.iloc[280:]
train = train.assign(city=train["city"].replace("Cairo", "Rome"))  # Cairo never appears in training

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

`city` is one feature although it has four training levels. The 28 evaluation rows from Cairo got the shared unseen code instead of shaping the encoding. `ConsensusSelector(categorical="auto")` returns the selected raw columns from `transform()`; put an encoder after it in a `Pipeline`. The CLI takes `--categorical auto` or `--categorical city,segment`.

### Sparse one-hot output

Since 2.0 the one-hot block stays sparse (scikit-learn's own sparse solvers) for every built-in linear model except six dense-only estimators: `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `GaussianNB`, `PLSRegression`, `SGDClassifier` and `Perceptron`. They get dense output (SGD's intercept update differs on sparse input).

- Sparse and dense solve the same optimisation problem, but solver-level differences can still move a few predictions on small samples (for example `ridge_classifier` and `huber_regression`).
- A one-hot block with density 1.0 (for example when the only categorical column has a single level) still gets dense output. Logistic models with `solver` `saga` or `sag` also get dense one-hot, because sparse input made saga stop early.
- Sparse input fixes a divergence of `linear_regression` and `ransac_regression` with categorical columns (dense least squares diverged on a full one-hot block plus intercept, giving scores like -5e21, from 1.9 through 1.x). Their least-squares solution on a full one-hot block is not unique, so their scores with categorical columns can also differ from the dense 1.x results when dense did not diverge.
- `max_categories` bounds the one-hot width; see [Performance](performance.md#large-data).

> [!WARNING]
> Levels are compared by their string forms, so `1` and `"1"` are the same level, and so are `1` and `1.0`. Clean such columns first. Target encoding and other supervised encoders are not offered, because they leak the target; build one into a custom pipeline if you need it. Date and time columns are rejected ("X must be numeric"), also when listed as categorical; convert them to numbers or labels first.

## Sample weights

Survey data, imbalanced classes and aggregated rows come with sample weights. Pass them as `sample_weight` to `fit()` (training rows), `rank_features()` (evaluation rows) or `rank_resampled()` (all rows; each split takes its slices). Each is a 1-D array-like of finite, non-negative numbers, one per row, with a positive sum; anything else raises `ValueError` before any model is fit. The weights are copied.

**What is weighted.** With training weights every model is fit with them. A built-in pipeline passes them to its final estimator only (the imputer, scaler and one-hot encoder stay unweighted, as in scikit-learn). The dummy baseline is fit with them too. With evaluation weights, the baseline score, every permuted score and the dummy's score are weighted. With `evidence=True` and training weights, the train score uses them. You can pass either set alone. The report records `sample_weight: {"train": ..., "eval": ...}`; without weights the key is absent.

**Models that cannot take weights.** A model whose final estimator has no `sample_weight` in `fit()` is not fit unweighted, because its importances would answer a different question. It is recorded in `failures` with stage `"fit"` and `error_type` `"SampleWeightNotSupported"`. Among the built-in models these are `lda` and `qda` (classification) and `pls_regression`. The same holds for your own models, bare or as the last step of a `Pipeline`. An estimator whose `fit` takes `**kwargs` counts as weight-capable, and a nested `Pipeline` is followed to its last step.

**Shuffle donors are not weighted.** The donors are the scored rows (after `max_samples`, zero-weight rows included), and weights play no part in who donates. With `shuffle="rows"` each permutation reorders the scored rows at random, so every row donates once whatever its weight. Shadow columns are shuffled copies of every row, also unweighted. Only the scores are weighted. The permuted data therefore follows the unweighted distribution of the evaluation rows, while the baseline is weighted.

> [!NOTE]
> This suits cost or importance weights ("a mistake on these rows costs more"). For survey or frequency weights, where weighted rows stand for a population, a heavy row is no more likely to donate its value than a light one. If the population is your target, drop zero-weight rows, or resample the evaluation rows by weight and pass no `sample_weight` for them, and compare the results. FeatRanker has no option that draws donors by weight.

**Unweighted evidence.** Correlation evidence and the SHAP summary describe the data and the fitted models, not the weighted objective, and stay unweighted.

```python
from sklearn.ensemble import RandomForestRegressor

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

**Selector, pipelines and CLI.** `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)` passes the weights to its resampled ranking. Inside a `Pipeline`, use scikit-learn metadata routing: `sklearn.set_config(enable_metadata_routing=True)`, `ConsensusSelector(...).set_fit_request(sample_weight=True)`, then `pipeline.fit(X, y, sample_weight=w)`. The command line takes `--sample-weight COLUMN`: the column must exist in both CSV files, is read as the weights and is not ranked as a feature.

## Labels and data handling

`FeatureRanker` does not select features or tune models. It changes data values in two places only:

- **Linear scaling.** Built-in linear models sit behind a `StandardScaler` fit on training data only. Tree models and your own models get the data as given.
- **Classification labels.** For `task="clf"`, `fit()` encodes the labels to `0..n_classes-1` with a `LabelEncoder`, because some estimators (XGBoost) require it. Every estimator and scorer, including yours, sees the encoded labels. `ranker.label_encoder.classes_` maps each code back: code `i` is `classes_[i]`.

`fit()`, `rank_features()` and `rank_resampled()` work on their own float copies of `X` and `y`. Changing your arrays afterwards changes nothing in the models, the evidence or later reports, even for estimators that store their training data (such as k-nearest neighbours). The cost is one extra copy per call.

A custom scorer must therefore refer to classes by code. To score F1 for the original class `"case"`:

```python
from sklearn.metrics import f1_score, make_scorer

labels = np.where(x1 + 0.5 * rng.normal(size=n) > 0, "case", "control")
clf = FeatureRanker(task="clf", group="fast", verbose=False)
clf.fit(X[:400], labels[:400], feature_names=names)
case_code = list(clf.label_encoder.classes_).index("case")
scorer = make_scorer(f1_score, pos_label=case_code)
report = clf.rank_features(X[400:], labels[400:], scoring=scorer, n_repeats=3, random_state=0)
print(list(clf.label_encoder.classes_), case_code, report["consensus"][0]["feature_group"])
```

```text
['case', 'control'] 0 x1
```
