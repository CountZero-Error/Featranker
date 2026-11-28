# Getting started

English · [简体中文](zh-CN/getting-started.md)

This page installs FeatRanker, ranks the features of a small data set, and shows you how to read the result.

[Manual](README.md) | [Concepts](concepts.md) | [Reference](reference.md) | [Troubleshooting](troubleshooting.md)

## Install

```bash
pip install featranker
```

FeatRanker needs Python 3.10 or newer and scikit-learn 1.5 or newer. Extras add optional parts:

| Extra | Adds |
|---|---|
| `featranker[xgboost]`, `featranker[catboost]` | XGBoost and CatBoost models in the built-in set |
| `featranker[all]` | Both of the above |
| `featranker[plot]` | matplotlib, for `report.plot()` |
| `featranker[shap]` | SHAP summaries for `evidence=True` (shap 0.49 or newer; not part of `[all]`) |

The `lightgbm` extra was removed in 1.0. Install `lightgbm` yourself if you want LightGBM in your own models.

## Your first ranking

You split the data, `fit()` the models on the training rows, and call `rank_features()` on separate evaluation rows. FeatRanker cannot tell whether you kept the two sets apart, so that part is yours.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(X_eval, y_eval, scoring="r2", n_repeats=5, random_state=42)

for row in report["consensus"][:4]:
    print(f"{row['feature_group']:>4}  median rank {row['median_rank']:.1f}  ({row['n_models']} models)")
```

```text
  s5  median rank 1.0  (4 models)
 bmi  median rank 2.0  (4 models)
  bp  median rank 3.0  (4 models)
  s6  median rank 5.5  (4 models)
```

- `task` is `"clf"` for classes and `"reg"` for numbers.
- `group="fast"` fits one model per family. Use the default `"all"` when you have time.
- NumPy input needs `feature_names`. A DataFrame keeps its column names.
- `scoring` is a scikit-learn scorer name. Higher is better, so error metrics start with `neg_`.

> [!TIP]
> Use `verbose=False` in scripts and tests. Progress bars are on by default.

## Read the result

`report` is a dict. The part you read first is `report["consensus"]`: one row per feature group, best first.

- `median_rank` is the middle of the ranks the voting model families gave the group. 1 is the most important.
- `n_models` is how many models voted. Models that did not beat a model that ignores the features do not vote.

Let `summarize()` run the standard checks for you:

```python
for finding in report.summary():
    print(finding["severity"], finding["code"])

print(report.to_frame()[["feature_group", "median_rank", "mean_rank", "n_families"]].head(3).to_string(index=False))
```

```text
info tied_groups
info tied_groups
info within_noise
feature_group  median_rank  mean_rank  n_families
           s5          1.0       1.25           4
          bmi          2.0       1.75           4
           bp          3.0       4.50           4
```

The findings say that some groups share a rank and that some are within noise, so read the order below the top with care. [Report](report.md#reading-the-report) explains every finding and column.

> [!WARNING]
> A ranking tells you what the fitted models use. It does not tell you what causes the target. Choose features inside your validation folds, never from the final test set. [ConsensusSelector](guide/selection.md#consensusselector-in-a-pipeline) does this for you.

## The command line

```bash
featranker --task reg --train-data train.csv --eval-data eval.csv \
  --target target --scoring neg_mean_absolute_error --output report.json
```

See [Command line](cli.md) for every flag.

## Next steps

| You want to | Read |
|---|---|
| Understand how the ranking is decided | [Concepts](concepts.md) |
| Pick a task, group or scorer | [Models and scoring](guide/models-and-scoring.md) |
| Use your own models | [Custom models](guide/custom-models.md) |
| Handle NaN, text columns or weights | [Data](guide/data.md) |
| Check how stable a ranking is | [Validation](guide/validation.md) |
| Tell signal from noise | [Noise and evidence](guide/noise-and-evidence.md) |
| Choose a subset of features | [Selection](guide/selection.md) |
| Speed up a run | [Performance](guide/performance.md) |
| Look up a parameter | [Reference](reference.md) |
