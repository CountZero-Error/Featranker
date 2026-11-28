# Models and scoring

English · [简体中文](../zh-CN/guide/models-and-scoring.md)

This page covers how to choose the task, the built-in model group and the scorer, and how to check a ranking under several metrics.

[Manual](../README.md) | [Concepts](../concepts.md) | [Models list](../models.md) | [Reference](../reference.md)

## Task and group

`task` is `"clf"` when the target is a set of classes and `"reg"` when it is a number.

`group` picks the built-in models.

| `group` | Models |
|---|---|
| `"all"` (default) | Every built-in model. The broadest consensus. |
| `"tree"` | Tree ensembles. They capture non-linear effects and interactions. |
| `"linear"` | Standardized linear, discriminant, naive Bayes and kernel models (RBF `SVR` included), so it is not purely additive. |
| `"fast"` | One representative per family from both: 7 for classification (22 in `"all"`) and 5 for regression (18 in `"all"`), all from scikit-learn. |

The **Fast** column of [Models](../models.md) shows which models run. `group="fast"` works the same in `rank_resampled()`, `ConsensusSelector(group="fast")` and the CLI (`--group fast`). It cannot be combined with `models=`.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
names = list(data.feature_names)
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(X_eval, y_eval, scoring="r2", random_state=42)
families = {m["family"] for m in report["models"].values()}
top = [row["feature_group"] for row in report["consensus"][:3]]
print(f"{len(report['models'])} models, {len(families)} families, top 3 {top}")
```

```text
5 models, 5 families, top 3 ['s5', 'bmi', 'bp']
```

> [!NOTE]
> A representative can fail or score below the dummy, and then its family does not vote. In `"all"` another member of that family may still vote, so the voting families can differ. Check `report["models"]` and `report["failures"]`. The trade-off is a thinner vote: one unlucky model moves its family's rank more. `"all"` stays the default.

## Choose a scorer

Pick a scorer that matches the problem. scikit-learn scorers are higher-is-better, so error metrics carry a `neg_` prefix.

| Problem | Good scorer names | Watch out for |
|---|---|---|
| Binary classification | `"accuracy"`, `"balanced_accuracy"`, `"f1"`, `"roc_auc"` | `"f1"` treats the second class in sorted order as positive. See [Labels](data.md#labels-and-data-handling). |
| Multiclass classification | `"accuracy"`, `"balanced_accuracy"`, `"f1_macro"`, `"roc_auc_ovr"` | `"f1"` and `"roc_auc"` are binary-only and fail every model, the dummy included. `"roc_auc_ovr"` needs `predict_proba`, so `linear_svm`, `sgd_classifier`, `ridge_classifier`, `perceptron` and `passive_aggressive` fail. |
| Regression | `"r2"`, `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"` | |

A model that cannot produce what the scorer needs is listed in `failures`, and the others still rank. If no model can, `rank_features()` raises.

A custom scorer is a callable `(estimator, X, y) -> float`, higher is better. With `sample_weight` it must accept a `sample_weight=` keyword; scikit-learn scorer names do. Non-finite baseline, shuffled or importance values fail that model's ranking; they never enter a successful report.

## Several metrics

`scoring` also takes a list or tuple, such as `["accuracy", "neg_log_loss"]`, to check a ranking under more than one metric without refitting.

- The first scorer is primary. The dummy gate, consensus, shadow evidence, stability, `summarize()` and `ConsensusSelector` use it alone, so the ranking equals a single-scorer run.
- Every other scorer is secondary. On the same permuted copies each model also gets its score and its importance per group (`mean`, `std`). Read them at `report["models"][model]["secondary"][scorer]`. There is no consensus per secondary scorer.
- Names must be unique. A callable is named by its `__name__` (its class name if that is not a non-empty string). A `make_scorer` object is named by its score function. To use two `make_scorer(f1_score, ...)` scorers, wrap one in a function with a distinct `__name__`. An empty list, a repeated name or an invalid entry raises before any model is ranked.
- If a secondary scorer raises or returns a non-finite value for a model, that model stays ranked, its entry is `null`, and `failures` gets an `"evidence"` entry.
- `max_samples` applies the same subsample, and `sample_weight` the same weights, to every scorer.
- Scorers that use the same prediction and the same positive label share one prediction per permuted copy, so the extra cost is mostly each metric's own computation. Only scorers scikit-learn builds (`get_scorer` names and exact `make_scorer` objects) share. A subclass of those, a plain callable, a scorer with another `pos_label`, and (with weights) a scorer whose function takes no weights predict on their own. A custom scorer must not modify its predictions in place.

It works the same in `rank_resampled()`, `ConsensusSelector(scoring=[...])` and the CLI (`--scoring accuracy,neg_log_loss`).

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

The consensus is the `r2` ranking. The MAE columns show whether each model would put the same group first under the second metric. See [Report](../report.md#several-scorers) for the stored fields.
