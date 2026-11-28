# 模型与评分

[English](../../guide/models-and-scoring.md) · 简体中文

本页介绍如何选择任务、内置模型组和评分器，以及如何用多个指标检查同一个排名。

[手册](../README.md) | [概念](../concepts.md) | [模型列表](../../models.md) | [参考](../reference.md)

## 任务与模型组

目标是一组类别时，`task` 为 `"clf"`；目标是数值时，`task` 为 `"reg"`。

`group` 用来选择内置模型。

| `group` | 模型 |
|---|---|
| `"all"`（默认） | 所有内置模型，共识最广。 |
| `"tree"` | 树集成模型，能捕捉非线性效应和交互作用。 |
| `"linear"` | 标准化后的线性、判别分析、朴素贝叶斯和核模型（包括 RBF `SVR`），因此并非纯加性模型。 |
| `"fast"` | 从上述两类中各取每个模型族的一个代表：分类 7 个（`"all"` 中为 22 个），回归 5 个（`"all"` 中为 18 个），全部来自 scikit-learn。 |

[模型列表](../../models.md)中的 **Fast** 列标出了会运行的模型。`group="fast"` 在 `rank_resampled()`、`ConsensusSelector(group="fast")` 和命令行（`--group fast`）中的行为相同，且不能与 `models=` 同时使用。

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
> 某个代表模型可能失败，或得分低于哑基线（dummy baseline），这时它所在的模型族不投票。在 `"all"` 中，同族的其他成员仍可能投票，因此参与投票的模型族可能不同。请检查 `report["models"]` 和 `report["failures"]`。代价是投票更单薄：一个运气不好的模型会让其模型族的排名波动更大。`"all"` 仍是默认值。

## 选择评分器

请选择与问题相匹配的评分器。scikit-learn 的评分器都是越高越好，因此误差类指标带有 `neg_` 前缀。

| 问题 | 合适的评分器名称 | 注意事项 |
|---|---|---|
| 二分类 | `"accuracy"`、`"balanced_accuracy"`、`"f1"`、`"roc_auc"` | `"f1"` 把按排序后的第二个类别视为正类。参见[标签](data.md#标签与数据处理)。 |
| 多分类 | `"accuracy"`、`"balanced_accuracy"`、`"f1_macro"`、`"roc_auc_ovr"` | `"f1"` 和 `"roc_auc"` 只适用于二分类，会让所有模型（包括哑基线）失败。`"roc_auc_ovr"` 需要 `predict_proba`，因此 `linear_svm`、`sgd_classifier`、`ridge_classifier`、`perceptron` 和 `passive_aggressive` 会失败。 |
| 回归 | `"r2"`、`"neg_mean_absolute_error"`、`"neg_root_mean_squared_error"` | |

无法产生评分器所需结果的模型会列在 `failures` 中，其余模型仍会参与排名。如果没有任何模型能做到，`rank_features()` 会抛出异常。

自定义评分器是一个可调用对象 `(estimator, X, y) -> float`，越高越好。使用 `sample_weight` 时，它必须接受 `sample_weight=` 关键字参数；scikit-learn 的评分器名称都支持。基线、打乱后或重要性中出现非有限值，会让该模型的排名失败，这些值绝不会进入成功的报告。

## 多个指标

`scoring` 也可以接受列表或元组，例如 `["accuracy", "neg_log_loss"]`，无需重新拟合即可用多个指标检查同一个排名。

- 第一个评分器为主评分器。哑基线门槛、共识、影子证据、稳定性、`summarize()` 和 `ConsensusSelector` 只使用它，因此排名与单评分器运行一致。
- 其余评分器为次要评分器。在相同的置换副本上，每个模型还会得到各特征组的得分和重要性（`mean`、`std`），可通过 `report["models"][model]["secondary"][scorer]` 读取。次要评分器不产生各自的共识。
- 名称必须唯一。可调用对象以其 `__name__` 命名（若 `__name__` 不是非空字符串，则用其类名）。`make_scorer` 对象以其得分函数命名。若要使用两个 `make_scorer(f1_score, ...)` 评分器，请把其中一个包装成带有不同 `__name__` 的函数。空列表、重复名称或无效条目会在任何模型排名之前抛出异常。
- 如果某个次要评分器对某个模型抛出异常或返回非有限值，该模型仍保持排名，其条目为 `null`，并且 `failures` 中会增加一条 `"evidence"` 记录。
- `max_samples` 对每个评分器使用相同的子样本，`sample_weight` 使用相同的权重。
- 使用相同预测和相同正类标签的评分器，在每个置换副本上共享一次预测，因此额外开销主要是各指标自身的计算。只有 scikit-learn 构建的评分器（`get_scorer` 名称和原样的 `make_scorer` 对象）才共享。它们的子类、普通可调用对象、使用其他 `pos_label` 的评分器，以及（带权重时）函数不接受权重的评分器，都会各自单独预测。自定义评分器不得原地修改其预测结果。

它在 `rank_resampled()`、`ConsensusSelector(scoring=[...])` 和命令行（`--scoring accuracy,neg_log_loss`）中的用法相同。

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

共识就是 `r2` 的排名。MAE 列显示每个模型在第二个指标下是否仍会把同一个特征组排在第一。存储的字段参见[报告](../../report.md#several-scorers)。
