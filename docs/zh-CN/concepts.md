# 概念

[English](../concepts.md) · 简体中文

本页说明 FeatRanker 如何确定排名，以及报告中每个数字的含义。

[手册](README.md) | [快速入门](getting-started.md) | [报告](../report.md) | [参考](reference.md)

## 一图看懂管道

```text
fit()            train every model and a dummy baseline on the training rows
rank_features()  score every model on the evaluation rows
                 shuffle each feature group n_repeats times and measure the score drop
                 rank the groups inside each model
                 models that beat the dummy vote, one vote per model family
                 ->  median and mean rank consensus
```

## 模型与模型族

FeatRanker 拟合多个模型，而不是一个。它内置 40 个模型（22 个用于分类，18 个用于回归；其中 4 个需要可选的 `xgboost` 或 `catboost` 包），你也可以带上自己的模型（[自定义模型](guide/custom-models.md)）。每个内置模型都属于某个模型族，例如 `logistic`、`bagging` 或 `boosting`；见[模型](../models.md)。

- `group` 选择运行哪些内置模型：`"linear"`、`"tree"`、`"fast"`（每个模型族一个）或 `"all"`。
- 一个模型族只投一票，无论它有多少个模型。四个逻辑回归只算一票。
- 你自己的模型各自是独立的模型族，除非用 `families=` 映射它们。

## 留出行上的置换重要性

对单个模型，特征组的重要性是指将该组打乱后其得分下降了多少：

```text
importance = model_score - model_score_after_shuffling
```

- 得分在评估集的行上计算，绝不在训练行上计算。
- 一个特征组的所有列使用同一个行打乱顺序，因此组内的关系保持不变。
- 未列入 `feature_groups` 的特征自成一组。
- 每组打乱 `n_repeats` 次。重复次数越多，均值越精确，耗时也越长。
- 所有模型使用同一个评分器。

## 哑基线门槛

同时拟合并评分一个忽略特征的哑基线（dummy baseline）模型。只有评估得分严格优于哑基线的模型才投票。

- `baseline_score` 是哑基线的得分，仅用于投票判定。
- 如果没有模型胜过哑基线，`consensus` 为空。可能是特征无法预测目标，也可能是评估集太小。
- 如果哑基线本身失败，则所有模型都投票，报告会注明这一点（`gate_off`）。对这样的共识要格外谨慎。`ConsensusSelector.fit()` 则会在任一划分中无法给哑基线评分时引发 `ValueError`。

## 共识排名

在每个模型内部，特征组按平均重要性排名。某个模型族对一个特征组的排名，是其成员排名的中位数。共识排名综合各模型族的排名。

| 数字 | 含义 |
|---|---|
| `median_rank` | 主要排名：各模型族排名的中位数。1 表示最重要。 |
| `mean_rank` | 用于打破 `median_rank` 的并列。 |
| `rank_std` | 各模型族分歧有多大。它不是抽样不确定性。 |
| `n_families` | 有多少个模型族投票。少于 3 个说明投票偏薄。 |
| `n_models` | 有多少个模型投票。这是覆盖数，不是置信度。 |

排名是相对的：排名只说明打乱该组比打乱其他组伤害更大，并不说明伤害很大。要看幅度，请查看 `models[name]["importance"][group]["mean"]` 和 `["std"]`。完整定义见[报告](../report.md#reading-the-report)。

## 影子特征与证据

排名并不说明某个特征组携带信号。两个可选工具可以补充这一点。

- **影子特征**（`shadows=True`）为每个特征添加一份打乱后的副本。当特征组胜过最强的影子时，它就高于影子。见[噪声与证据](guide/noise-and-evidence.md#影子特征)。
- **证据**（`evidence=True`）添加训练得分、过拟合差距、相关特征组和 SHAP 摘要。见[噪声与证据](guide/noise-and-evidence.md#证据)。

## 排名无法告诉你的事

> [!WARNING]
> 置换重要性描述的是已拟合的模型如何使用某个特征。它不说明改变该特征会发生什么，因此不是因果结论。

- 相关特征会分摊功劳。把它们分组，或使用 [`auto_groups()`](guide/selection.md#用-auto_groups-为相关特征分组)。
- 排名取决于评估样本、评分器和模型。`median_rank` 相近的特征组应视为并列。
- 评估行只有几十行时，排名基本是噪声。用 [`rank_resampled()`](guide/validation.md#重采样排名) 查看它变动多少。
- 在验证折内部选择特征。来自最终测试集的排名绝不能用于选择特征。
