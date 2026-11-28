# 故障排查

[English](../troubleshooting.md) · 简体中文

本页列出常见的错误和警告、它们的含义以及应对办法，后面是常见问题的简短解答。

[手册](README.md) | [快速入门](getting-started.md) | [报告](../report.md) | [参考](reference.md)

## 错误

| 错误或消息 | 原因 | 解决办法 |
|---|---|---|
| `X must contain only finite values; impute or drop missing ...` | 特征中有 NaN 或无穷值。 | 先插补或丢弃它们，或对 NaN 传入 `missing="impute"`。无穷值始终会被拒绝。 |
| `feature_names are required for NumPy input.` | `X` 是 NumPy 数组且未给出名称。 | 传入 `feature_names=[...]`，或传入 DataFrame。 |
| `Evaluation feature names and order must match fit().` | `X_eval` 的列与训练数据不同。 | 使用相同的列，且顺序相同。 |
| `y contains labels not seen during fit(): ...` | 评估集中有训练集没有的类别。 | 对划分做分层，使每个类别都出现在训练数据中。 |
| `Regression targets must be numeric.` | `task="reg"` 但目标是文本。 | 将目标转换为数值，或使用 `task="clf"`。 |
| `n_repeats must be a positive integer.` | `n_repeats` 为 0、负数或不是 `int`。 | 使用 1 或更大的值。 |
| `No models fit successfully.` | 每个模型在 `fit()` 中都失败了。消息会列出每个失败。 | 阅读列出的估计器错误；检查数据和所有自定义估计器。 |
| `No models ranked successfully.` | 每个模型在 `rank_features()` 中都失败了。消息会列出每个失败。 | 检查评分器是否适合任务和模型的输出（[模型与评分](guide/models-and-scoring.md#选择评分器)），并查找非有限的得分。 |
| `call fit() again` (`ValueError`) | 在 `fit()` 之后更改了 `missing`、`model_input`、`max_categories` 或 `shadows` 等选项。 | 再次调用 `fit()`，或新建一个 `FeatureRanker`。 |
| `X must be numeric` | 文本、日期或时间列没有使用 `categorical=`，或是日期列（即使使用了也不行）。 | 对其编码，将文本列列入 `categorical`，或将日期转换为数值或标签（[数据](guide/data.md#类别特征)）。 |
| 使用 `n_jobs` 时的 Pickling 错误 | 自定义估计器或评分器无法发送到工作进程。 | 让它持有的所有对象都可被 joblib 序列化（锁和连接不行），或使用 `n_jobs=None`。 |

## 警告与报告条目

| 你看到的内容 | 含义 | 该怎么做 |
|---|---|---|
| `Skipping model '<name>': ...` | 可选库（XGBoost、CatBoost）缺失或损坏。 | 安装对应的附加项，或忽略；其他模型仍会运行。 |
| `No model beat the dummy baseline, so the consensus is empty.` | 没有模型比忽略特征的模型做得更好。 | 见[阅读报告](../report.md#reading-the-report)：可能是特征无法预测目标，或评估集太小。 |
| `failures` 条目，`error_type` 为 `SampleWeightNotSupported` | 使用权重时，`fit()` 中没有 `sample_weight` 的模型（内置的 `lda`、`qda`、`pls_regression`）不会以无权重方式拟合。 | 忽略它，或将该模型换成支持权重的模型（[数据](guide/data.md#样本权重)）。 |
| `failures` 条目，阶段为 `shadows` 且为 `ShadowColumnsDropped` | 自定义管道（Pipeline）丢弃或隐藏了影子列，因此它被排除在影子证据之外。 | 保留影子列，例如 `remainder="passthrough"`（[自定义模型](guide/custom-models.md#影子特征约定)）。 |
| 关于 `shap` 的 `UserWarning` | `evidence=True` 但 `shap` 缺失、损坏或版本低于 0.49。 | `pip install 'featranker[shap]'`。其余证据仍会计算。 |
| 关于 `constant_within_groups` 的 `UserWarning` | 使用 `shuffle="within_groups"` 时，某些特征组在打乱组内从不变化，因此其重要性按构造为零。 | 配合分组划分器使用 `shuffle="rows"`（[验证](guide/validation.md#within_groups-衡量的是什么)）。 |
| `max_shadow_q` 实验性警告 | 影子 q 值是未校准的参考值。 | 将该过滤器视为启发式方法，并检查留出得分。 |
| `fit()` 警告没有特征组通过 | `ConsensusSelector` 的规则什么都没保留，因此不保留任何列。 | 放宽影子规则，或使用正的 `k`（[特征选择](guide/selection.md#管道中的-consensusselector)）。 |
| `insufficient_splits` 发现 | 划分太少，影子命中检验无法达到 0.05 的 q 值。 | 使用 `RepeatedKFold` 或有 10 个以上划分的划分器。 |

## 常见问题

**我需要单独的评估集吗？** 需要。`fit()` 接收训练行，`rank_features()` 接收不同的行。FeatRanker 无法检测重复使用，因此把它们分开由你负责。

**为什么某个模型出现在报告里却没有参与投票？** 它的得分没有严格优于哑基线（dummy baseline），因此 `in_consensus: False`。见[概念](concepts.md#哑基线门槛)。

**为什么两个相关特征看起来都不重要？** 它们分摊功劳。把它们放进同一个 `feature_groups` 条目，或使用 [`auto_groups()`](guide/selection.md#用-auto_groups-为相关特征分组)。

**为什么评估样本缩小一半会改变前三名？** 样本小时排名有噪声。在用行数换速度之前，先用 [`rank_resampled()`](guide/validation.md#重采样排名) 检查稳定性。

**我能用这个排名为最终模型挑选特征吗？** 只能在验证内部进行。使用[管道中的 `ConsensusSelector`](guide/selection.md#管道中的-consensusselector)，并在没有任何步骤接触过的测试集上报告性能。

**影子 p 值经过校准吗？** 没有。它们是用于给特征组排序的参考值（[噪声与证据](guide/noise-and-evidence.md#影子命中检验)）。

**如何让一次运行可复现？** 使用带种子的估计器、固定的整数 `random_state`，使用影子时再加固定的 `shadow_seed`。见[自定义模型](guide/custom-models.md#并行模型)。

**`to_prompt()` 在哪里？** 它与 `explain()` 一起在 2.0 中被移除。见 [API](../api.md#removed)。
