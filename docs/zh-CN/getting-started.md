# 快速入门

[English](../getting-started.md) · 简体中文

本页说明如何安装 FeatRanker，对一份小数据集的特征排名，并教你阅读结果。

[手册](README.md) | [概念](concepts.md) | [参考](reference.md) | [故障排查](troubleshooting.md)

## 安装

```bash
pip install featranker
```

FeatRanker 需要 Python 3.10 或更高版本，以及 scikit-learn 1.5 或更高版本。附加项（extras）可添加可选部分：

| 附加项 | 添加内容 |
|---|---|
| `featranker[xgboost]`, `featranker[catboost]` | 内置模型集中的 XGBoost 和 CatBoost 模型 |
| `featranker[all]` | 以上两者 |
| `featranker[plot]` | matplotlib，用于 `report.plot()` |
| `featranker[shap]` | 用于 `evidence=True` 的 SHAP 摘要（shap 0.49 或更高版本；不包含在 `[all]` 中） |

`lightgbm` 附加项已在 1.0 中移除。如果想在自己的模型中使用 LightGBM，请自行安装 `lightgbm`。

## 第一次排名

先划分数据，用训练数据 `fit()` 模型，再在独立的评估集上调用 `rank_features()`。FeatRanker 无法判断你是否把两个数据集分开了，这一点由你负责。

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

- `task` 为 `"clf"` 表示分类，`"reg"` 表示回归。
- `group="fast"` 每个模型族只拟合一个模型。时间充裕时使用默认值 `"all"`。
- NumPy 输入需要 `feature_names`。DataFrame 会保留列名。
- `scoring` 是 scikit-learn 评分器名称。越高越好，因此误差类指标以 `neg_` 开头。

> [!TIP]
> 在脚本和测试中使用 `verbose=False`。进度条默认开启。

## 阅读结果

`report` 是一个字典。首先要看的是 `report["consensus"]`：每个特征组一行，最好的在前。

- `median_rank` 是参与投票的各模型族给该特征组的排名的中位数。1 表示最重要。
- `n_models` 是参与投票的模型数。没有胜过忽略特征的模型的模型不参与投票。

让 `summarize()` 替你运行标准检查：

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

这些发现表明有些特征组并列，有些在噪声范围内，因此对榜首以下的顺序要谨慎解读。[报告](../report.md#reading-the-report)解释了每个发现和每一列。

> [!WARNING]
> 排名只说明已拟合的模型在使用什么，不说明什么导致了目标。请在验证折内部选择特征，不要用最终测试集选择。[ConsensusSelector](guide/selection.md#管道中的-consensusselector) 会替你做到这一点。

## 命令行

```bash
featranker --task reg --train-data train.csv --eval-data eval.csv \
  --target target --scoring neg_mean_absolute_error --output report.json
```

全部参数见[命令行](cli.md)。

## 下一步

| 你想要 | 阅读 |
|---|---|
| 理解排名如何确定 | [概念](concepts.md) |
| 选择任务、模型组或评分器 | [模型与评分](guide/models-and-scoring.md) |
| 使用自己的模型 | [自定义模型](guide/custom-models.md) |
| 处理 NaN、文本列或权重 | [数据](guide/data.md) |
| 检查排名有多稳定 | [验证](guide/validation.md) |
| 区分信号与噪声 | [噪声与证据](guide/noise-and-evidence.md) |
| 选择特征子集 | [特征选择](guide/selection.md) |
| 加快运行 | [性能](guide/performance.md) |
| 查找参数 | [参考](reference.md) |
