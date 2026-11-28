# 性能

[English](../../guide/performance.md) · 简体中文

本页介绍如何让运行更快并控制内存：更小的模型集、更少的评估行、训练子样本、独热编码上限以及并行模型。

[手册](../README.md) | [模型与评分](models-and-scoring.md) | [自定义模型](custom-models.md) | [参考](../reference.md)

## 时间花在哪里

排名时间等于模型数 x 特征组数 x 重复次数 x (1 + 影子特征) x 被评分的行数。下面的每个选项都削减其中一个因子。

| 要削减 | 使用 |
|---|---|
| 模型 | `group="fast"`（每个模型族一个模型，约为模型总数的三分之一） |
| 特征组 | 你自己的 `feature_groups`，或 [`auto_groups`](selection.md#用-auto_groups-为相关特征分组) |
| 重复次数 | 更小的 `n_repeats` |
| 评估行 | `rank_features(max_samples=)` |
| 训练行 | `fit(max_samples=)` |
| 独热编码宽度 | `max_categories` |
| 墙钟时间 | `n_jobs`，参见[并行模型](custom-models.md#并行模型) |

> [!NOTE]
> 回归中核模型族的唯一模型 `svr_regression`，所需时间与行数的平方成正比。在大数据上，请改为传入你自己的 `models=`（`group="fast"` 不能与 `models=` 同时使用）。

## 更少的评估行：max_samples

置换重要性对每个模型、每个特征组、每次重复都要在所有评估行上预测一遍，因此在大型评估集上，这些时间大多换不来多少精度。`max_samples` 改为在一个随机子样本上评分。整数表示行数（超过总行数则使用全部）；区间 (0, 1] 内的浮点数表示比例（`int(max_samples * n)`，向下取整，至少为 1：200 行的 0.29 是 57）。其他值会在评分之前抛出 `ValueError`。

- 哑基线（dummy baseline）门槛、每个基线和每个置换后的得分使用同一个子样本，它用 `random_state` 无放回抽取，因此它们保持可比。
- 训练得分和证据不变：SHAP 值和影子特征仍使用每一行，`overfit_gap` 是训练得分减去模型在子样本上的得分。
- 使用 `sample_weight` 时，权重跟随其所在的行。如果抽到的行权重全为零，`rank_features()` 会抛出点明 `max_samples` 和 `sample_weight` 的 `ValueError`（在 `rank_resampled()` 中则是该划分失败）。
- 报告会增加 `max_samples` 和 `n_eval_rows`。`run["data_sha256"]` 仍然是你传入的所有评估行的指纹。
- 它在 `rank_resampled()`（每个划分的评估行）、`ConsensusSelector(max_samples=...)` 和命令行（`--max-samples`）中的用法相同。

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

只用一半大小的样本，就已经把 `bp` 挤出了前三，因此在用行数换速度之前，请先判断排名的稳定性。

> [!WARNING]
> 抽样是单纯的随机抽样，不分层。较小的样本可能漏掉稀有类别，这时 `roc_auc` 之类的评分器就可能失败。如果所有模型都失败，`rank_features()` 会抛出列出各项失败的 `RuntimeError`。请让 `max_samples` 大到足以包含每个类别；没有分层抽样选项。

## 大数据

时间和内存在不同的地方增长，每个选项只约束其中之一。下面的数字是在一台 16 GB 笔记本上测得的（Python 3.12，scikit-learn 1.5.1）；原始数据行见 [`spikes/memory/FINDINGS.md`](https://github.com/CountZero-Error/Featranker/blob/main/spikes/memory/FINDINGS.md)。

| 选项 | 约束 | 不约束 |
|---|---|---|
| `rank_features(max_samples=)` | 被评分的行：每个模型的预测和被评分矩阵的临时副本。在 100k x 500、一个低开销模型上，峰值 RSS 从 2.7 GB 降到 2.0 GB。 | 拟合、影子块、证据，或所有评估行的已校验副本。 |
| `fit(max_samples=)` | 模型拟合所用的行，因此包括树模型的拟合时间，以及训练影子特征和证据副本。在 100k x 500、使用七个 `"fast"` 模型时，用 5000 行拟合耗时 11 秒。 | 传入的每一行的那份已校验副本，以及评估一侧。仅用 `fit(max_samples=5000)` 时，运行峰值为 3.3 GB；同时再加上 `rank_features(max_samples=5000)`，则为 1.9 GB。 |
| `max_categories` | 内置线性模型的独热编码宽度：每个类别列最多这么多列。对大多数内置线性模型，独热块是稀疏的（10k 行、一个 1000 水平的列，Ridge 和 Lasso：稀疏峰值 1.4 MB，对比稠密 193 MB）。在 4000 行、10 个各有 1000 个水平的类别列且使用稠密输出时，峰值 RSS 从 3.1 GB 降到 0.37 GB（拟合从 46 秒降到 2.7 秒）；这仍适用于六个只支持稠密输入的估计器。 | 树模型和自定义模型，它们本来就是每列使用一个整数编码。不常见的水平共用一列，因此线性模型的得分可能与不设上限时不同。 |
| `n_repeats`、`feature_groups` | 排名时间。`feature_groups`（你自己的，或 `"auto"`）会减少特征组的数量。 | 内存。 |

影子块是原地构建的（峰值为矩阵的 2 倍，而不是 3 倍：在 100k x 500 时从 2.5 GB 降到 2.1 GB）。`evidence=True` 和 `auto_groups()` 背后的 Spearman 矩阵一次只对一块列做排序，因此在 500 列时其峰值从约 7 倍矩阵降到 1.2 倍；100k x 500 上的 `auto_groups` 峰值为 1.8 GB。模型自身的临时内存，例如 `GaussianNB` 在 1000 个加倍后的影子列上做预测，不属于本软件包，不会被降低。

### 训练子样本

`fit(max_samples=)` 接受整数或 (0, 1] 内的浮点数，与 `rank_features()` 的相同，并用它自己的 `random_state` 抽取行（与评估抽样是不同的随机流；`None` 表示使用新的熵）。它保留原有的行顺序，在 `shuffle="within_groups"` 下则抽取整个组，使每个被抽到的组仍可被打乱。

- 如果被抽到的行的 `sample_weight` 值全为零，它会抛出 `ValueError`。
- 抽样是单纯的随机抽样。如果它漏掉了分类目标的某个类别（每个模型都会失败），`fit()` 会抛出点明该类别的 `ValueError`，因此请调大 `max_samples`。
- `run["n_train_rows"]`、`run["train_max_samples"]`、`run["train_random_state"]` 和 `run["train_rows_sha256"]` 标识这份样本。`run["train_data_sha256"]` 仍然是完整数据的指纹，因此 [`ablate()`](selection.md#消融曲线) 接受该报告，并在同样被抽样的行上重新拟合。在其他行上拟合的排名器（另一个 `random_state` 或 `max_samples`）会被拒绝，并抛出 `ValueError`。
- `rank_resampled(max_train_samples=)`、`ConsensusSelector(max_train_samples=)` 和 `--max-train-samples` 把它应用到每个划分的训练行上，使用该划分的随机种子。
- `max_categories`（`ConsensusSelector(max_categories=)`、`--max-categories`）记录在 `run["max_categories"]` 中。它是不小于 2 且小于 2**63 的整数。只有水平数不少于 `max_categories` 的列，训练中未见过的水平才会共用不常见水平那一列；否则它是全零，与不设上限时相同。

下面这段示意需要一个包含 `X_train`、`X_eval` 和类别目标的大数据集，因此只展示而不运行：

```text
ranker = FeatureRanker(task="clf", group="fast", categorical="auto", max_categories=20)
ranker.fit(X_train, y_train, max_samples=50_000, random_state=0)
report = ranker.rank_features(X_eval, y_eval, scoring="roc_auc", max_samples=10_000, random_state=0)
print(report["run"]["n_train_rows"], report["run"]["max_categories"])  # rows fitted on, level cap
```

## 综合运用

对于大规模运行，请组合使用 `group="fast"`、`max_samples`、更小的 `n_repeats`、你自己的 `feature_groups` 和 [`n_jobs`](custom-models.md#并行模型)。请先在更小的问题上用 [`rank_resampled()`](validation.md#重采样排名) 检查排名的稳定性。
