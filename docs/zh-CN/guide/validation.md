# 验证

[English](../../guide/validation.md) · 简体中文

本页介绍如何检查排名有多稳定、如何对按时间排序的行和分组的行进行排名，以及如何用嵌套验证保证特征选择的可信。

[手册](../README.md) | [数据](data.md) | [噪声与证据](noise-and-evidence.md) | [选择](selection.md) | [参考](../reference.md)

## 重采样排名

一次划分只给出一个排名。`rank_resampled()` 在你传入数据的每个交叉验证划分上拟合并排名，然后报告每个特征组的名次在划分之间变动了多少。当你想知道排名是否稳定时，请使用它。

```python
from sklearn.datasets import load_diabetes

from featranker import FeatureRanker

data = load_diabetes()
ranker = FeatureRanker(task="reg", group="fast", verbose=False)
report = ranker.rank_resampled(
    data.data,
    data.target,
    feature_names=list(data.feature_names),
    scoring="neg_mean_absolute_error",
    cv=5,
    n_repeats=3,
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
  s5  median place 1  range 1-1  top 3 in 100% of splits
 bmi  median place 2  range 1-2  top 3 in 100% of splits
  bp  median place 3  range 3-8  top 3 in 60% of splits
 sex  median place 4  range 3-9  top 3 in 20% of splits
```

特征组在某个划分中的名次，是它在该划分已排序的共识中的位置。每个稳定性行给出名次中位数、各划分间的范围，以及 `top_k_share`，即该特征组进入前 `top_k` 的划分所占的比例。在 `median_rank` 和 `mean_rank` 上都并列的特征组共享较好的名次，因此一个划分中的前列特征组可能多于 `top_k` 个。

- `cv` 可以是折数（会打乱；`task="clf"` 时为分层），可以是任何 scikit-learn 划分器（如 `RepeatedKFold` 或 `GroupKFold`），也可以是 `(train_indices, eval_indices)` 对的列表。两个训练索引和评估索引完全相同的划分会抛出 `ValueError`，因为重复的划分会在影子检验中被当作独立试验。
- 每个划分都使用一个带有此排名器设置的新排名器，因此你调用它的那个排名器不会被改动。
- 完全失败的划分会以 `stage: "split"` 记录在 `failures` 中，运行继续。只有没有任何划分能排名时才会抛出异常。
- 无效的 `cv`、`top_k`、评分器、`n_repeats` 或特征组，会在任何模型拟合之前抛出异常。
- `sample_weight`、`max_samples` 和 `max_train_samples` 按划分应用。第 `i` 个划分使用随机种子 `(random_state + i)` 对 2**64 取模。
- 输出参见[报告](../../report.md#resampled-reports)。

> [!WARNING]
> 请只传入开发数据，把最终测试集留在外面。这里的划分属于内部验证。根据这些结果选出特征，再在同一份数据上给模型打分，会高估它的性能。请在每个训练划分内部拟合学习得到的预处理：把它放进自定义模型的管道中，使用 `missing="impute"`（[数据](data.md#缺失值)）和 `categorical=`（[数据](data.md#类别特征)），或使用手动划分的工作流。不要在这次调用之前就在全部开发行上拟合它。

> [!NOTE]
> 一次调用的开销约为 `n_splits` 倍的一次普通 `fit()` 加 `rank_features()`。在 `group="all"` 且 `cv=10` 时，就是十次完整运行。请从 `cv=5`、`group="fast"` 或更少的 `n_repeats` 开始。

## 时间序列与分组数据

对于有顺序或成簇的行，请传入与模型实际使用方式相匹配的划分器。`TimeSeriesSplit` 用于按时间顺序排列的行，带 `groups=` 的 `GroupKFold` 用于属于同一整体的行，例如每位患者有多行。这样，模型就不会在早于其训练行或与训练行同组的行上被评估。整数 `cv` 搭配 `groups` 并不会按组划分：它使用打乱的 `KFold` 或 `StratifiedKFold`，并发出提及 `GroupKFold` 的 `UserWarning`，折不变。

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

rng = np.random.default_rng(0)
n_rows = 300
X = rng.normal(size=(n_rows, 4))
y = 2.0 * X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=n_rows)
names = ["strong", "weak", "noise_1", "noise_2"]
patients = np.repeat(np.arange(30), 10)  # 10 rows per patient

ordered = FeatureRanker(task="reg", group="fast", shuffle="circular", verbose=False)  # see "Shuffling dependent rows"
grouped = FeatureRanker(task="reg", group="fast", shuffle="within_groups", verbose=False)

by_time = ordered.rank_resampled(  # rows in time order: train on the past, evaluate on the future
    X, y, feature_names=names, scoring="r2", cv=TimeSeriesSplit(n_splits=4),
    n_repeats=3, random_state=0,
)
print("time:", [row["feature_group"] for row in by_time["stability"]])

by_patient = grouped.rank_resampled(  # rows of one patient never sit on both sides of a split
    X, y, feature_names=names, scoring="r2", cv=GroupKFold(n_splits=5), groups=patients,
    n_repeats=3, random_state=0,
)
print("patient:", [row["feature_group"] for row in by_patient["stability"]])

pipeline = Pipeline([  # the same splitters in a pipeline: selection never sees the rows it is scored on
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

划分器只控制如何划分行。默认情况下，影子特征和置换会自由地打乱行，这忽略了相邻行之间或同一组的行之间的依赖。`shuffle` 选项让两者都遵循这种结构。如果不使用它，在这类数据上请把影子特征和命中检验视为不太可靠。

## 打乱存在依赖的行

`FeatureRanker(shuffle=...)` 和 `ConsensusSelector(shuffle=...)` 设定置换重要性和影子列如何打乱行。

| `shuffle` | 行为 |
|---|---|
| `"rows"`（默认） | 行被自由打乱。 |
| `"within_groups"` | 一列的值只在具有相同组标签的行之间移动。把每行一个标签作为 `shuffle_groups` 传给 `fit()`（影子特征）和 `rank_features()`（置换）。`rank_resampled()` 和 `ConsensusSelector.fit()` 使用它们的 `groups`，并按划分切片。只有一行的组中的行保留其值；如果每个组都只有一行，调用会抛出 `ValueError`。每组行数很少时，不同的打乱方式也很少，因此重要性噪声更大。标签必须可排序：`None`、NaN 和混合类型会抛出 `ValueError`。 |
| `"circular"` | 每次打乱都是按 1 到 n-1 之间的随机偏移量做循环移位，这保留了序列的自相关。它假定各行按时间顺序排列（如 `TimeSeriesSplit` 所做的）；在无序的行上它只是一次移位。`rank_resampled()` 会先把每个划分的行按其原始顺序排好。 |

`"within_groups"` 必须提供 `shuffle_groups`（即使 `shadows=False` 也是如此，在 `fit()` 和 `rank_features()` 中都需要，不过此时 `fit()` 会忽略它）；与其他模式一起使用则是错误。当 `shuffle` 不是 `"rows"` 时，报告会记录它。命令行没有分组列，始终使用 `"rows"`。

联合影子特征共用一个偏移量（或一个组内顺序）；`shadow_mode="independent"` 则为每列各抽取一个。使用 `max_samples` 时，置换只打乱被抽样的行。在 `"within_groups"` 下，子样本由整组构成，一直抽取到达到目标行数，因此可能略微超出（`n_eval_rows` 记录实际行数）。在 `"circular"` 下，被抽样的行保持其顺序，但子采样使序列变稀疏，因此移位不再恰好按整个时间步移动数值。相关性和 SHAP 证据不做打乱，因此不受影响。

对于单次拟合：

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

### within_groups 衡量的是什么

只在同一组的行之间移动数值，检验的是模型在多大程度上依赖组内的变化。它无法检验模型对组间变化的依赖。在每个组内部为常数的预测变量（把患者层面的值重复写在每一行上）不会被这种打乱改变，因此无论它对目标的驱动多强，其重要性按构造就是 0。在一次检查中，有 40 名患者、每人 4 行，用一个线性模型和一个患者层面的 `x`，`y = 3x` 加小噪声，`x` 的重要性在自由打乱下约为 2，在 `"within_groups"` 下恰好为 0.0。

要衡量组层面的效应，请使用 `shuffle="rows"` 配合分组划分器（`GroupKFold`）：打乱是自由的，但同一组的行绝不会同时出现在划分的两侧。

在 `"within_groups"` 下，`rank_features()` 会记录 `constant_within_groups`：已排序的特征组，其每一列在被评分的行上（经过 `max_samples` 之后），在每个打乱组内都只有一个值（缺失值也算一个值）。在其他模式下该键不存在。非空列表还会引发 `UserWarning`。

- `rank_resampled()` 的每个划分报告都带有自己的列表。重采样报告在顶层有一个 `constant_within_groups`，是它们的并集，`summarize()` 会增加一条警告，按特征组列出它在哪些划分中为常数（证据 `splits`）。
- 有一列在变化的特征组不会被列出，即使其他列是常数。只在少数几个组中变化的近似常数预测变量同样不会被标记。该列表只能捕捉到这种精确的情形。

## 嵌套验证

根据全部数据的排名挑选特征，再在这些特征上对模型做交叉验证，会高估它的性能。请改为在每一折的训练部分内部进行排名。

- 把 [`ConsensusSelector`](selection.md#管道中的-consensusselector) 放进 `cross_val_score` 下的 `Pipeline`，它会在每个训练折上重新排名。
- 要诚实地选择 `k`，请把 `GridSearchCV(pipe, {"select__k": [2, 3, 5]})` 放进 `cross_val_score` 内部（嵌套交叉验证）。从[消融曲线](selection.md#消融曲线)中选出固定的 `k` 再做交叉验证，结果仍然偏乐观，因为选择它用的是同一批行。
- 请在整个流程中任何一步都没有接触过的测试集上报告最终性能。
