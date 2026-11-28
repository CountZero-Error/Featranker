# 命令行

[English](../cli.md) · 简体中文

本页介绍 `featranker` 命令：它读取什么、全部参数以及示例。

[手册](README.md) | [快速入门](getting-started.md) | [报告](../report.md) | [参考](reference.md)

## 快速示例

该命令接收分开的训练和评估 CSV 文件，并将报告写成 JSON。

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

运行 `featranker --help` 查看参数列表。

## 输入文件

- 两个 CSV 文件都需要表头、顺序相同的相同特征列，以及目标列。
- 特征值必须是数值，除非用 `--categorical` 指定。目标可以是数值或字符串。
- 默认情况下，特征中的空单元格会被拒绝。使用 `--missing impute` 时，它们被读作缺失值，并在每个训练划分内插补（[数据](guide/data.md#缺失值)）。只有空单元格和文本 `nan`（不区分大小写）被读作缺失；`NA`、`N/A` 和 `null` 不会，并会引发数值错误。

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.2,8.1,1,0,12.4
2.1,7.3,0,1,14.8
3.4,6.8,1,0,18.2
4.0,5.9,0,1,20.1
```

这是 `train.csv`。`eval.csv` 的列相同：

```csv
feature_a,feature_b,segment_red,segment_blue,target
1.7,7.8,1,0,13.6
3.8,6.2,0,1,19.4
```

`groups.yaml` 将特征组名称映射到其列：

```yaml
segment:
  - segment_red
  - segment_blue
```

## 必需参数

| 参数 | 含义 |
|---|---|
| `--task` | `clf` 表示分类，`reg` 表示回归。 |
| `--train-data` | 含训练特征和目标的 CSV。 |
| `--eval-data` | 含留出评估特征和目标的 CSV。 |
| `--target` | 两个文件中目标列的名称。 |
| `--scoring` | 一个 scikit-learn 评分器名称，或用逗号分隔的多个名称（`--scoring accuracy,neg_log_loss`）。第一个为主要指标；见[模型与评分](guide/models-and-scoring.md#多个指标)。 |

## 模型与排名

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--group` | `all` | 内置模型组：`linear`、`tree`、`all` 或 `fast`。 |
| `--n-repeats` | `10` | 每个特征组的置换重复次数。 |
| `--random-state` | 无 | 可复现置换的随机种子。同时也为 `--max-train-samples` 的抽样设定种子。 |
| `--n-jobs` | 顺序执行 | 在模型之间并行的工作进程数；`-1` 使用全部核心。 |
| `--evidence` | 关 | 向报告添加[证据](guide/noise-and-evidence.md#证据)：训练得分、过拟合差距、相关特征组，以及安装了 `shap` 时的 SHAP。 |
| `--quiet` | 关 | 隐藏进度条。警告和错误仍会显示。 |
| `--output` | stdout | 将 JSON 报告写入此路径。缺少扩展名时会追加 `.json`。已有的报告在新报告完成后才被替换（仅当其所在文件夹为只读时才原地写入），并保留其所有者、组、权限模式和访问规则；在 Linux 和 macOS 上，若对其没有写权限则会被拒绝。新报告使用你的默认权限（umask）。 |

## 特征组

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--feature-groups` | 无 | 一个 `group: [columns]` 形式的 YAML 文件，或者单词 `auto`，用 [`auto_groups()`](guide/selection.md#用-auto_groups-为相关特征分组) 将相关的训练列分组。名为 `auto` 的文件必须写成 `./auto`。 |
| `--auto-groups-threshold` | `0.7` | 配合 `--feature-groups auto`，将斯皮尔曼相关系数绝对值不低于此值的列分为一组。 |

## 数据选项

| 参数 | 含义 |
|---|---|
| `--missing` | `error`（默认）拒绝空的特征单元格。`impute` 为每个内置模型提供仅在训练数据上拟合的中位数插补器。 |
| `--categorical` | `auto` 将训练 CSV 中含有非数值单元格的列读作文本（一个多余的 `NA` 会使整列变为类别特征；名为 `auto` 的列无法列出）。逗号分隔的列表（如 `city,segment`）将这些列读作文本。水平（level）仅来自训练 CSV；此类列中的空单元格是一个缺失水平。见[类别特征](guide/data.md#类别特征)。 |
| `--sample-weight` | 两个文件中都有的一列，用作拟合和评分的样本权重，不作为特征参与排名（[样本权重](guide/data.md#样本权重)）。 |
| `--drop` | 在读取任何特征之前，从两个文件中移除的逗号分隔列，如 `id,created_at`，使它们不会成为特征（被丢弃的文本列也不会被读作类别特征）。未知名称、目标、`--sample-weight` 列或 `--categorical` 列会引发错误，丢弃全部特征也会。 |

## 大数据

| 参数 | 含义 |
|---|---|
| `--max-samples` | 在评估行的子样本上评分：整数（行数），如 `5000`，或带小数点、位于 (0, 1] 的浮点数（比例），如 `0.25`。 |
| `--max-train-samples` | 在训练行的子样本上拟合模型，形式相同。使用 `--random-state` 抽取。 |
| `--max-categories` | 至少为 2。线性模型中每个类别列最多保留这么多个独热编码列；较少见的水平共用一列。 |

各参数限制的是什么，见[性能](guide/performance.md#大数据)。

## 消融曲线

| 参数 | 含义 |
|---|---|
| `--ablate` | 向写出的报告添加[`ablation` 块](guide/selection.md#消融曲线)。它为每个 `k` 重新拟合投票的模型，因此运行时间更长。 |
| `--ablate-ks` | 配合 `--ablate`，逗号分隔的正整数，如 `1,2,5`。默认值：1、2、3、5、10，依此类推。大于特征组数量的 `k` 会在拟合任何模型之前引发错误。 |

> [!WARNING]
> 命令行的曲线是在选出特征组的那份评估 CSV 上评分的，因此偏乐观。见[消融曲线](guide/selection.md#消融曲线)中的警告。如果没有模型胜过哑基线（dummy baseline），就没有共识：`--ablate` 会向 stderr 输出警告，写出的报告不含 `ablation` 块。

## 限制

命令行保持数组输入，并始终自由地打乱行（`shuffle="rows"`），因为它没有分组列。对于时间序列、分组行、你自己的模型或 `rank_resampled()`，请使用 Python API。
