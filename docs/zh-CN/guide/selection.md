# 特征选择

[English](../../guide/selection.md) · 简体中文

本页介绍如何把排名变成特征子集：管道中的 `ConsensusSelector`、消融曲线、`summarize()`、比较运行，以及用 `auto_groups()` 进行分组置换。

[手册](../README.md) | [验证](validation.md) | [噪声与证据](noise-and-evidence.md) | [报告](../../report.md) | [参考](../reference.md)

## 管道中的 ConsensusSelector

根据全部数据的排名挑选特征，再在这些特征上对模型做交叉验证，会高估它的性能。`ConsensusSelector` 通过在 `fit()` 内部排名来避免这一点：作为 `cross_val_score` 或 `GridSearchCV` 下 `Pipeline` 的一个步骤，它会在每个训练折上重新排名，并且从不接触用来给它打分的行。

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

X, y = load_diabetes(return_X_y=True, as_frame=True)
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=4, group="fast", cv=3, n_repeats=3,
        random_state=0,
    )),
    ("model", Ridge()),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"nested-CV MAE: {-scores.mean():.1f} +/- {scores.std():.1f}")

pipe.fit(X, y)  # final model on all development data
print("selected:", pipe.named_steps["select"].selected_groups_)
```

```text
nested-CV MAE: 50.2 +/- 2.1
selected: ['s5', 'bmi', 'bp', 's3']
```

这个得分是可信的，因为每个外层折都从自己的训练行中选出了自己的四个特征组。最后一行显示的是为最终模型从全部数据中选出的特征组。

至少要设置一条规则：`k`、`min_shadow_share`、`max_shadow_q` 或 `min_shadow_families`。先应用影子过滤，再取通过过滤的前 `k` 个特征组。每个参数见[参考](../reference.md#consensusselector)。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

rng = np.random.default_rng(0)
values = np.hstack([X.to_numpy(), rng.normal(size=(len(X), 2))])  # two pure-noise columns
names = list(X.columns) + ["noise_1", "noise_2"]

selector = ConsensusSelector(
    "reg", "r2", max_shadow_q=0.05, group="fast", n_repeats=3, random_state=0,
    shadow_seed=0, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
)
selector.fit(pd.DataFrame(values, columns=names), y)
print(selector.selected_groups_)
```

```text
['s5', 'bmi']
```

被选中的特征组保留其所有列，并保持原有顺序。特征名称来自 DataFrame 的列；对于数组，名称是 `x0`、`x1`、……，`feature_groups` 必须使用这些名称。`fit()` 之后你可以使用 `selected_groups_`、`report_`（本次拟合的重采样报告）、`n_features_in_`、`feature_names_in_`（DataFrame）以及标准的 `get_support()` 和 `get_feature_names_out()`。

- `report_` 只反映本次 fit 所见的数据。要得到整个开发集的排名，请直接调用 `rank_resampled()`。
- 如果有任何划分失败或没有共识，`fit()` 会警告有多少个划分没有投票。这些划分在各项占比中已经算作未命中；请查看 `report_["failures"]` 和 `report_["splits"]` 了解原因。
- 如果没有特征组通过，`fit()` 会发出警告，选择器不保留任何列，与 scikit-learn 自带的选择器一样。此后大多数下游估计器会因零列而失败。请放宽影子规则，或只使用正的 `k`。
- 使用影子规则时，如果在任何划分中都没有模型保留影子列，`fit()` 会抛出 `ValueError`，而不是在没有影子证据的情况下做出选择。
- 无效的规则会在 `fit()` 中抛出 `ValueError`，而不是在 `__init__` 中，因此 `clone`、`set_params` 和对 `select__k` 的 `GridSearchCV` 都能工作。
- 对于感知分组的内层 `cv`（如 `GroupKFold`），请传入 `fit(X, y, groups=...)`；在 `Pipeline` 中则传入 `select__groups=...`（例如 `cross_val_score(..., params={"select__groups": groups})`）。使用 `shuffle="within_groups"` 时，同样的分组也会限定打乱的范围。
- 在 `fit()` 和 `transform()` 中，DataFrame 的列标签按字符串比较。失败的 `fit()` 会清除之前已拟合的状态。
- 在[真实数据研究](../../realdata.md)的 13 个数据集中，`min_shadow_share=0.5` 在其中一些上只保留了 1% 的特征，并损失了留出得分。`max_shadow_q` 是实验性的，需要不少于 10 个划分的 `cv`；`min_shadow_families` 是一种启发式方法（[噪声与证据](noise-and-evidence.md#模型族证据)）。

> [!NOTE]
> 每次 `fit()` 就是一次 `rank_resampled()` 运行：`cv` 次排名，开启影子特征时翻倍。在 `cross_val_score(cv=n)` 内部，这是 `n x cv` 次排名，拟合最终管道时再加一次运行。`GridSearchCV` 会把开销乘以候选数量，再加上它最后的重新拟合。搜索期间请使用 `group="fast"` 或 `"tree"`、较小的 `cv` 和较少的 `n_repeats`。

## 消融曲线

置换重要性显示，当一个特征组变成噪声时，已拟合的模型会损失多少。它并不表明前几个特征组是否足够：相关的特征组共享功劳，因此每个单独看起来都很小，而把两个都去掉就会受损。`ablate()` 通过重新拟合来检查排名。对于每个 `k`，每个在共识中投过票的模型都会在前 `k` 个特征组的列上重新拟合（沿用 `fit()` 所用的配置，之后对你的估计器调用 `set_params` 没有效果），并在评估数据上评分。这条曲线回答“我需要多少个特征组”：曲线变平的地方，后面的特征组增益很小。

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
names = list(data.feature_names)
X_dev, X_test, y_dev, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.4, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(X_val, y_val, scoring="r2", random_state=42)

ks = [1, 2, 3, 5, 10]  # same refits, scored on validation rows and on untouched test rows
validation = ranker.ablate(X_train, y_train, X_val, y_val, report, ks=ks)
test = ranker.ablate(X_train, y_train, X_test, y_test, report, ks=ks)
for on_val, on_test in zip(validation["ablation"]["curve"], test["ablation"]["curve"]):
    print(f"k={on_val['k']:>2}  validation {on_val['median_score']:.2f}  "
          f"test {on_test['median_score']:.2f}  added {on_val['feature_groups'][-1]}")
```

```text
k= 1  validation 0.26  test 0.04  added bmi
k= 2  validation 0.43  test 0.16  added s5
k= 3  validation 0.41  test 0.17  added bp
k= 5  validation 0.46  test 0.13  added s3
k=10  validation 0.44  test 0.15  added sex
```

两条曲线在相同的训练行上重新拟合相同的模型，使用相同的前 `k` 个特征组；只有被评分的行不同。验证曲线从 `k=1` 时的 0.26 升到 `k=2` 时的 0.43，之后保持在 0.41 到 0.46 之间，因此前两三个特征组承载了大部分信息。测试行在每个 `k` 上的得分都更低，包括没有排除任何特征组的 `k=10`：0.15 对 0.44。这个差距大部分来自两个小划分之间的差异，而不是特征选择。如果你根据验证曲线选了 `k=3`，那么是验证行选出了 `k`，所以 0.41 是一个验证得分。未被触碰的测试行（同样的特征组得分 0.17）才是应该报告的数字。

> [!WARNING]
> 默认情况下，这条曲线是在同时产生排名的评估数据上评分的，因此偏乐观：前几个特征组正是因为在那里得分高才被选中。把第三个数据集作为 `X_eval` 和 `y_eval`，可以为你事先固定的 `k` 消除这种乐观偏差。一旦你用这条曲线来选择 `k`，那个数据集就已被用于选择，它的最佳得分就是一个验证得分。请在另一个未被触碰的测试集上报告最终性能，或者在整个流程外面套一层外层交叉验证（`cross_val_score` 下 `Pipeline` 中的 `ConsensusSelector`）。请在该循环内部选择 `k`，例如把 `GridSearchCV(pipe, {"select__k": [2, 3, 5]})` 作为传给 `cross_val_score` 的估计器。从曲线中选出固定的 `k` 再做交叉验证，结果仍然偏乐观，因为选择它用的是同一批行。

工作方式：

- **特征组与列。** 保留 `report["consensus"]` 的前 `k` 行，按列表顺序。它们的列是各自 `feature_groups` 条目的并集，按训练列的顺序排列。特征组绝不会被拆开。
- **模型。** 只有投票的模型（`in_consensus: true`）会被重新拟合。共识为空会抛出 `ValueError`。
- **相同的预处理。** 每个 `k` 都会构建一个新的 `FeatureRanker`，沿用该排名器的 `task`、`group` 或 `models`、`n_jobs`、`missing`、`model_input` 以及 `categorical` 列（其中被保留的那些），不带影子特征或证据。插补、类别编码、具名输入和样本权重的行为与排名运行时一致。
- **开销。** 每个投票模型重新拟合 `len(ks)` 次。更短的 `ks` 列表可以节省时间。
- **失败。** 在某个 `k` 上拟合或评分失败的模型，在该处得到 `null`，并在 `ablation["failures"]` 中增加一条记录；其他模型和其他 k 继续进行。无法在很少的列上拟合的模型（内置的 `pls_regression` 只有一列时）会在较小的 `k` 上以这种方式失败。`median_score` 是未失败模型的中位数，如果全部失败则为 `null`。
- **配对中位数。** 失败的模型可能随 `k` 而变化，因此即使没有任何模型的得分改变，`median_score` 也可能变化。所以每一行还带有 `n_models_scored` 和 `paired_median_score`，后者是 `paired_models`（在每个 `k` 上都有得分的模型）的中位数（如果没有则为 `null`）。当 `n_models_scored` 变化时，请优先使用 `paired_median_score`。
- **来源。** 该块记录 `data_sha256`（`X_eval` 和 `y_eval` 的[指纹](#比较运行)）、`weights_sha256`（没有权重时为 `null`）、`n_eval_rows` 以及 `same_holdout`：当该数据和权重就是报告排名时所用的数据和权重时为 `true`。`false` 表示不完全相同（行顺序也算在内），并不表示这些行是新的。
- **基线。** `baseline_score` 是重新拟合本身得到的哑基线（dummy baseline）得分（相同的训练行和权重、每个 `X_eval` 行、所用的评分器），因此它与曲线可比。它可能与报告的 `baseline_score` 不同，后者使用的是报告自己的行、评分器和 `max_samples`。只有没有任何重新拟合得到评分时，它才为 `null`。
- **汇总。** `summarize()` 会增加一条 `ablation_curve` 发现，并且当各行中被评分的模型集合不同时，增加一条 `ablation_cohort_changed` 警告。
- **权重。** 排名器不记得权重，因此请再次传入 `sample_weight` 和 `eval_sample_weight`；它们会与报告的 `run["train_weights_sha256"]` 和 `run["weights_sha256"]` 核对。第三个数据集可以使用任何权重。

这条曲线是描述性的。FeatRanker 不会替你选择 `k`，也不会一次去掉一个特征组。重采样报告没有消融；请对留出报告做消融。在 1.22 之前 pickle 的、带有用户模型的排名器，会重新拟合其已拟合模型的克隆，而不是当前的模板。

> [!NOTE]
> `ablate()` 会拒绝与排名器不匹配的报告，并抛出 `ValueError`：重采样报告、空的共识、属于另一个类或参数不同的投票模型、另一份训练数据、另一组权重，或排名器最近一次 `fit()` 没有拟合过的投票模型。在另一个 scikit-learn 版本下生成的报告，会按两个版本共有的参数进行比较，只有不同时才发出警告。参见[参考](../reference.md#输入规则)。

## 汇总报告

`summarize(report)` 替你应用[阅读指南](../../report.md#reading-the-report)。它返回一个发现列表，不会改变报告。它适用于 `rank_features()` 返回的字典、从命令行 JSON 加载的报告，以及重采样报告。

```python
from featranker import summarize

for finding in summarize(report):
    print(finding["severity"], finding["code"])
```

```text
info tied_groups
info tied_groups
info within_noise
```

每个发现都是一个字典，包含 `code`、`severity`（`"info"` 或 `"warning"`）、`message`、可 JSON 序列化的 `evidence`、`basis`（`"descriptive"`；当它依赖影子 p 值或 q 值时为 `"experimental"`，目前只有 `insufficient_splits`）和 `scoring`。代码是公开的，消息不是。发现按以下顺序出现：

| 代码 | 严重程度 | 何时出现 |
|---|---|---|
| `gate_off` | warning | 哑基线失败，因此每个模型都投了票。 |
| `empty_consensus` | warning | 没有模型胜过哑基线。 |
| `failures` | `fit` 和 `ranking` 为 warning；`initialization` 和 `evidence` 为 info | 每个失败阶段一条发现，列出相关模型。 |
| `few_families` | warning | 投票的模型族少于 3 个。 |
| `tied_groups` | info | 若干特征组具有相同的 `median_rank`；每组并列一条发现。它并不衡量排名的不确定性。 |
| `within_noise` | info | 在大多数投票模型中，该特征组的平均重要性不大于它在各次打乱之间的标准差。当 `n_repeats=1` 时，它只能捕捉到重要性为 0 或更小的情形。 |
| `below_shadow` | info | 开启影子特征时，`above_shadow_share` 低于 0.5 的特征组。这是描述性的，不是显著性检验。 |
| `overfit` | info | 开启证据时：在评估数据上损失了至少一半相对哑基线的训练优势的模型，即 `(train_score - evaluation_score) / (train_score - baseline_score) >= 0.5`。当 `baseline_score` 为 `None` 时跳过。 |
| `correlated_groups` | info | 开启证据时：在训练数据上相关的特征组（Spearman rho 的绝对值至少为 0.5）。 |
| `constant_within_groups` | warning | 以 `shuffle="within_groups"` 排名的报告中，有特征组在每个打乱组内为常数；它们的重要性按构造为零。 |
| `ablation_curve` | info | 报告含有带得分的 `ablation` 块：`paired_median_score` 最高的 `k`（并列时取最小者）及其在最大 `k` 处的得分。证据：`best_k`、`best_median_score`、`max_k`、`max_k_median_score`。 |
| `ablation_cohort_changed` | warning | 曲线各行被评分的模型集合不同，因此未配对的 `median_score` 比较的是不同的集合。 |

空列表表示没有什么特别突出的地方，而不是说明排名已被证明。这些检查只覆盖一个评估集。

### 汇总重采样报告

重采样报告有自己的发现，按以下顺序，仅当条件成立时才出现。要阅读某一个划分，请汇总 `report["splits"][i]["report"]`。

| 代码 | 严重程度 | 何时出现 |
|---|---|---|
| `split_failures` | warning | 彻底失败的划分，附带索引和错误类型。它们在每项占比中都算作未命中。 |
| `failures` | 与单个划分相同 | 跨划分汇总的模型失败，每个模型附带它失败的划分。 |
| `gate_off` | warning | `baseline_score` 为 `None` 的划分。 |
| `empty_consensus` | warning | 共识为空的划分。 |
| `below_shadow` | info | 开启影子特征时，稳定性 `above_shadow_share` 低于 0.5 的特征组。 |
| `constant_within_groups` | warning | 在某些划分中，在每个打乱组内为常数的特征组（证据 `groups` 和 `splits`）。 |
| `insufficient_splits` | info | 开启影子特征时，命中检验在这么多划分下无法达到 0.05 的 q 值（`shadow_test.min_p_value > 0.05`，即少于 5 个划分）。 |

```python
ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
resampled = ranker.rank_resampled(
    data.data, data.target, feature_names=names,
    scoring="r2", cv=3, n_repeats=3, random_state=0,
)
for finding in summarize(resampled):
    print(finding["severity"], finding["code"])
```

```text
info below_shadow
info insufficient_splits
```

各划分自己的发现（`few_families`、`tied_groups` 等）不在此重复。

## 比较运行

每份报告都有一个 `run` 块，说明它是由什么计算出来的，因此无需保留数据，就可以区分两份报告。

- `task`、`group`：预设。
- `fingerprint_scheme`：`2`。整数列和布尔列（例如类别标签）按其数据类型种类打上标记后被精确地哈希，因此大于 2**53 的整数会有所不同。布尔值按 0 和 1 哈希，对于相同的值，`int32` 等于 `int64`；数值相同的整数列和浮点列仍然不同。`ablate()` 把没有此键的报告视为方案 1。
- `data_sha256`：对 `rank_features()` 而言是评估数据的指纹，对 `rank_resampled()` 而言是传入的所有数据的指纹。
- `train_data_sha256`：传给 `fit()` 的数据的指纹（仅 `rank_features()` 报告；对于 1.12 之前 pickle 的排名器为 `null`）。
- `weights_sha256`（以及 `train_weights_sha256`）：样本权重的指纹，仅在提供了权重时才有。
- `shuffle_groups_sha256`（以及 `train_shuffle_groups_sha256`）：打乱标签的指纹，仅在提供了标签时才有。
- 仅 `rank_resampled()`：`cv`（给定的整数、去掉内存地址的划分器 `repr`，或 `"iterable of splits"`）和 `splits_sha256`，即每个划分的训练行索引和评估行索引的指纹。
- `auto_groups`：`{"threshold": 0.7, "method": "spearman-complete-linkage"}`，仅当 `feature_groups="auto"` 选择了分组时才有。

指纹是特征名称、特征值和目标值的 SHA-256 摘要。在相同的库版本下，相同的输入给出相同的指纹。改变任何一个值、一个名称或行的顺序都会使它改变。在不同的 NumPy 或 pandas 版本之间，指纹可能不同，尤其是对文本和其他 object 列（版本记录在 `provenance` 中）；NumPy 输入中的类别 `None` 与 pandas 的 NaN 哈希结果不同。指纹不涵盖模型设置。

```python
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)


def rank(X_eval, y_eval):
    ranker = FeatureRanker(task="reg", group="fast", verbose=False)
    ranker.fit(X_train, y_train, feature_names=names)
    return ranker.rank_features(X_eval, y_eval, scoring="r2", n_repeats=3, random_state=0)


first = rank(X_eval, y_eval)
print("same inputs:", first["run"] == rank(X_eval, y_eval)["run"])

changed = y_eval.copy()
changed[0] += 1.0
other = rank(X_eval, changed)
print("evaluation data differs:", first["run"]["data_sha256"] != other["run"]["data_sha256"])
print("training data differs:", first["run"]["train_data_sha256"] != other["run"]["train_data_sha256"])
```

```text
same inputs: True
evaluation data differs: True
training data differs: False
```

要比较两份重采样报告，请先核对 `run["data_sha256"]`、`run["cv"]` 和 `run["splits_sha256"]`：三者都相等时，稳定性表来自相同划分中的相同行。

## 分组置换

相关的列可以用 `feature_groups` 作为一个整体来打乱：

```text
feature_groups = {
    "segment": ["segment_red", "segment_blue", "segment_other"],
    "channel": ["channel_web", "channel_store", "channel_partner"],
}
```

在一次重复中，多列特征组中的每一列都接受相同的行置换。这保留了块内部的关系，同时破坏了它与目标的关系。未列入 `feature_groups` 的特征成为单列特征组。空的特征组、未知的特征、重复归属、重叠以及特征组名称冲突都会抛出错误。

### 用 auto_groups 为相关特征分组

置换重要性会在近似副本之间分摊功劳：打乱两个相关列之一，对能使用另一列的模型几乎没有影响，因此两者看起来都不重要。`auto_groups(X)` 仅根据 `X` 提出分组（它从不接触目标），并返回一个可以作为 `feature_groups` 传入的字典。

```python
from featranker import auto_groups

Xf, _ = load_diabetes(return_X_y=True, as_frame=True)

print(auto_groups(Xf))
print(auto_groups(Xf, threshold=0.4))
print(auto_groups(Xf, threshold=0.9))
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{'bmi+s5': ['bmi', 's5'], 's1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{}
```

- 列按 Spearman 相关性做完全连接聚类，因此组内每一对列的相关性绝对值都至少为 `threshold`（容差为 `1e-12`，这使得完全相同的副本在 `threshold=1` 时也能成组）。降低阈值会得到更大的组。常数列绝不会被分组。
- 请使用足够多的行。行数很少时，样本相关性会被夸大（只有两行时，每个非常数的列对相关性绝对值都是 1），因此几乎不相关的列也可能被分到一组。
- 只返回含两列或更多列的组，以其成员用 `+` 连接命名，按列顺序排列。与现有列或另一个组同名会抛出 `ValueError`。
- `X` 的校验方式与 `fit()` 的输入相同。名称取自 DataFrame，否则取自 `feature_names`，再否则为 `x0`、`x1`、……。使用 `categorical` 时，这些列按文本接受，且绝不会被分组。
- 请在你的训练数据或开发数据上调用它，并把结果传给 `rank_features()` 或 `rank_resampled()`。相关性证据会列出从 0.5 起的特征组对，这显示了处在分界线附近的是什么。
- 相关矩阵和聚类的开销是列数的 O(p 平方)，对几百列来说没有问题。

在你无法自己计算分组的地方，请使用 `"auto"`：管道中的 `ConsensusSelector(feature_groups="auto")`，或命令行中的 `--feature-groups auto`。两者都使用默认阈值 0.7；可以通过 `ConsensusSelector(auto_groups_threshold=...)` 或 `--auto-groups-threshold` 修改。报告会把阈值记录在 `run["auto_groups"]` 中，并且只在分组是自动进行时才记录。在 `cross_val_score` 下，每个外层训练折都计算自己的分组，因此评估行绝不会影响分组。

```python
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=3, group="fast", cv=3, n_repeats=3,
        feature_groups="auto", random_state=0,
    )),
    ("model", Ridge()),
])
pipe.fit(Xf, y)

select = pipe.named_steps["select"]
print({name: cols for name, cols in select.report_["feature_groups"].items() if len(cols) > 1})
print("selected:", select.selected_groups_)
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
selected: ['s5', 'bmi', 's3+s4']
```
