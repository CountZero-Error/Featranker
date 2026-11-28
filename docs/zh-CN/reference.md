# 参考

[English](../reference.md) · 简体中文

本页列出 FeatRanker 的每个公开名称，包括签名、参数、返回值和错误。内容与 `help()` 中的文字相同。

[手册](README.md) | [快速入门](getting-started.md) | [报告](../report.md) | [输入规则](#输入规则)

`featranker.__all__` 中的所有名称都是公开的：`FeatureRanker`、`ConsensusSelector`、`Report`、`auto_groups` 和 `summarize`，另有 `featranker.__version__`。稳定性策略见 [API](../api.md)。

## FeatureRanker

### `FeatureRanker`

```text
FeatureRanker(task, group="all", models=None, n_jobs=None, verbose=True, shadows=False,
              shadow_seed=None, evidence=False, shadow_mode="joint", missing="error",
              categorical=None, model_input="array", shuffle="rows", families=None,
              max_categories=None)
```

通过多个模型的置换重要性对特征组排名。创建对象时不读取任何数据。无效的选项会引发 `ValueError`（类型错误时引发 `TypeError`）。限制的细节见[输入规则](#输入规则)。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `"clf"` or `"reg"` | required | 分类或回归。 |
| `group` | `"all"`, `"linear"`, `"tree"`, `"fast"` | `"all"` | 内置模型组。`"fast"` 每个模型族只运行一个模型。不能与 `models` 同时使用。见[模型](../models.md)。 |
| `models` | dict | `None` | 非空的 `{name: estimator}`，用于替换内置模型。名称须为非空字符串，且不能是 `dummy_baseline`。见[自定义模型](guide/custom-models.md)。 |
| `n_jobs` | int | `None` | 在模型之间并行的工作进程数。`None` 或 `1` 为顺序执行，`-1` 使用全部核心。 |
| `verbose` | bool | `True` | 显示进度条。警告始终显示。 |
| `shadows` | bool | `False` | 同时训练并排名每个特征的打乱副本，以显示哪些特征组胜过噪声。见[影子特征](guide/noise-and-evidence.md#影子特征)。 |
| `shadow_seed` | int | `None` | 影子打乱的随机种子。`None` 使每次运行的影子都不同。 |
| `evidence` | bool | `False` | 保留训练数据，并增加训练得分、过拟合差距、相关特征组，以及（使用 `featranker[shap]` 时）SHAP 摘要。见[证据](guide/noise-and-evidence.md#证据)。 |
| `shadow_mode` | `"joint"` or `"independent"` | `"joint"` | `"joint"` 用同一个共享的行顺序打乱所有影子列，`"independent"` 对每列单独打乱。仅在 `shadows=True` 时使用。 |
| `missing` | `"error"` or `"impute"` | `"error"` | `"error"` 拒绝 NaN。`"impute"` 接受 NaN，并为每个内置模型加上仅用其训练行拟合的中位数插补器；自定义模型收到的 NaN 保持不变。见[缺失值](guide/data.md#缺失值)。 |
| `categorical` | `"auto"` or list of str | `None` | `None` 只接受数值特征。`"auto"` 把 object、string 和 category 类型的 DataFrame 列视为类别特征；列表则指定列名。见[类别特征](guide/data.md#类别特征)。 |
| `model_input` | `"array"` or `"dataframe"` | `"array"` | `"array"` 给模型传入浮点 NumPy 数组。`"dataframe"` 传入带特征名（以及 `shadow__<name>` 列）的 DataFrame。需要 pandas。见[按名称选择列](guide/custom-models.md#按名称选择列)。 |
| `shuffle` | `"rows"`, `"within_groups"`, `"circular"` | `"rows"` | 置换和影子打乱行的方式：自由打乱、只在 `shuffle_groups` 中标签相同的行之间打乱，或循环移位（行按时间顺序排列）。见[打乱存在依赖的行](guide/validation.md#打乱存在依赖的行)。 |
| `families` | dict | `None` | `{model_name: family}`，用于设置自定义或内置模型的共识模型族。同一模型族的模型只投一票；未映射的自定义模型各自独立。见[模型族](guide/custom-models.md#模型族)。 |
| `max_categories` | int | `None` | 至少为 2。内置线性模型对每个类别列最多独热编码这么多个水平，其余水平共用一列。树模型和自定义模型不受影响。见[大数据](guide/performance.md#大数据)。 |

已拟合的属性：`trained_models`（名称到已拟合模型）、`feature_names_`、`label_encoder`（分类任务：`classes_[i]` 是编码 `i` 对应的标签）、`categorical_features_`、`categorical_levels_`、`fit_failures_` 和 `initialization_failures_`（失败列表）。

### `FeatureRanker.fit`

```text
fit(X, y, feature_names=None, sample_weight=None, shuffle_groups=None, max_samples=None,
    random_state=None)
```

在训练数据上拟合每个已配置的模型和哑基线（dummy baseline）。返回排名器自身。数据或选项无效时引发 `ValueError`，没有任何模型拟合成功时引发 `RuntimeError`。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 训练特征，至少一行一列。为数值，外加 `categorical` 指定的列。无穷值会被拒绝；除非 `missing="impute"`，否则 NaN 也会被拒绝。 |
| `y` | array | required | 训练目标，每行一个值。类别标签可以是数字或字符串；回归目标必须是数值。 |
| `feature_names` | list of str | `None` | 唯一且非空的列名。NumPy 输入必须提供；DataFrame 则取自其列名。 |
| `sample_weight` | array | `None` | 有限、非负的权重，每行一个，总和为正。每个模型都带权重拟合；不支持权重的模型会失败。见[样本权重](guide/data.md#样本权重)。 |
| `shuffle_groups` | array | `None` | 每个训练行一个组标签。使用 `shuffle="within_groups"` 时必须提供；与其他 `shuffle` 值同用则报错。 |
| `max_samples` | int or float | `None` | 在随机子样本上拟合：整数（行数）或 (0, 1] 内的浮点数（比例）。影子和证据副本也使用该样本。见[训练子样本](guide/performance.md#训练子样本)。 |
| `random_state` | int | `None` | `max_samples` 抽样的随机种子。`None` 使用新的随机熵。未设 `max_samples` 时不起作用。 |

### `FeatureRanker.rank_features`

```text
rank_features(X_eval, y_eval, *, scoring, feature_groups=None, n_repeats=10, random_state=None,
              sample_weight=None, max_samples=None, shuffle_groups=None)
```

在 `fit()` 没见过的评估数据上对特征组排名。返回 [`Report`](../report.md)。未调用 `fit()` 或所有模型都失败时引发 `RuntimeError`，数据或选项无效时引发 `ValueError`。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X_eval` | array or DataFrame | required | 评估特征，列与 `fit()` 相同且顺序一致。自 `fit()` 以来更改过的选项会引发 `ValueError`（“call fit() again”）；见[输入规则](#输入规则)。 |
| `y_eval` | array | required | 评估目标。类别标签必须在训练中出现过。 |
| `scoring` | str, callable or list | required | 仅限关键字参数。scikit-learn 评分器名称、可调用对象 `(estimator, X, y) -> float`（越高越好），或由它们组成的非空列表；第一个为主评分器。见[评分](guide/models-and-scoring.md)。 |
| `feature_groups` | dict | `None` | `{group_name: [feature, ...]}`，其中的特征一起打乱。未分组的特征单独排名。见[分组置换](guide/selection.md#分组置换)。 |
| `n_repeats` | int | `10` | 每个特征组的打乱次数，为正整数。重复越多，均值越精确，耗时也越长。 |
| `random_state` | int | `None` | 置换打乱的随机种子。要得到可复现的报告，还需使用带随机种子的估计器、确定性的评分器，以及（使用影子时）固定的 `shadow_seed`。 |
| `sample_weight` | array | `None` | 有限、非负的评估权重，每行一个，总和为正。基线、置换和哑基线得分都带权重。 |
| `max_samples` | int or float | `None` | 在评估行的一个随机子样本上评分：整数（行数）或 (0, 1] 内的浮点数（比例）。见[更少的评估行](guide/performance.md#更少的评估行max_samples)。 |
| `shuffle_groups` | array | `None` | 每个评估行一个组标签。使用 `shuffle="within_groups"` 时必须提供；与其他 `shuffle` 值同用则报错。 |

### `FeatureRanker.rank_resampled`

```text
rank_resampled(X, y, *, scoring, cv=5, groups=None, feature_names=None, feature_groups=None,
               n_repeats=10, random_state=None, top_k=5, sample_weight=None, max_samples=None,
               max_train_samples=None)
```

在每个交叉验证划分上拟合并排名，并报告排名的稳定性。每个划分使用一个沿用本排名器设置的新排名器，因此本排名器不会被改变。返回重采样的 [`Report`](../report.md#resampled-reports)。单个划分失败会记录在 `failures` 中；只有没有任何划分能排名时才引发 `RuntimeError`。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 全部开发数据，规则与 `fit()` 相同。每个划分取各自的行。请把最终测试集留在外面。 |
| `y` | array | required | 目标，每行一个值。 |
| `scoring` | str, callable or list | required | 仅限关键字参数。同 `rank_features()`；稳定性使用主评分器。 |
| `cv` | int, splitter or list | `5` | 折数（会打乱；分类任务为分层），或 scikit-learn 划分器（如 `RepeatedKFold` 或 `GroupKFold`），或由 `(train_indices, eval_indices)` 组成的列表。两个相同的划分会引发 `ValueError`。 |
| `groups` | array | `None` | 传给划分器的 `split()`。使用 `shuffle="within_groups"` 时，它们同时也是按划分切片的打乱分组，且必须提供。 |
| `feature_names` | list of str | `None` | 同 `fit()`。 |
| `feature_groups` | dict | `None` | 同 `rank_features()`。 |
| `n_repeats` | int | `10` | 每个划分中每个特征组的打乱次数。 |
| `random_state` | int | `None` | 为整数 `cv` 的打乱设定随机种子。第 `i` 个划分使用随机种子 `(random_state + i)` 对 2**64 取模。 |
| `top_k` | int | `5` | 某特征组在一个划分中的名次不超过 `top_k` 时即为该划分的靠前组；用于 `top_k_share`。并列者共享较好的名次，因此一个划分可能有更多靠前组。 |
| `sample_weight` | array | `None` | `X` 全部行的权重，按划分切片：每个划分用自己那些行的权重训练和评分。 |
| `max_samples` | int or float | `None` | 同 `rank_features()`，应用于每个划分的评估行。 |
| `max_train_samples` | int or float | `None` | 同 `fit()` 的 `max_samples`，应用于每个划分的训练行。报告会增加 `max_train_samples`。 |

### `FeatureRanker.ablate`

```text
ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None,
       eval_sample_weight=None)
```

用留出报告中的投票模型，仅在前 `k` 个特征组上重新拟合并评分，得到一条曲线。返回新的 `Report`：`report` 的深拷贝，外加一个 [`ablation` 块](../report.md#ablation)。`report` 不会被改变。代价是每个投票模型重新拟合 `len(ks)` 次。见[消融曲线](guide/selection.md#消融曲线)。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 本排名器拟合时用的训练数据，会与报告中的 `run["train_data_sha256"]` 核对。 |
| `y` | array | required | 训练目标。 |
| `X_eval` | array or DataFrame | required | 用来给重新拟合的模型评分的数据，列与 `fit()` 相同。每一行都会评分；报告中的 `max_samples` 被忽略。 |
| `y_eval` | array | required | `X_eval` 的目标。 |
| `report` | Report | required | 来自本排名器 `rank_features()` 的留出报告。重采样报告或共识为空的报告会引发 `ValueError`。见[输入规则](#输入规则)。 |
| `ks` | list of int | `None` | 仅限关键字参数。前几个特征组的数量，从 1 到共识特征组的数量。默认值：1、2、3、5、10、20、50、100、200、500、1000 等直到该数量，外加该数量本身。 |
| `scoring` | str or callable | `None` | `None` 使用报告中的评分器名称。报告用可调用对象排名时需要传入。 |
| `sample_weight` | array | `None` | 训练权重；传入 `fit()` 用过的那组。会与报告中的 `run["train_weights_sha256"]` 核对。 |
| `eval_sample_weight` | array | `None` | 评估权重；传入 `rank_features()` 用过的那组。当 `X_eval` 是报告自己的数据时，会与 `run["weights_sha256"]` 核对。 |

## ConsensusSelector

### `ConsensusSelector`

```text
ConsensusSelector(task, scoring, *, k=None, min_shadow_share=None, max_shadow_q=None,
                  min_shadow_families=None, group="all", models=None, feature_groups=None,
                  auto_groups_threshold=0.7, cv=5, n_repeats=10, random_state=None,
                  max_samples=None, shadow_seed=None, shadow_mode="joint", missing="error",
                  categorical=None, model_input="array", shuffle="rows", families=None,
                  max_categories=None, max_train_samples=None, n_jobs=None, verbose=False)
```

scikit-learn 风格的选择器，在 `fit()` 内部排名，并保留规则选出的特征组。可作为 `Pipeline` 的一步。`k`、`min_shadow_share`、`max_shadow_q` 和 `min_shadow_families` 至少要设置一个：先应用影子过滤，再取通过者中的前 `k` 个。无效的规则在 `fit()` 中引发 `ValueError`，而不是在此处。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `"clf"` or `"reg"` | required | 同 `FeatureRanker`。 |
| `scoring` | str, callable or list | required | 同 `rank_features()`。使用列表时，第一个评分器用于选择，其余的记录在 `report_` 中。 |
| `k` | int | `None` | 按稳定性顺序保留前 `k` 个特征组（不足则全部保留）。 |
| `min_shadow_share` | float | `None` | 保留稳定性指标 `above_shadow_share` 不低于该值（0 到 1）的特征组。会开启影子。 |
| `max_shadow_q` | float | `None` | 保留 `shadow_q_value` 不高于该值（大于 0，不超过 1）的特征组。属启发式和实验性功能：`fit()` 会发出警告。会开启影子。请使用 10 个或更多划分的 `cv`。 |
| `min_shadow_families` | int | `None` | 保留至少有这么多个模型族在大多数划分中把它排在其影子之前的特征组。为正整数；属启发式。会开启影子。 |
| `group` | `"all"`, `"linear"`, `"tree"`, `"fast"` | `"all"` | 同 `FeatureRanker`。 |
| `models` | dict | `None` | 同 `FeatureRanker`。 |
| `feature_groups` | dict or `"auto"` | `None` | 同 `rank_resampled()`。`"auto"` 对传给 `fit()` 的数据用 `auto_groups()` 把相关的列分组。其他任何字符串都会引发 `ValueError`。 |
| `auto_groups_threshold` | float | `0.7` | `feature_groups="auto"` 的相关性阈值，大于 0 且不超过 1。`feature_groups` 为其他值时忽略。 |
| `cv` | int, splitter or list | `5` | 同 `rank_resampled()`。 |
| `n_repeats` | int | `10` | 同 `rank_resampled()`。 |
| `random_state` | int | `None` | 同 `rank_resampled()`。 |
| `max_samples` | int or float | `None` | 同 `rank_resampled()`。 |
| `shadow_seed` | int | `None` | 同 `FeatureRanker`。 |
| `shadow_mode` | `"joint"` or `"independent"` | `"joint"` | 同 `FeatureRanker`。 |
| `missing` | `"error"` or `"impute"` | `"error"` | 同 `FeatureRanker`。使用 `"impute"` 时，NaN 可通过 `fit()` 和 `transform()`；请在选择器之后放一个插补器。 |
| `categorical` | `"auto"` or list of str | `None` | 同 `FeatureRanker`。选出的原始列按原样返回；请在选择器之后放一个编码器。 |
| `model_input` | `"array"` or `"dataframe"` | `"array"` | 同 `FeatureRanker`。 |
| `shuffle` | `"rows"`, `"within_groups"`, `"circular"` | `"rows"` | 同 `FeatureRanker`。使用 `"within_groups"` 时，必须向 `fit()` 提供 `groups`。 |
| `families` | dict | `None` | 同 `FeatureRanker`。 |
| `max_categories` | int | `None` | 同 `FeatureRanker`。 |
| `max_train_samples` | int or float | `None` | 同 `rank_resampled()`：每个划分的模型在其训练行的子样本上拟合。 |
| `n_jobs` | int | `None` | 同 `FeatureRanker`。 |
| `verbose` | bool | `False` | 显示进度条。默认关闭，因为管道（Pipeline）经常拟合。 |

已拟合的属性：`selected_groups_`、`report_`（本次拟合的重采样报告）、`n_features_in_`、`feature_names_in_`（DataFrame 时），以及标准的 `get_support()` 和 `get_feature_names_out()`。

### `ConsensusSelector.fit`

```text
fit(X, y, groups=None, sample_weight=None)
```

仅在 `X` 和 `y` 上排名（重采样）并保留选出的特征组。返回选择器。失败的 `fit()` 会清除先前已拟合的状态。规则或选项无效，或设置了影子规则而每个模型都丢弃了影子列，或任一划分中无法给哑基线评分时，引发 `ValueError`。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 训练特征。名称取自 DataFrame 的列；数组则为 `x0`、`x1` 等。 |
| `y` | array | required | 训练目标。 |
| `groups` | array | `None` | 供 `GroupKFold` 等分组感知 `cv` 使用的行分组。在管道中请传 `select__groups=...`。使用 `shuffle="within_groups"` 时，它们也限定打乱的范围。 |
| `sample_weight` | array | `None` | 行权重，同 `rank_resampled()`。在管道中请启用元数据路由，并对选择器调用 `set_fit_request(sample_weight=True)`。 |

### `ConsensusSelector.transform`

```text
transform(X)
```

按原来的顺序保留选出特征组的列。若没有选出任何特征组，则返回零列（此时 `fit()` 已发出警告）。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 列与 `fit()` 相同的数据。DataFrame 的列标签按字符串比较。 |

## Report

`Report` 是 `dict` 的子类：索引、`json.dumps`、pickle 序列化和字典相等性都与普通字典一样。各键的说明见[报告](../report.md)。它的三个辅助方法只在调用时才导入 pandas 或 matplotlib。

### `Report.summary`

```text
summary()
```

返回此报告的 [`summarize()`](#summarize) 发现，适用于单次划分报告和重采样报告。不接受任何参数。

### `Report.to_frame`

```text
to_frame()
```

以 pandas DataFrame 的形式返回共识表（重采样报告则为稳定性表），每个特征组一行。嵌套的各模型族和各划分的排名列不包含在内。没有 pandas 时引发 `ImportError`。不接受任何参数。

### `Report.plot`

```text
plot(ax=None)
```

返回 matplotlib 的 `Axes`。单次划分报告显示每个投票模型族对每个特征组的排名，并标出中位数；重采样报告显示每个特征组在各划分中的名次范围，并标出中位数。最重要的特征组在最上方。没有 matplotlib 时引发 `ImportError`（`pip install 'featranker[plot]'`）。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ax` | matplotlib `Axes` | `None` | 要绘制的坐标轴。`None` 会新建一个图形。 |

## 函数

### `auto_groups`

```text
auto_groups(X, *, threshold=0.7, feature_names=None, categorical=None)
```

仅根据 `X`，用完全连接的 Spearman 相关性提出 `feature_groups`。它从不接触目标。对于由两列或更多列组成的特征组，返回 `{group_name: [column, ...]}`，组名由其成员用 `+` 连接而成。`threshold` 或数据无效，或生成的名称与某一列或另一个特征组重名时，引发 `ValueError`。见[用 auto_groups 为相关特征分组](guide/selection.md#用-auto_groups-为相关特征分组)。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array or DataFrame | required | 特征，验证方式与 `fit()` 的输入相同。请使用足够多的行：行数少时，样本相关性会被夸大。 |
| `threshold` | float | `0.7` | 仅限关键字参数。组内每一对的绝对相关性都至少为该值（大于 0，不超过 1）。值越低，特征组越大。 |
| `feature_names` | list of str | `None` | 仅限关键字参数。非 DataFrame 输入的列名；默认 `x0`、`x1` 等。 |
| `categorical` | `"auto"` or list of str | `None` | 仅限关键字参数。同 `FeatureRanker`。这些列按文本接受，且从不分组。 |

### `summarize`

```text
summarize(report)
```

返回结构化的发现，把[阅读指南](../report.md#reading-the-report)应用于一份报告。报告不会被改变。每个发现包含 `code`、`severity`、`message`、`evidence`、`basis` 和 `scoring`。`report` 不是字典时引发 `TypeError`，缺少报告应有的键时引发 `ValueError`。见[汇总报告](guide/selection.md#汇总报告)。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `report` | dict | required | 来自 `rank_features()` 或 `rank_resampled()` 的报告，或从 CLI 的 JSON 加载的报告。 |

## 输入规则

这些是上述各表背后的确切限制。除非另有说明，它们都在拟合任何模型之前检查。

- **随机种子。** `random_state` 和 `shadow_seed` 是 0 到 2**64 - 1 的整数。接受 NumPy 整数。布尔值、NumPy 时间差和生成器会引发 `TypeError`；超出范围的值会引发 `ValueError`。对于整数 CV，小于 2**32 的随机种子直接传给划分器。更宽的随机种子用 NumPy SeedSequence 确定性地生成一个 32 位的划分器随机种子。随机种子会在任何改动之前验证，因此被拒绝的 `fit(random_state=...)` 会让已拟合的排名器保持已拟合状态。任何其他被拒绝的 `fit()`（例如 `max_samples=0`）会让排名器处于未拟合状态，以免误对先前的拟合排名。`rank_resampled()` 再加上划分序号并对 2**64 取模。
- **计数。** `n_repeats`、`top_k`、`k`、`max_samples`（为整数时）、`max_train_samples` 和 `min_shadow_families` 是小于 2**63 的正整数。`max_categories` 是至少为 2 且小于 2**63 的整数。布尔值、浮点数和字符串会被拒绝。
- **`n_jobs`。** `None` 或任何非零整数；负数从全部核心数往回数，与 joblib 相同（`-1` = 全部）。`0`、布尔值、浮点数、字符串或 NumPy 时间差会引发指明 `n_jobs` 的错误。
- **`max_samples` 和 `max_train_samples`。** 整数表示行数（超过总行数则使用全部）。浮点数必须在 (0, 1] 内，表示比例，向下取整，至少为 1。`sample_weight` 值全为零的子样本会引发 `ValueError`。缺少某个类别的 `fit()` 样本会引发指明该类别的 `ValueError`。
- **`models`。** 非空字典。名称为非空字符串，且不能是 `dummy_baseline`。不能与不是 `"all"` 的 `group` 同用。
- **`families`。** 非空字符串到非空字符串的映射，会复制为字典。不是已配置模型的名称，会在 `fit()`、`rank_resampled()` 和 `ConsensusSelector.fit()` 中引发 `ValueError`。初始化失败的内置模型的名称会被接受；不在所选 `group` 或 `task` 内的内置名称会引发错误。
- **`missing`。** 使用内置模型时，构造之后更改 `missing` 会引发 `ValueError`；请新建一个 `FeatureRanker`。
- **`fit()` 之后更改的选项。** `missing`、`model_input`、`max_categories`、`shadows`、`shadow_seed`、`shadow_mode`（使用影子时还有 `shuffle`）会在 `rank_features()` 中引发 `ValueError`（“call fit() again”）。
- **`X` 和 `y`。** `X` 是二维的，至少一行一列。除非 `missing="impute"`，否则 NaN 会被拒绝；无穷值始终被拒绝。回归的目标必须是数值。评估列的名称和顺序必须与 `fit()` 一致。日期和时间列会被拒绝。
- **`feature_names`。** 唯一且非空。NumPy 输入必须提供。
- **`sample_weight`。** 一维、有限、非负、每行一个、总和为正。其他情况都会引发 `ValueError`。
- **`shuffle_groups` 和 `groups`。** 每行一个标签。使用 `"within_groups"` 时标签必须可排序（`None`、NaN 和混合类型会引发 `ValueError`），且如果每个分组都只有一行，调用会引发 `ValueError`。
- **`scoring`。** 列表中的名称必须唯一。空列表、重复的名称或无效的条目会在任何模型排名之前引发错误。
- **`cv`。** 训练和评估索引完全相同的两个划分会引发 `ValueError`。
- **`ablate()` 中的 `ks`。** 由 1 到共识特征组数量之间的整数组成的非空列表；会去重并排序。其他情况引发 `ValueError`。
- **`feature_groups`。** 空组、未知特征、重复归属、重叠和组名冲突都会引发错误。
- **`ablate()` 与报告。** 在以下情况引发 `ValueError`：排名器未拟合、报告是重采样报告、共识为空、训练数据与 `run["train_data_sha256"]` 不同（该值为 `null` 时跳过）、权重与记录的指纹不同、排名器最近一次 `fit()` 没有拟合某个投票模型、某个投票模型属于另一个类（`models[m]["class"]`）、`model_input` 或类别列与排名器不同，或同名模型的参数与排名器的不同。有训练指纹时，最近一次拟合必须与报告的训练数据和权重相符。非标量参数通过 `models[m]["param_hashes"]` 比较，嵌套的组件类通过 `models[m]["components"]` 比较。缺少这些键、缺少 `params` 或缺少 `run` 块的报告不对这些项检查，且排名器仍必须使用了每个训练行。在另一个 scikit-learn 版本下生成的报告，只比较两个版本都有的参数，参数不同时仅发出警告（`UserWarning`）。在 1.22 之前 pickle 序列化的、带有用户模型的排名器，会重新拟合其已拟合模型的克隆，而不是当前的模板。

训练证据要求使用 `evidence=True` 拟合。重新拟合会丢弃先前训练集保留下来的证据。
