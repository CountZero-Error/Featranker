# 自定义模型

[English](../../guide/custom-models.md) · 简体中文

本页介绍如何用你自己的估计器和管道进行排名、把它们归入模型族、按名称选择列，以及并行运行模型。

[手册](../README.md) | [模型与评分](models-and-scoring.md) | [数据](data.md) | [参考](../reference.md)

## 你自己的模型

传入 `models`，即可用你自己的估计器代替内置模型集进行排名。当你需要特定的估计器，或带有自己预处理的管道（Pipeline）时，请使用它。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featranker import FeatureRanker

ranker = FeatureRanker(
    task="reg",
    models={
        "scaled_ridge": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "forest": RandomForestRegressor(n_estimators=50, random_state=0),
    },
    verbose=False,
)
print(sorted(ranker.user_models))
```

```text
['forest', 'scaled_ridge']
```

- 估计器会在 `fit()` 时被克隆，因此你的原始对象不会被修改。`Pipeline` 的预处理只在训练行上拟合。
- 名称必须是非空字符串，且不能是 `dummy_baseline`。`models` 不能与不是 `"all"` 的 `group` 同时使用。
- 每个模型都在共享的特征、目标和权重数组的只读视图上拟合和评分。向其输入写入数据的估计器会抛出异常，该异常只会被记录为这个模型的 `fit`（或 `ranking`）失败。会先复制再处理的 scikit-learn 估计器可以正常工作，例如 `LinearRegression(copy_X=False)`。有些估计器（如 `Ridge`）在拟合时会再复制一次只读的 `X`，这在大数据上可能使峰值内存增加约六分之一。

> [!IMPORTANT]
> 默认情况下，即使你传入 DataFrame，你的估计器收到的也是浮点 NumPy 数组。在管道中请按位置选择列，例如 `ColumnTransformer([("scale", StandardScaler(), [0, 1])])`，或使用 `model_input="dataframe"`（见下文）。到那时，类别值必须已是数字：请自行编码，或传入 `categorical=`，这样你的估计器收到的是整数编码（[数据](data.md#类别特征)）。使用 `missing="impute"` 时，你的估计器会原样收到 NaN（[数据](data.md#缺失值)）。

## 模型族

默认情况下，每个自定义模型自成一个模型族，每个模型族投一票。三个几乎相同的岭回归模型就是三票，可能压过另一种类型的模型。这种保护只对内置模型族有效。

用 `families={model_name: family}` 说明哪些模型属于同一族。映射到同一模型族的模型只投一票，取其成员的中位数。报告的 `family` 字段显示这一映射。

```python
ranker = FeatureRanker(
    task="reg",
    models={
        "ridge_small": Ridge(alpha=0.1),
        "ridge_large": Ridge(alpha=10.0),
        "forest": RandomForestRegressor(n_estimators=50, random_state=0),
    },
    families={"ridge_small": "ridge", "ridge_large": "ridge"},  # forest stays its own
    verbose=False,
)
print(ranker.families)
```

```text
{'ridge_small': 'ridge', 'ridge_large': 'ridge'}
```

- 不是已配置模型的名称，会在 `fit()` 中抛出 `ValueError`（在 `rank_resampled()` 和 `ConsensusSelector.fit()` 中则在任何划分之前抛出）。
- 空的或非字符串的名称或模型族，以及不是映射的值，会在构造函数中抛出 `ValueError`。任何映射都会被接受，并复制为 dict。
- `families` 也接受内置模型名称，用于重新分组。初始化失败的内置模型的名称也会被接受。不在所选 `group` 或 `task` 内的内置名称会抛出异常。
- `ConsensusSelector(families=...)` 会把它传下去。

## 按名称选择列

`model_input="dataframe"` 会让每个模型（内置和自定义）都在 pandas DataFrame 上拟合、评分和置换。它的列依次是各特征名称，开启 `shadows=True` 时，再加上每个影子列的 `shadow__<name>`。这是 scikit-learn 中预处理混合数据的常规做法：`ColumnTransformer([("num", StandardScaler(), ["age", "income"])])`。

- 对不依赖列名的模型，重要性、得分、训练得分和 SHAP 摘要与 `"array"` 模式相同。每次评分调用都会从置换后的数组重建 DataFrame，这在宽数据上会多花一点时间。
- 报告会记录 `model_input: "dataframe"`；默认模式下该键不存在。
- `rank_resampled()` 和 `ConsensusSelector(model_input=...)` 在每个划分中都使用它。命令行保持数组输入。
- 没有 pandas 时，`model_input="dataframe"` 会在构造时抛出 `ImportError`。
- 开启影子特征时，特征名称不能以 `shadow__` 开头。

### 影子特征约定

开启影子特征时，自定义管道必须保留影子列。从未见过影子列的模型，其影子重要性为零，于是每个真实特征都“胜过”了影子特征。

拟合之后，FeatRanker 会在两种输入模式下推算每个模型使用了哪些输入列。它会沿 `Pipeline` 进入第一步，沿 `ColumnTransformer` 进入每个分支，沿 `FeatureUnion` 进入每个部分，并沿已拟合的选择器（`SelectKBest`、`SelectFromModel` 等）找到它保留的列。后面的 `Pipeline` 步骤如果是或包含选择器、`FeatureUnion`、`ColumnTransformer` 或不透明的转换器，只有在前面每一步都逐列保留各列时才会被检查（`StandardScaler`、`MinMaxScaler`、`MaxAbsScaler`、`RobustScaler`，以及不添加指示列也不丢弃列的 `SimpleImputer`）。在任何其他步骤之后，它都算作未验证。

如果转换器和 `remainder` 没有使用某个影子列，或者无法推算出所用的列，该模型就会被排除在该报告的所有影子比较之外：没有 `above_shadow` 标志，没有 `shadow_max`，也不计入 `family_above_shadow` 或 `above_shadow_share`。`failures` 会增加一条记录，阶段为 `"shadows"`，`error_type` 为 `"ShadowColumnsDropped"`。它的重要性和排名投票不受影响。消息会说明该模型是“does not use every shadow column”还是“could not be verified”。使用 `remainder="passthrough"`，或列出 `shadow__` 列，可让它留在影子证据中。

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({"age": rng.uniform(20, 70, n), "income": rng.normal(50, 10, n), "noise": rng.normal(size=n)})
y = 0.3 * df["age"] + 0.1 * df["income"] + rng.normal(0, 1, n)
train, evaluation = df.iloc[:200], df.iloc[200:]

models = {
    # Lists only the real columns and drops the rest, shadows included.
    "drops_shadows": make_pipeline(
        ColumnTransformer([("num", StandardScaler(), ["age", "income", "noise"])]), Ridge()
    ),
    "keeps_shadows": make_pipeline(
        ColumnTransformer(
            [("num", StandardScaler(), ["age", "income"])], remainder="passthrough"
        ),
        Ridge(),
    ),
}
ranker = FeatureRanker(
    task="reg", models=models, shadows=True, shadow_seed=0,
    model_input="dataframe", verbose=False,
)
ranker.fit(train, y[:200])
report = ranker.rank_features(evaluation, y[200:], scoring="r2", random_state=0)

print(report["model_input"])
print([(f["model"], f["stage"], f["error_type"]) for f in report["failures"]])
print([m for m, r in report["models"].items() if "shadow_max" in r])
for row in report["consensus"]:
    print(f"{row['feature_group']:>6}  median rank {row['median_rank']:.1f}  "
          f"n_models {row['n_models']}  above_shadow_share {row['above_shadow_share']:.1f}")
```

```text
dataframe
[('drops_shadows', 'shadows', 'ShadowColumnsDropped')]
['keeps_shadows']
   age  median rank 1.0  n_models 2  above_shadow_share 1.0
income  median rank 2.0  n_models 2  above_shadow_share 1.0
 noise  median rank 3.0  n_models 2  above_shadow_share 0.0
```

两个模型都参与共识投票，但只有 `keeps_shadows` 提供影子证据。`summarize()` 会把 `"shadows"` 阶段报告为警告。

> [!WARNING]
> 这项检查无法发现所有丢弃列的方式。
> - 带函数的 `FunctionTransformer`，或用于转换输入列的自定义转换器，是不透明的。当管道的中间步骤同时提供 `predict()` 时，它仍然起转换器的作用。它可能丢弃列，因此该模型算作未验证。
> - 被元估计器（`TransformedTargetRegressor`、`BaggingRegressor`、`GridSearchCV` 等）包裹的选择器、`FeatureUnion` 或 `ColumnTransformer` 算作未验证。
> - scikit-learn 自带的转换器（缩放器、编码器、插补器）和来自任何库的最终预测器，都被假定会使用它们收到的每一列。其他会忽略部分输入的估计器同样被假定使用全部输入，这由你自己负责。
> - 在 dataframe 模式下，XGBoost 会拒绝包含 `[`、`]` 或 `<` 的特征名称；这些拟合失败会出现在 `failures` 中。请重命名列，或使用 array 模式。

## 并行模型

`n_jobs=2` 用 joblib 并行运行 `fit()` 和 `rank_features()` 中每个模型的工作。`None` 或 `1` 为顺序执行，`-1` 使用所有核心。其他值会抛出点明 `n_jobs` 的错误。

- 当每个估计器都是确定性的或设定了随机种子、评分器是确定性的，并且 `random_state` 是固定整数时，报告与 `n_jobs=None` 一致，模型顺序也相同。每个带随机性的内置模型都设定了随机种子（`random_state=42`）。
- 未设随机种子的自定义估计器，例如没有 `random_state` 的 `RandomForestRegressor()`，每次运行都可能给出不同的报告，无论是否并行。
- 开启 `shadows=True` 时，可复现还需要固定的整数 `shadow_seed`。只固定 `random_state` 不会给影子特征的构造设定种子。
- 每个模型的失败与顺序执行时一样被记录。
- 无论使用何种并行设置，自定义评分器和估计器都会被发送到工作进程，因此它们持有的一切都必须可序列化。lambda 和局部函数没有问题；锁和连接则不行。如果遇到 pickle 错误，请使用 `n_jobs=None`。
