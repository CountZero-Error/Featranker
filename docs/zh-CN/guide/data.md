# 数据

[English](../../guide/data.md) · 简体中文

本页介绍缺失值、类别特征、样本权重，以及 FeatRanker 如何处理标签和你的数组。

[手册](../README.md) | [自定义模型](custom-models.md) | [验证](validation.md) | [参考](../reference.md)

## 缺失值

默认情况下，特征中出现 NaN 会抛出 `ValueError`。设置 `missing="impute"` 后，FeatRanker 接受 NaN（无穷值仍会被拒绝），并在每个模型内部进行插补，因此不会有评估行参与塑造插补值。

- 每个内置模型都变成以 `SimpleImputer(strategy="median")` 开头的管道（Pipeline），线性模型随后再接常用的缩放器。插补器只在模型拟合所用的行上拟合，不使用其他数据。在 `rank_resampled()` 和 `ConsensusSelector` 下，每个训练划分各自学习自己的中位数。
- 这对所有内置模型都成立，包括原生支持 NaN 的模型（如 `HistGradientBoosting`），因此各模型和各 scikit-learn 版本遵循同一条规则。
- 在所有训练行中都缺失的列会变成常数 0，并保留原位置。它的重要性没有意义，`train_counts` 会显示这一点。
- 你自己的 `models=` 会原样收到 NaN，必须自行处理，例如使用 `make_pipeline(SimpleImputer(...), ...)`。
- 哑基线（dummy baseline）忽略特征值，不受影响。
- 证据在有 NaN 时仍可用。相关特征组和 `auto_groups()` 对涉及含 NaN 列的每一对列，使用两者都有值的行。有效行少于 30 的列对视为不相关。SHAP 把最前面的插补器视为逐列步骤。存在 NaN 时，成对相关的开销为 O(列数的平方乘以行数)。
- 报告会增加 `missing`，给出每个特征的训练缺失值数量（[报告](../../report.md#structure)）。

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X = data.data.copy()
rng = np.random.default_rng(0)
X[rng.random(len(X)) < 0.2, 2] = np.nan  # 20% of bmi missing
X[rng.random(len(X)) < 0.1, 3] = np.nan  # 10% of bp missing
X_train, X_eval, y_train, y_eval = train_test_split(
    X, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", missing="impute", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)

print(report["missing"]["train_counts"])
for row in report["consensus"][:3]:
    print(f"{row['feature_group']:>5}  median rank {row['median_rank']:.1f}")

imputer = ranker.trained_models["ridge_regression"][0]  # holds the training median of bmi
print(round(imputer.statistics_[2], 4), round(np.nanmedian(X_train[:, 2]), 4),
      round(np.nanmedian(X_eval[:, 2]), 4))
```

```text
{'age': 0, 'sex': 0, 'bmi': 67, 'bp': 27, 's1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0, 's6': 0}
   s5  median rank 1.0
  bmi  median rank 2.0
   bp  median rank 3.5
-0.0105 -0.0105 0.0013
```

最后一行显示，已拟合的插补器对 `bmi` 使用的中位数等于训练集中位数（-0.0105），而不是评估集中位数（0.0013）。

`ConsensusSelector("reg", "r2", k=2, missing="impute")` 在 `fit()` 和 `transform()` 中接受 NaN。它只负责选择列，因此请在 `Pipeline` 中把插补器放在它后面。在命令行中，`--missing impute` 会把 CSV 中的空单元格读作缺失值。

> [!WARNING]
> 中位数插补会把缺失情况与数值混在一起。只对病情较重的患者开具的化验项目，其缺失是有原因的；插补会在中位数处留下一个尖峰，树模型和核模型可以学到它。这样，一个特征的重要性可能混合了它的取值和它的缺失情况（在一次只由缺失情况驱动目标的探测中，树模型的重要性约为 1.9，SVR 约为 1.05，而岭回归为 0.006）。若要把两者分开，请使用带 `SimpleImputer(add_indicator=True)` 或原生 NaN 处理的自定义模型，并查看 `train_counts`。目标中的缺失值不受支持。

## 类别特征

默认情况下，每个特征都必须是数值。使用 `categorical` 后，FeatRanker 会原样接受文本列和类别列，在每次 `fit()` 内部学习如何编码，并把每个原始列作为一个特征来排名。

- `categorical="auto"` 把 DataFrame 中 object、string 或 category 类型的列视为类别列（bool 和数值类型仍视为数值）。NumPy 数组没有逐列的数据类型，因此 `"auto"` 在其中什么也找不到。
- `categorical=["city", ...]` 无论数据类型如何都会标记这些列，对 NumPy object 数组同样适用（需提供 `feature_names`）。不是特征的名称会在 `fit()` 中抛出 `ValueError`。
- **水平（level）只来自训练行。** 每列的水平是其训练值的字符串形式按字典序排列的结果，编码为 `0..k-1`；缺失值（`None` 或 NaN）自成一个水平，排在最后。训练中未出现的值（包括训练时没有缺失值而评估时出现的缺失值），在评估阶段共用一个额外的“未见过”编码 `k`。`rank_resampled()` 和 `ConsensusSelector` 的每个划分都学习自己的水平。已拟合的排名器暴露 `categorical_features_` 和 `categorical_levels_`（名称到水平的映射，缺失水平用 `None` 表示）。
- **每个原始列对应一个特征。** 无论有多少个水平，`city` 在 `feature_groups`、共识和影子特征中都只是一个特征组。
- **内置线性模型**对类别列做独热编码（训练中从未见过的编码会得到全零行），并对数值列做缩放。**内置树模型**把编码当作数字使用，这使各水平带有任意的顺序。**你自己的模型**收到的是编码矩阵（浮点数）；请读取 `ranker.categorical_features_` 来确定哪些列存放编码。对于高基数列或其他编码方式，请传入你自己的 `models=`。
- 在 `missing="impute"` 下，先对数值列做插补，缺失的类别值则自成一个水平。
- 报告会增加 `categorical`：包含这些列、它们的训练水平数，以及评估值中有多少是未见过的。
- 证据：类别列不参与 Spearman 相关（`auto_groups(X, categorical=...)` 绝不会把它们分组）。SHAP 对树模型按编码解释（类别特征组的 `direction` 为 `null`），对内置线性模型则把独热列按来源列求和。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 400
city = rng.choice(["Oslo", "Rome", "Lima", "Kyiv", "Cairo"], n)
premium = {"Oslo": 30.0, "Rome": -20.0, "Lima": 25.0, "Kyiv": -25.0, "Cairo": 0.0}
size = rng.uniform(30, 120, n)
age = rng.uniform(0, 60, n)
price = size * 2 + np.array([premium[c] for c in city]) + rng.normal(0, 8, n)
df = pd.DataFrame({"city": city, "size": size, "age": age})
train, evaluation = df.iloc[:280], df.iloc[280:]
train = train.assign(city=train["city"].replace("Cairo", "Rome"))  # Cairo never appears in training

ranker = FeatureRanker(task="reg", group="fast", categorical="auto", verbose=False)
ranker.fit(train, price[:280])
report = ranker.rank_features(
    evaluation, price[280:], scoring="neg_mean_absolute_error", random_state=42
)

print(ranker.categorical_features_, ranker.categorical_levels_)
print(report["categorical"])
print(list(report["feature_groups"]))
for row in report["consensus"]:
    print(f"{row['feature_group']:>5}  median rank {row['median_rank']:.1f}")
```

```text
['city'] {'city': ['Kyiv', 'Lima', 'Oslo', 'Rome']}
{'columns': ['city'], 'train_levels': {'city': 4}, 'unseen_eval': {'city': 28}}
['city', 'size', 'age']
 size  median rank 1.0
 city  median rank 2.0
  age  median rank 3.0
```

`city` 虽有四个训练水平，但只是一个特征。来自 Cairo 的 28 个评估行得到的是共享的“未见过”编码，而没有参与塑造编码。`ConsensusSelector(categorical="auto")` 的 `transform()` 返回所选的原始列；请在 `Pipeline` 中把编码器放在它后面。命令行接受 `--categorical auto` 或 `--categorical city,segment`。

### 稀疏独热编码输出

自 2.0 起，除六个只支持稠密输入的估计器外，每个内置线性模型的独热块都保持稀疏（使用 scikit-learn 自带的稀疏求解器）：`LinearDiscriminantAnalysis`、`QuadraticDiscriminantAnalysis`、`GaussianNB`、`PLSRegression`、`SGDClassifier` 和 `Perceptron`。它们得到稠密输出（SGD 的截距更新在稀疏输入上有所不同）。

- 稀疏与稠密求解的是同一个优化问题，但求解器层面的差异仍可能使小样本上的少数预测发生变化（例如 `ridge_classifier` 和 `huber_regression`）。
- 密度为 1.0 的独热块（例如唯一的类别列只有一个水平）仍使用稠密输出。`solver` 为 `saga` 或 `sag` 的逻辑回归模型也使用稠密独热编码，因为稀疏输入会使 saga 提前停止。
- 稀疏输入修复了 `linear_regression` 和 `ransac_regression` 在类别列上的发散问题（稠密最小二乘在完整独热块加截距上发散，得到 -5e21 之类的得分，从 1.9 到 1.x 均如此）。它们在完整独热块上的最小二乘解并不唯一，因此在稠密求解没有发散的情况下，它们带类别列的得分也可能与稠密的 1.x 结果不同。
- `max_categories` 限制独热编码的宽度；参见[性能](performance.md#大数据)。

> [!WARNING]
> 水平按字符串形式比较，因此 `1` 和 `"1"` 是同一个水平，`1` 和 `1.0` 也是。请先清理这类列。不提供目标编码和其他有监督编码器，因为它们会泄漏目标；如有需要，请在自定义管道中自行构建。日期和时间列会被拒绝（"X must be numeric"），即使列为类别列也一样；请先把它们转换成数字或标签。

## 样本权重

调查数据、类别不平衡和聚合行都会带有样本权重。把它们作为 `sample_weight` 传给 `fit()`（训练行）、`rank_features()`（评估行）或 `rank_resampled()`（所有行；每个划分取各自的切片）。每个权重都是一维类数组，由有限的非负数组成，每行一个，且总和为正；否则会在任何模型拟合之前抛出 `ValueError`。权重会被复制。

**哪些部分被加权。** 有训练权重时，每个模型都带权重拟合。内置管道只把权重传给最后的估计器（插补器、缩放器和独热编码器保持不加权，与 scikit-learn 一致）。哑基线也带权重拟合。有评估权重时，基线得分、每个置换后的得分以及哑基线的得分都被加权。在 `evidence=True` 且有训练权重时，训练得分也使用它们。两组权重可以只传其中一组。报告会记录 `sample_weight: {"train": ..., "eval": ...}`；没有权重时该键不存在。

**不能接受权重的模型。** 如果某个模型的最终估计器的 `fit()` 没有 `sample_weight`，它不会以不加权方式拟合，因为它的重要性回答的是另一个问题。它会以阶段 `"fit"` 和 `error_type` `"SampleWeightNotSupported"` 记录在 `failures` 中。内置模型中有 `lda` 和 `qda`（分类）以及 `pls_regression`。你自己的模型也是如此，无论是单独的估计器还是 `Pipeline` 的最后一步。`fit` 接受 `**kwargs` 的估计器视为支持权重，嵌套的 `Pipeline` 会一路追踪到其最后一步。

**置换的供体不加权。** 供体是被评分的行（在 `max_samples` 之后，包括零权重行），权重对谁来充当供体不起作用。在 `shuffle="rows"` 下，每次置换都随机重排被评分的行，因此无论权重多大，每一行都恰好供体一次。影子列是对每一行的打乱副本，同样不加权。只有得分被加权。因此，置换后的数据遵循评估行的不加权分布，而基线是加权的。

> [!NOTE]
> 这适用于代价权重或重要性权重（“这些行上的错误代价更高”）。对于调查权重或频数权重，加权行代表一个总体，权重大的行并不比权重小的行更有可能贡献出它的值。如果你关心的是总体，请删除零权重行，或按权重对评估行重采样并且不再为它们传入 `sample_weight`，然后比较结果。FeatRanker 没有按权重抽取供体的选项。

**不加权的证据。** 相关性证据和 SHAP 摘要描述的是数据和已拟合的模型，而不是加权后的目标，因此保持不加权。

```python
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(0)
n = 600
survey = np.arange(n) % 2 == 0  # rows that carry almost all of the weight
x1, x2 = rng.normal(size=n), rng.normal(size=n)
X = np.column_stack([x1, x2, survey.astype(float)])
y = np.where(survey, x1, x2) + 0.1 * rng.normal(size=n)
w = np.where(survey, 1.0, 0.001)
names = ["x1", "x2", "is_survey"]
models = {"forest": RandomForestRegressor(n_estimators=50, random_state=0)}

for label, weights in [("unweighted", None), ("weighted", w)]:
    ranker = FeatureRanker(task="reg", models=models, verbose=False)
    ranker.fit(X[:400], y[:400], feature_names=names,
               sample_weight=None if weights is None else weights[:400])
    report = ranker.rank_features(
        X[400:], y[400:], scoring="r2", n_repeats=5, random_state=0,
        sample_weight=None if weights is None else weights[400:],
    )
    importance = report["models"]["forest"]["importance"]
    print(label, report.get("sample_weight"),
          {name: round(importance[name]["mean"], 2) for name in names[:2]})

builtin = FeatureRanker(task="clf", group="all", verbose=False)
builtin.fit(X[:400], (y[:400] > 0).astype(int), feature_names=names, sample_weight=w[:400])
print([(f["model"], f["error_type"]) for f in builtin.fit_failures_])
```

```text
unweighted None {'x1': 0.85, 'x2': 0.8}
weighted {'train': True, 'eval': True} {'x1': 1.44, 'x2': 0.02}
[('lda', 'SampleWeightNotSupported'), ('qda', 'SampleWeightNotSupported')]
```

在权重很大的行中，`x1` 驱动目标；在几乎没有权重的行中，驱动目标的是 `x2`。不加权时两者看起来相近；加权后，`x2` 几乎消失。

**选择器、管道和命令行。** `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)` 会把权重传给它的重采样排名。在 `Pipeline` 中，请使用 scikit-learn 的元数据路由：`sklearn.set_config(enable_metadata_routing=True)`、`ConsensusSelector(...).set_fit_request(sample_weight=True)`，然后 `pipeline.fit(X, y, sample_weight=w)`。命令行接受 `--sample-weight COLUMN`：该列必须同时存在于两个 CSV 文件中，会被读作权重，且不会作为特征参与排名。

## 标签与数据处理

`FeatureRanker` 不做特征选择，也不调参。它只在两处改变数据值：

- **线性缩放。** 内置线性模型前面有一个只在训练数据上拟合的 `StandardScaler`。树模型和你自己的模型得到的是原样数据。
- **分类标签。** 对 `task="clf"`，`fit()` 用 `LabelEncoder` 把标签编码为 `0..n_classes-1`，因为某些估计器（XGBoost）要求如此。包括你的在内，每个估计器和评分器看到的都是编码后的标签。`ranker.label_encoder.classes_` 把每个编码映射回去：编码 `i` 对应 `classes_[i]`。

`fit()`、`rank_features()` 和 `rank_resampled()` 在各自的 `X` 和 `y` 浮点副本上工作。之后再修改你的数组，不会改变模型、证据或后续报告，即使是保存了训练数据的估计器（如 k 近邻）也是如此。代价是每次调用多一份副本。

因此，自定义评分器必须按编码引用类别。要为原始类别 `"case"` 计算 F1：

```python
from sklearn.metrics import f1_score, make_scorer

labels = np.where(x1 + 0.5 * rng.normal(size=n) > 0, "case", "control")
clf = FeatureRanker(task="clf", group="fast", verbose=False)
clf.fit(X[:400], labels[:400], feature_names=names)
case_code = list(clf.label_encoder.classes_).index("case")
scorer = make_scorer(f1_score, pos_label=case_code)
report = clf.rank_features(X[400:], labels[400:], scoring=scorer, n_repeats=3, random_state=0)
print(list(clf.label_encoder.classes_), case_code, report["consensus"][0]["feature_group"])
```

```text
['case', 'control'] 0 x1
```
