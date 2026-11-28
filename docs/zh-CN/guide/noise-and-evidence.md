# 噪声与证据

[English](../../guide/noise-and-evidence.md) · 简体中文

本页介绍如何用影子特征、影子命中检验和模型族证据区分信号与噪声，以及如何借助证据和 SHAP 弄清排名为什么是现在这个样子。

[手册](../README.md) | [验证](validation.md) | [选择](selection.md) | [报告](../../report.md) | [参考](../reference.md)

## 影子特征

排名只说明某个特征组比其他特征组更重要，并不说明它带有任何信号。设置 `shadows=True` 后，每个模型还会得到全部特征的一份*影子*副本：同样的行，但顺序被打乱。影子块保留了特征值和行内关系，同时随机地重新分配行，因此它是一个随机化的参照。当某个真实特征组在一个模型中的平均重要性大于最强影子的平均重要性时，它在该模型中就是 `above_shadow`。

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker, summarize

data = load_diabetes()
rng = np.random.default_rng(0)
X = np.hstack([data.data, rng.normal(size=(len(data.data), 2))])  # two pure-noise columns
names = list(data.feature_names) + ["noise_1", "noise_2"]
X_train, X_eval, y_train, y_eval = train_test_split(
    X, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(
    X_eval, y_eval, scoring="neg_mean_absolute_error", random_state=42
)

for row in report["consensus"]:
    print(
        f"{row['feature_group']:>7}  median rank {row['median_rank']:4.1f}  "
        f"family-weighted shadow-hit share {row['above_shadow_share']:.0%}"
    )
```

```text
    bmi  median rank  1.5  family-weighted shadow-hit share 100%
     s5  median rank  1.5  family-weighted shadow-hit share 100%
     bp  median rank  3.0  family-weighted shadow-hit share 100%
noise_2  median rank  5.5  family-weighted shadow-hit share 25%
    sex  median rank  6.0  family-weighted shadow-hit share 0%
     s3  median rank  6.5  family-weighted shadow-hit share 25%
     s4  median rank  7.0  family-weighted shadow-hit share 25%
     s6  median rank  7.0  family-weighted shadow-hit share 25%
     s2  median rank  8.0  family-weighted shadow-hit share 25%
     s1  median rank  8.5  family-weighted shadow-hit share 50%
    age  median rank  9.0  family-weighted shadow-hit share 25%
noise_1  median rank 10.5  family-weighted shadow-hit share 0%
```

`bmi`、`s5` 和 `bp` 在每个模型族中都胜过了它们的影子特征。两个纯噪声列的占比为 25% 或更低。

- `above_shadow_share` 按模型族加权：每个投票的模型族贡献的是该特征组胜过最强影子的成员所占的比例。
- `above_shadow` 是相对比较，不能证明存在有用的信号。如果 `shadow_max` 为负，即使是为负的真实重要性也可能胜过它。请把真实的平均值、打乱的离散程度和评估得分与这个占比放在一起看。
- 它是描述性的，不是 p 值。模型族不是独立试验，占比低于 0.5 也不能证明某个特征组无用。要做跨划分的检验，请参见[命中检验](#影子命中检验)。
- 排名仍然只比较真实特征组。额外的噪声列仍可能改变某些模型的拟合方式（例如每次划分只抽取一部分特征的森林），因此开启影子特征后的排名可能与不开启时略有不同。
- **开销。** 模型在两倍的列上训练，每个特征组被置换两次：预计时间和内存约为原来的两倍。
- `rank_resampled()` 使用同样的设置。第 `i` 个划分使用 `(shadow_seed + i)` 对 2**64 取模。每个稳定性行会增加 `above_shadow_share`（各划分的值之和除以 `n_splits`：失败的、共识为空的或缺少该特征组的划分计为 0）、命中检验的键和模型族证据的键。`ConsensusSelector(min_shadow_share=...)` 就根据这个值过滤。

> [!WARNING]
> 只有当每个模型都能看到每一列时，影子特征才有效。会丢弃列的自定义管道，例如使用默认 `remainder="drop"` 的按位置选列的 `ColumnTransformer`，永远看不到影子特征，因此每个特征组都轻易地胜过它们。影子列被附加在真实列之后，所以对真实列按位置选择的选择器仍然有效。参见[影子特征约定](custom-models.md#影子特征约定)。

## 影子命中检验

开启影子特征时，`rank_resampled()` 还会对每个特征组检验：它在各划分中胜过自己影子的次数，是否多于随机情形，且无需额外的模型拟合。这就是应用到你的划分上的 Boruta 命中检验。在每个划分中，当某个特征组的共识 `above_shadow_share` 严格大于 0.5 时，它就有一次*命中*。每个稳定性行会增加：

- `shadow_hits`：命中的划分数。特征组在其中没有共识行的划分计为未命中；试验次数始终是 `n_splits`。
- `shadow_p_value`：在 `n_splits` 次试验、成功概率为 0.5 的条件下，至少出现 `shadow_hits` 次命中的单侧二项概率。
- `shadow_q_value`：跨所有特征组的 Benjamini-Hochberg 校正后 p 值。

报告含有一个 `shadow_test` 块（方法、命中规则、`null_probability`、校正方法、`n_trials`、`min_p_value`、`calibrated: false` 和一条 `note`）。对于任意估计器或重叠的划分，0.5 的零假设概率和错误发现率控制都没有得到确立，因此不要把这些值当作经过校准的显著性。

```python
from sklearn.model_selection import RepeatedKFold

ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
report = ranker.rank_resampled(
    X, data.target, feature_names=names, scoring="r2", n_repeats=3, random_state=0,
    cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
)

test = report["shadow_test"]
print(f"{test['n_trials']} trials, smallest possible p-value {test['min_p_value']:.4f}")
for row in report["stability"]:
    print(
        f"{row['feature_group']:>7}  hits {row['shadow_hits']:>2}/{test['n_trials']}  "
        f"p {row['shadow_p_value']:.4f}  q {row['shadow_q_value']:.4f}"
    )
```

```text
10 trials, smallest possible p-value 0.0010
     s5  hits 10/10  p 0.0010  q 0.0059
    bmi  hits 10/10  p 0.0010  q 0.0059
     bp  hits  7/10  p 0.1719  q 0.6875
     s3  hits  4/10  p 0.8281  q 1.0000
     s6  hits  3/10  p 0.9453  q 1.0000
     s4  hits  0/10  p 1.0000  q 1.0000
    sex  hits  0/10  p 1.0000  q 1.0000
    age  hits  0/10  p 1.0000  q 1.0000
     s2  hits  0/10  p 1.0000  q 1.0000
     s1  hits  1/10  p 0.9990  q 1.0000
noise_2  hits  0/10  p 1.0000  q 1.0000
noise_1  hits  0/10  p 1.0000  q 1.0000
```

需要记住的局限：

- **未经校准的零假设。** 该检验与 0.5 比较，这是 Boruta 的选择，但这里没有任何东西表明，没有效应的特征组在一半或更少的划分中命中。自定义估计器可能破坏它：一个对纯噪声列的重要性总是高于其影子的模型，会给噪声带来很小的 q 值。请用 `shadow_p_value` 和 `shadow_q_value` 来给特征组排序，而不要把它们当作概率或错误率。
- **划分之间不独立。** 交叉验证的训练集相互重叠，因此各划分不是独立试验，即使对表现良好的模型，二项尾概率也不精确。
- **划分少时检验力低。** `n_splits` 允许的最小 p 值是 `0.5 ** n_splits`（`shadow_test["min_p_value"]`）：默认的 `cv=5` 时为 0.031。在 `cv=5` 时，只有在全部 5 个划分中都命中的特征组才能达到它（5 次中命中 4 次，p = 0.19），而 Benjamini-Hochberg 会把它乘以特征组数量再除以其名次。有 12 个特征组、其中两个 5 次全中时，两者的 q 都是 0.19。请使用 `RepeatedKFold` 或其他有 10 个或更多划分的划分器。划分更少时不会发出警告，只是当 `ConsensusSelector` 的 `max_shadow_q` 什么也没选出时，会在它的警告中加上 `n_splits` 和最小可能的 p 值。
- **较小的 q 意味着什么。** 该特征组在各划分中胜过了被打乱的噪声。这并不能证明存在因果效应，也不能证明它对新的目标有用。
- **不适用于单划分报告。** `rank_features()` 只有一次试验，因此没有检验。

在[校准研究](../../calibration.md)中，在零数据上（1000 行、10 个独立特征、10 个 `RepeatedKFold` 划分、内置的 `fast` 和 `tree` 模型），没有任何特征组的 `shadow_p_value <= 0.05`，因此这些值在那里是保守的。这在一定程度上是内置的：划分为 10 个时，该检验的实际水平是 0.0107（至少 9 次命中）。该检验的检验力也较低：`max_shadow_q=0.05` 在 0 到 75% 的随机种子下找回了 `c*d` 交互作用。

`ConsensusSelector(max_shadow_q=0.05)` 保留 `shadow_q_value <= 0.05` 的特征组（[选择](selection.md#管道中的-consensusselector)）。它是一种启发式过滤，不是显著性水平，并且是**实验性的**：`fit()` 会发出一条这样说明的 `UserWarning`，报告的 `shadow_test` 块含有 `"experimental": true`。

## 模型族证据

`above_shadow_share` 对每个模型族取一票的平均，因此它衡量的是各模型族之间的一致程度。只有部分模型族能建模的特征组，例如加性模型看不到的纯交互作用，可以被这些模型族发现，但平均下来仍低于 0.5。开启影子特征时，报告还会说明是哪些模型族发现了每个特征组：

- 每个共识行会增加 `family_above_shadow`：对每个投票的模型族，给出其投票成员中重要性胜过该模型最强影子的成员所占的比例。`above_shadow_share` 是这些值的平均值；键按模型族名称排序。
- 在重采样报告中，每个稳定性行会增加 `family_above_shadow_share`（对每个至少在一个划分中投过票的模型族，其各划分比例之和除以 `n_splits`；该模型族没有投票的划分计为 0）和 `shadow_families`，即占比严格大于 0.5 的模型族数量。

这个例子植入了一个交互作用（`x1` 和 `x2` 只有在一起时才起作用，通过逻辑连接函数，外加一个线性特征 `x0` 和两个噪声列）：

```python
from featranker import ConsensusSelector

rng = np.random.default_rng(0)
Xi = rng.normal(size=(800, 5))
score = 1.5 * Xi[:, 0] + 2.0 * Xi[:, 1] * Xi[:, 2]  # x1 and x2 matter only together; x3, x4 are noise
yi = rng.binomial(1, 1 / (1 + np.exp(-score)))  # logistic link, sampled 0/1 labels

options = dict(
    group="fast", cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
    n_repeats=3, random_state=0, shadow_seed=0,
)
by_share = ConsensusSelector("clf", "roc_auc", min_shadow_share=0.5, **options).fit(Xi, yi)
by_family = ConsensusSelector("clf", "roc_auc", min_shadow_families=2, **options).fit(Xi, yi)

for row in by_family.report_["stability"]:
    print(
        f"{row['feature_group']}  above_shadow_share {row['above_shadow_share']:.2f}  "
        f"shadow_families {row['shadow_families']}"
    )
row = next(r for r in by_family.report_["stability"] if r["feature_group"] == "x2")
print({family: round(share, 2) for family, share in row["family_above_shadow_share"].items()})
print("min_shadow_share=0.5   keeps", by_share.selected_groups_)
print("min_shadow_families=2  keeps", by_family.selected_groups_)
```

```text
x0  above_shadow_share 1.00  shadow_families 7
x1  above_shadow_share 0.53  shadow_families 3
x2  above_shadow_share 0.41  shadow_families 3
x4  above_shadow_share 0.19  shadow_families 1
x3  above_shadow_share 0.06  shadow_families 0
{'bagging': 1.0, 'boosting': 1.0, 'discriminant': 0.0, 'linear_margin': 0.0, 'logistic': 0.0, 'naive_bayes': 0.3, 'single_tree': 0.6}
min_shadow_share=0.5   keeps ['x0', 'x1']
min_shadow_families=2  keeps ['x0', 'x1', 'x2']
```

四个加性模型族（logistic、linear margin、discriminant、naive Bayes）在大多数划分中都没有让 `x2` 超过它们的影子特征，因此它的占比停留在 0.5 以下，占比规则把它丢掉了（它保留 `x1` 只是因为占比是 0.53）。bagging、boosting 和单棵树能够表示这个交互作用，因此有三个模型族通过。

`ConsensusSelector(min_shadow_families=N)` 保留 `shadow_families >= N` 的特征组。它回答的是“是否至少有 `N` 个模型族发现了信号”，会开启影子特征，并可与其他规则组合（所有过滤都应用后，再取 `k`）。

- **何时使用。** 当你怀疑存在只有某些类型的模型才能捕捉的交互作用或效应时。
- **选择 `N`。** `N=1` 会保留任何单个模型族发现的特征组，因此一个模型族拟合噪声（例如单棵决策树）就可以让噪声自己通过；在上面的输出中，噪声列 `x4` 的 `shadow_families` 已经是 1。要求两个模型族，则意味着两种模型必须达成一致。[校准研究](../../calibration.md)在合成数据上测量了这一点。它的预先注册规则推荐 `N=1`：在零数据上，`N=1` 和 `N=2` 各自最多选中 1% 的噪声特征组，并且在 `group="fast"` 下，两者在每个随机种子下都找到了 `c*d` 交互作用。事后来看，在信号研究中 `N=1` 保留了一些噪声（每次运行最多 4 个噪声特征中的 0.4 个），`N=2` 最多 0.1 个，召回率没有损失。如果更看重召回率，请使用 `N=1`；如果更在意避免错误选择，请使用 `N=2`。在分类中，`group="fast"` 下请避免 `N=3`（它只在 0 到 15% 的随机种子下找到了 `c*d`）。这只是使用内置模型的一种合成设计：请像 `max_shadow_q` 一样，把它当作一种启发式方法。
- **哪些模型族计入。** 每个投票的模型族权重相同，包括无法表示你所关心的效应的模型族。如果愿意，可以用 `group` 限制它们。使用你自己的 `models=` 时，除非 `families=` 作了映射，否则每个模型自成一个模型族。在 `group="tree"` 下，回归和分类各有三个模型族（bagging、boosting、单棵树），因此 `N=2` 在那里就是多数。`"fast"` 有七个分类模型族和五个回归模型族。

> [!NOTE]
> 在真实数据上，`min_shadow_share=0.5` 可能过于严格；参见[真实数据研究](../../realdata.md)。

## 影子模式

默认情况下（`shadow_mode="joint"`），影子块是用同一个打乱顺序应用到每一列的行，因此两个共线的真实列在影子特征中保持完全相同的共线性。`shadow_mode="independent"` 则用各自的行顺序打乱每个影子列：每一列仍然是其真实列的一个排列，但列之间的关系被破坏。这些顺序由一个以 `shadow_seed` 为种子的生成器逐列抽取，因此运行保持可复现，而 `"joint"` 产生的正是早期版本的影子特征。报告会在 `shadows` 和 `shadow_seed` 旁边记录 `shadow_mode`；没有影子特征的报告没有 `shadow_mode`。`FeatureRanker` 和 `ConsensusSelector` 都接受它。

[校准研究](../../calibration.md)没有发现支持或反对 `independent` 的证据：共线来源的召回率提升在随机种子噪声之内，因此默认值仍为 `joint`。该模式会改变哪个影子最强，因此两种模式的结果可能不同。

```python
rng = np.random.default_rng(0)
x0 = rng.normal(size=400)  # x1 below is a near copy of x0; x2 is noise
Xc = np.column_stack([x0, x0 + 0.05 * rng.normal(size=400), rng.normal(size=400)])
yc = x0 + 0.5 * rng.normal(size=400)

for mode in ("joint", "independent"):
    ranker = FeatureRanker(
        task="reg", group="fast", shadows=True, shadow_seed=0, shadow_mode=mode, verbose=False
    )
    report = ranker.rank_resampled(
        Xc, yc, feature_names=["x0", "x1", "x2"], scoring="r2", n_repeats=3, random_state=0,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
    )
    shares = {r["feature_group"]: round(r["above_shadow_share"], 2) for r in report["stability"]}
    print(report["shadow_mode"], shares)
```

```text
joint {'x0': 1.0, 'x1': 1.0, 'x2': 0.1}
independent {'x0': 0.96, 'x1': 1.0, 'x2': 0.1}
```

在这份数据上，两种模式的结果一致，只是 `x0` 有一点小的偏移。

## 证据

排名说明哪些特征组重要，但不说明为什么。设置 `evidence=True` 后，`fit()` 还会保留训练数据（作为它自己的副本，因此之后修改你的数组不会改变证据），而 `rank_features()` 会加入关于排名背后机制的证据。

- **训练得分与过拟合差距。** 每个模型得到 `train_score`（在它所拟合的数据上用同一个评分器得到的得分）和 `overfit_gap = train_score - evaluation_score`。差距大意味着模型在训练数据上的得分远好于评估数据。死记硬背是原因之一；训练数据与评估数据之间的偏移，或评估样本过小，也可能产生同样的差距，因此请先检查这些。
- **相关的特征组。** `evidence.correlation` 列出这样的特征组对：其成员列中最强的 Spearman 相关性在训练数据上至少为 0.5，并附带符号。置换重要性可能在相关的特征组之间分摊功劳；可以考虑把它们放进同一个 `feature_groups` 条目。类别列不包含在内。
- **SHAP 摘要。** 需要 `pip install 'featranker[shap]'`（shap 0.49 或更新；更旧的 shap 被视同未安装）。`models[name]["shap"]` 给出每个特征组的 `mean_abs`（平均绝对 SHAP 值，对该特征组的各列求和）和 `direction`（单列特征组的取值与其 SHAP 值之间的 Spearman 相关性；为正表示取值越高，输出被推得越高）。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(
    task="reg",
    models={
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "forest": RandomForestRegressor(n_estimators=50, random_state=0),
        "tree": DecisionTreeRegressor(random_state=0),
    },
    evidence=True,
    verbose=False,
)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
report = ranker.rank_features(X_eval, y_eval, scoring="r2", random_state=0)

for name, model in report["models"].items():
    print(f"{name:>6}  train {model['train_score']:.2f}  eval {model['evaluation_score']:.2f}"
          f"  overfit gap {model['overfit_gap']:.2f}")

for pair in report["evidence"]["correlation"]["pairs"][:3]:
    print(pair["groups"], f"rho {pair['rho']:.2f}")

for group, effect in report["models"]["ridge"]["shap"]["groups"].items():
    print(f"{group:>4}  mean|SHAP| {effect['mean_abs']:5.1f}  direction {effect['direction']:+.2f}")

print([f["code"] for f in report.summary()])
```

```text
 ridge  train 0.55  eval 0.39  overfit gap 0.16
forest  train 0.92  eval 0.29  overfit gap 0.63
  tree  train 1.00  eval -0.13  overfit gap 1.13
['s1', 's2'] rho 0.87
['s3', 's4'] rho -0.81
['s2', 's4'] rho 0.65
 age  mean|SHAP|   1.9  direction -1.00
 sex  mean|SHAP|   9.1  direction -1.00
 bmi  mean|SHAP|  21.4  direction +1.00
  bp  mean|SHAP|  10.7  direction +1.00
  s1  mean|SHAP|  14.3  direction -1.00
  s2  mean|SHAP|   5.4  direction +1.00
  s3  mean|SHAP|   4.5  direction -1.00
  s4  mean|SHAP|   2.5  direction +1.00
  s5  mean|SHAP|  24.5  direction +1.00
  s6  mean|SHAP|   0.8  direction +1.00
['few_families', 'tied_groups', 'tied_groups', 'within_noise', 'overfit', 'correlated_groups']
```

未剪枝的树完美拟合了它的训练数据，在评估数据上却不如哑基线（dummy baseline），森林也损失了大部分训练优势，因此 `summarize()` 对两者都增加了 `overfit` 发现。`s1`/`s2` 和 `s3`/`s4` 相关，因此它们的置换重要性分摊功劳（`correlated_groups`）。对于岭回归模型，`s5` 和 `bmi` 把预测向上推，且 `mean_abs` 最大。

### 解读 SHAP 值

- `mean_abs` 和 `direction` 最多在 1000 个均匀间隔的评估行上计算，因此不依赖 `random_state`。对于线性模型，背景数据最多为 1000 个均匀间隔的训练行。
- `mean_abs` 采用解释器输出的尺度，这并不总是 `predict()` 的尺度。对于线性模型，它是线性预测量，因此带有连接函数时（例如 `PoissonRegressor`），它以连接函数的单位表示，如对数单位。对于树模型，它是原始输出：回归为预测值；分类器为对数几率、间隔或概率。请在同一个模型内比较特征组，而不要在模型之间比较。
- 对于多分类模型，`mean_abs` 是对各类别取平均，`direction` 为 `null`。对于多列特征组和常数特征，`direction` 也为 `null`。对于二分类，被解释的输出是正类，即排序后的第二个标签。
- 对于线性模型，当系数不为零时，`direction` 是系数的符号，即使 `mean_abs` 接近 0 也是如此。系数为零（例如丢掉了该列的 `Lasso`）时给出 `null`。

```python
from sklearn.linear_model import Lasso, PoissonRegressor

rng = np.random.default_rng(0)
Xp = rng.normal(size=(400, 3))
yp = rng.poisson(np.exp(4.0 + 1.0 * Xp[:, 0] + 0.5 * Xp[:, 1]))
Xp_train, Xp_eval, yp_train, yp_eval = train_test_split(Xp, yp, test_size=0.3, random_state=0)

ranker = FeatureRanker(
    task="reg",
    models={"poisson": PoissonRegressor(), "lasso": Lasso(alpha=1e3)},
    evidence=True,
    verbose=False,
)
ranker.fit(Xp_train, yp_train, feature_names=["a", "b", "c"])
report = ranker.rank_features(Xp_eval, yp_eval, scoring="r2", random_state=0)

prediction = ranker.trained_models["poisson"].predict(Xp_eval)
print(f"Poisson predictions span {prediction.min():.1f} to {prediction.max():.1f}")
for name in ("poisson", "lasso"):
    for group, effect in report["models"][name]["shap"]["groups"].items():
        direction = effect["direction"]
        shown = "null" if direction is None else f"{direction:+.2f}"
        print(f"{name:>7} {group}  mean|SHAP| {effect['mean_abs']:6.2f}  direction {shown}")
```

```text
Poisson predictions span 4.9 to 1429.9
poisson a  mean|SHAP|   0.69  direction +1.00
poisson b  mean|SHAP|   0.44  direction +1.00
poisson c  mean|SHAP|   0.00  direction -1.00
  lasso a  mean|SHAP|   0.00  direction null
  lasso b  mean|SHAP|   0.00  direction null
  lasso c  mean|SHAP|   0.00  direction null
```

Poisson 的 `mean_abs` 值以对数单位表示，而预测值的范围是 4.9 到 1429.9。带有大惩罚项的 `Lasso` 所有系数都为零，因此它的 `mean_abs` 为 0，`direction` 为 `null`。

### 哪些模型有 SHAP

当模型是线性模型（`LinearExplainer`），或是 SHAP 的树解释器所支持的树模型时，它才会被解释。其他模型出现在 `evidence.shap.unsupported` 中；没有与模型无关的后备方案。有多个输出的回归器，例如多分位数模型，会被列为不支持。

- 树 SHAP 覆盖 scikit-learn、xgboost、lightgbm 和 catboost 的精确类。子类或自定义树模型不受支持。
- 如果 `Pipeline` 的前面各步都是 `StandardScaler`、`MinMaxScaler`、`MaxAbsScaler`、`RobustScaler`、恒等步骤（`"passthrough"`、`None`）或不带指示列的 `SimpleImputer`，它就会被解释。任何其他步骤、这些步骤或 `Pipeline` 的子类（包括作为最后一步），以及你用 `models=` 传入的模型中的任何 `ColumnTransformer`，都会使该模型不受支持。折叠独热列只适用于内置模型。
- 对类别特征带有独热步骤的内置线性模型可以得到 SHAP，独热列按来源列求和。SHAP 行（最多 1000 行）会被稠密化到独热宽度，因此一个没有设上限、水平数极多的列，每份副本可能需要约 1000 x 水平数 x 8 字节；请使用 `max_categories`。
- 只有当线性模型（带有 `coef_` 和 `intercept_` 的模型）的最终估计器恰好是 scikit-learn 或 xgboost 的类，并且除 `covariance_estimator` 外没有取值为估计器的参数时，它才能得到 SHAP。子类、自定义或第三方估计器，或 `FrozenEstimator` 这样的包装器，都不受支持。
- 同样不受支持的还有：多分类的 xgboost `gblinear` 模型、`missing` 既不是 `None` 也不是 NaN 的 xgboost `gblinear` 模型、使用原生类别特征拟合的直方图提升模型、grow_policy 为 Lossguide 或 Depthwise 的 CatBoost 模型、类别数超过两个的 `SVC` 或 `NuSVC`，以及 `boosting_type="rf"` 的 LightGBM 模型。xgboost `gblinear` 模型把 NaN 单元格解释为 0，与该模型自身的做法一致。
- 没有可用的 `shap`（缺失、损坏或低于 0.49）时，会发出 `UserWarning`，其余证据仍会计算。

### 开销与存储

- 证据对每个模型多做一次评分，在安装了 SHAP 时还要加上 SHAP（大型森林每个模型需要数秒）。它默认关闭，不带它生成的报告不受影响。
- 以 `evidence=True` 拟合的排名器会保留自己的训练矩阵副本，因此 pickle 后的排名器包含它的训练数据：文件大小预计会增加约训练数组的大小（开启影子特征时约为两倍，因为会增加影子列）。请像对待数据一样谨慎地对待它。
- `rank_resampled()` 把这一设置传给每个划分，因此每个 `report["splits"][i]["report"]` 都带有自己的证据。没有任何内容是跨划分汇总的。
- 在命令行中，添加 `--evidence`。
