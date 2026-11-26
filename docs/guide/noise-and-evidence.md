# Noise and evidence

This page shows how to tell signal from noise with shadow features, the shadow hits test and family evidence, and how to see why a ranking looks the way it does with evidence and SHAP.

[Manual](../README.md) | [Validation](validation.md) | [Selection](selection.md) | [Report](../report.md) | [Reference](../reference.md)

## Shadow features

A rank says a group is more important than others, not that it carries any signal. With `shadows=True`, every model also gets a *shadow* copy of all features: the same rows in a shuffled order. The shadow block keeps feature values and within-row relations while randomly reassigning rows, so it is a randomized reference. A real group is `above_shadow` in a model when its mean importance is greater than the strongest shadow's.

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

`bmi`, `s5` and `bp` beat their shadows in every family. The two pure-noise columns are at 25% or less.

- `above_shadow_share` is family-weighted: each voting family contributes the fraction of its members in which the group beat the strongest shadow.
- `above_shadow` is a relative comparison, not proof of useful signal. If `shadow_max` is negative, even a negative real importance can beat it. Read the real mean, the shuffle spread and the evaluation score next to the share.
- It is descriptive, not a p-value. Families are not independent trials, and a share below 0.5 does not prove a group is useless. For a test across splits see the [hits test](#shadow-hits-test).
- Ranks still compare real groups only. Extra noise columns can still change how some models fit (for example forests that sample a fraction of features per split), so ranks with shadows on can differ slightly from a run without.
- **Cost.** Models train on twice the columns and every group is permuted twice: expect about double the time and memory.
- `rank_resampled()` uses the same setting. Split `i` uses `(shadow_seed + i)` modulo 2**64. Each stability row gains `above_shadow_share` (the sum of the per-split values divided by `n_splits`: a split that failed, had an empty consensus or lacks the group counts as 0), the hits test keys and the family evidence keys. `ConsensusSelector(min_shadow_share=...)` filters on this value.

> [!WARNING]
> Shadows only work if every model sees every column. A custom pipeline that drops columns, such as a positional `ColumnTransformer` with the default `remainder="drop"`, never sees the shadows, so every group trivially beats them. Shadow columns are appended after the real ones, so positional selectors of real columns still work. See [the shadow contract](custom-models.md#the-shadow-contract).

## Shadow hits test

With shadows on, `rank_resampled()` also tests, for every group, whether it beats its shadows across splits more often than chance, at no extra model fits. It is Boruta's hits test applied to your splits. In each split a group has a *hit* when its consensus `above_shadow_share` is strictly greater than 0.5. Each stability row gains:

- `shadow_hits`: the number of splits that were hits. A split where the group has no consensus row counts as a non-hit; the number of trials is always `n_splits`.
- `shadow_p_value`: the one-sided binomial probability of at least `shadow_hits` hits in `n_splits` trials with success probability 0.5.
- `shadow_q_value`: the Benjamini-Hochberg adjusted p-value across all groups.

The report has a `shadow_test` block (method, hit rule, `null_probability`, correction, `n_trials`, `min_p_value`, `calibrated: false` and a `note`). A 0.5 null probability and false-discovery-rate control are not established for arbitrary estimators or overlapping splits, so do not read these values as calibrated significance.

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

Limits to keep in mind:

- **Uncalibrated null.** The test compares against 0.5, Boruta's choice, but nothing here shows that a group with no effect hits in half the splits or fewer. A custom estimator can break it: a model whose importance for a pure-noise column is reliably above its shadow's gives small q-values on noise. Use `shadow_p_value` and `shadow_q_value` to order groups, not as probabilities or error rates.
- **Dependent splits.** Cross-validation training sets overlap, so the splits are not independent trials and the binomial tail is not exact even for well-behaved models.
- **Low power with few splits.** The smallest p-value `n_splits` allows is `0.5 ** n_splits` (`shadow_test["min_p_value"]`): 0.031 for the default `cv=5`. At `cv=5` only a group that hits in all 5 splits reaches it (4 of 5 gives p = 0.19), and Benjamini-Hochberg multiplies it by the number of groups over its rank. With 12 groups of which two hit 5 of 5, both get q = 0.19. Use `RepeatedKFold` or another splitter with 10 or more splits. No warning is raised for fewer, except that a `ConsensusSelector` with `max_shadow_q` that selects nothing adds `n_splits` and the smallest possible p-value to its warning.
- **What a small q means.** The group beats shuffled noise across splits. It is not proof of a causal effect or of usefulness for a new target.
- **Not for single-split reports.** `rank_features()` has one trial, so it gets no test.

In the [calibration study](../calibration.md), on null data (1000 rows, 10 independent features, 10 `RepeatedKFold` splits, built-in `fast` and `tree` models) no group had `shadow_p_value <= 0.05`, so the values were conservative there. That is partly built in: with 10 splits the test's real level is 0.0107 (at least 9 hits). The test also has low power: `max_shadow_q=0.05` recovered the `c*d` interaction in 0 to 75% of seeds.

`ConsensusSelector(max_shadow_q=0.05)` keeps the groups with `shadow_q_value <= 0.05` ([Selection](selection.md#consensusselector-in-a-pipeline)). It is a heuristic filter, not a significance level, and it is **experimental**: `fit()` emits a `UserWarning` saying so, and the report's `shadow_test` block has `"experimental": true`.

## Family evidence

`above_shadow_share` averages one vote per model family, so it measures agreement across families. A group that only some families can model, such as a pure interaction that additive models cannot see, can be found by those families and still average below 0.5. With shadows on, the report also says which families found each group:

- Each consensus row gains `family_above_shadow`: for each voting family, the fraction of its voting members whose importance beat that model's strongest shadow. `above_shadow_share` is the mean of these values; keys are sorted by family name.
- In a resampled report each stability row gains `family_above_shadow_share` (for each family that voted in at least one split, the sum of its per-split fractions divided by `n_splits`; a split where the family did not vote counts as 0) and `shadow_families`, the number of families whose share is strictly above 0.5.

The example plants an interaction (`x1` and `x2` matter only together, through a logistic link, plus one linear feature `x0` and two noise columns):

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

The four additive families (logistic, linear margin, discriminant, naive Bayes) do not put `x2` above their shadows in most splits, so its share stays below 0.5 and the share rule drops it (it keeps `x1` only because the share is 0.53). Bagging, boosting and the single tree can represent the interaction, so three families pass.

`ConsensusSelector(min_shadow_families=N)` keeps groups with `shadow_families >= N`. It answers "do at least `N` families find signal?", turns shadows on, and combines with the other rules (all filters apply, then `k`).

- **When to use it.** When you suspect interactions or effects that only some model types capture.
- **Choosing `N`.** `N=1` keeps a group any single family finds, so one family fitting noise (a single decision tree, say) can pass noise on its own; in the output above `x4`, a noise column, already has `shadow_families` of 1. Requiring two families means two kinds of model must agree. The [calibration study](../calibration.md) measured this on synthetic data. Its preregistered rule recommends `N=1`: `N=1` and `N=2` each selected at most 1% of noise groups on null data, and with `group="fast"` both found the `c*d` interaction in every seed. Post hoc, `N=1` kept some noise in the signal study (up to 0.4 of 4 noise features per run) and `N=2` at most 0.1, with no loss of recall. Use `N=1` if recall matters more and `N=2` if false selections matter more. Avoid `N=3` with `group="fast"` in classification (it found `c*d` in only 0 to 15% of seeds). One synthetic design with built-in models: treat it as a heuristic, like `max_shadow_q`.
- **Which families count.** Every voting family counts equally, including families that cannot represent the effect you care about. Restrict them with `group` if you want to. With your own `models=`, each model is its own family unless `families=` maps it. With `group="tree"`, regression and classification each have three families (bagging, boosting, single tree), so `N=2` is a majority there. `"fast"` has seven classification and five regression families.

> [!NOTE]
> On real data `min_shadow_share=0.5` can be too strict; see the [real-data study](../realdata.md).

## Shadow mode

By default (`shadow_mode="joint"`) the shadow block is the rows in one shuffled order applied to every column, so two collinear real columns stay exactly as collinear in the shadows. `shadow_mode="independent"` shuffles each shadow column with its own row order: each is still a permutation of its real column, but relations between columns are broken. The orders are drawn column by column from one generator seeded with `shadow_seed`, so runs stay reproducible, and `"joint"` produces exactly the shadows of earlier versions. The report records `shadow_mode` next to `shadows` and `shadow_seed`; reports without shadows have no `shadow_mode`. `FeatureRanker` and `ConsensusSelector` both accept it.

The [calibration study](../calibration.md) found no evidence for or against `independent`: gains in recall of a collinear source were within seed noise, so the default stays `joint`. The mode changes which shadow is strongest, so results can differ between modes.

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

The two modes agree on this data apart from a small shift in `x0`.

## Evidence

A rank says which groups matter, not why. With `evidence=True`, `fit()` also keeps the training data (as its own copy, so changing your arrays afterwards does not change the evidence) and `rank_features()` adds evidence about the mechanism behind the ranking.

- **Train score and overfit gap.** Each model gets `train_score` (the same scorer on the data it was fit on) and `overfit_gap = train_score - evaluation_score`. A large gap means the model scores much better on its training data. Memorization is one cause; a shift between training and evaluation data, or a small evaluation sample, can produce the same gap, so check those first.
- **Correlated groups.** `evidence.correlation` lists pairs of feature groups whose strongest member-column Spearman correlation on the training data is at least 0.5, with its sign. Permutation importance can split credit between correlated groups; consider putting them in one `feature_groups` entry. Categorical columns are left out.
- **SHAP summary.** Needs `pip install 'featranker[shap]'` (shap 0.49 or newer; an older shap is treated like a missing one). `models[name]["shap"]` gives every group's `mean_abs` (mean absolute SHAP value, summed over the group's columns) and `direction` (Spearman correlation between a single-column group's value and its SHAP value; positive means higher values push the output up).

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

The unpruned tree fits its training data perfectly and does worse than the dummy on the evaluation data, and the forest lost most of its training advantage, so `summarize()` adds an `overfit` finding for both. `s1`/`s2` and `s3`/`s4` are correlated, so their permutation importances share credit (`correlated_groups`). For the ridge model, `s5` and `bmi` push predictions up and carry the largest `mean_abs`.

### Reading SHAP values

- `mean_abs` and `direction` are computed on at most 1000 evaluation rows, evenly spaced, so they do not depend on `random_state`. For linear models the background is at most 1000 evenly spaced training rows.
- `mean_abs` is on the explainer's output scale, which is not always the scale of `predict()`. For linear models it is the linear predictor, so with a link function (for example `PoissonRegressor`) it is in link units, such as log units. For tree models it is the raw output: the prediction for regression; log-odds, margin or probability for classifiers. Compare groups within one model, not between models.
- For multiclass models `mean_abs` is averaged over classes and `direction` is `null`. `direction` is also `null` for multi-column groups and constant features. For binary classification the explained output is the positive class, the second label in sorted order.
- For linear models `direction` is the sign of the coefficient when it is not zero, even when `mean_abs` is near 0. A zero coefficient (for example a `Lasso` that dropped the column) gives `null`.

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

The Poisson `mean_abs` values are in log units while the predictions span 4.9 to 1429.9. The `Lasso` with a large penalty has all coefficients at zero, so its `mean_abs` is 0 and its `direction` is `null`.

### Which models get SHAP

A model is explained when it is a linear model (`LinearExplainer`) or a tree model that SHAP's tree explainer supports. Other models appear in `evidence.shap.unsupported`; there is no model-agnostic fallback. A regressor with several outputs, such as a multi-quantile model, is listed as unsupported.

- Tree SHAP covers exact scikit-learn, xgboost, lightgbm and catboost classes. A subclass or custom tree model is unsupported.
- A `Pipeline` is explained if its earlier steps are all `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, identity steps (`"passthrough"`, `None`) or a `SimpleImputer` without indicator columns. Any other step, a subclass of these steps or of `Pipeline` (also as the final step), and any `ColumnTransformer` in a model you pass with `models=` make the model unsupported. Folding one-hot columns is for the built-in models only.
- Built-in linear models with a one-hot step for categorical features get SHAP, with the one-hot columns summed per source column. The SHAP rows (at most 1000) are densified to the one-hot width, so an uncapped column with very many levels can need about 1000 x levels x 8 bytes per copy; use `max_categories`.
- A linear model (one with `coef_` and `intercept_`) gets SHAP only if its final estimator is exactly a scikit-learn or xgboost class with no estimator-valued parameter other than `covariance_estimator`. A subclass, a custom or third-party estimator, or a wrapper such as `FrozenEstimator` is unsupported.
- Also unsupported: a multiclass xgboost `gblinear` model, an xgboost `gblinear` model whose `missing` is neither `None` nor NaN, a histogram boosting model fitted with native categorical features, a CatBoost model with grow_policy Lossguide or Depthwise, an `SVC` or `NuSVC` with more than two classes, and a LightGBM model with `boosting_type="rf"`. An xgboost `gblinear` model explains NaN cells as 0, as the model does.
- Without a usable `shap` (missing, broken or older than 0.49) a `UserWarning` is raised and the rest of the evidence is still computed.

### Cost and storage

- Evidence costs one extra scoring pass per model, plus SHAP when installed (seconds per model for a large forest). It is off by default, and reports made without it are unchanged.
- A ranker fitted with `evidence=True` keeps its own copy of the training matrix, so a pickled ranker contains its training data: expect the file to grow by roughly the size of the training arrays (about twice that with shadows, which adds the shadow columns). Treat it with the same care as the data.
- `rank_resampled()` passes the setting to every split, so each `report["splits"][i]["report"]` carries its own evidence. Nothing is aggregated across splits.
- On the command line, add `--evidence`.
