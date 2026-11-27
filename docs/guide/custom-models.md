# Custom models

This page shows how to rank with your own estimators and pipelines, group them into families, select columns by name, and run models in parallel.

[Manual](../README.md) | [Models and scoring](models-and-scoring.md) | [Data](data.md) | [Reference](../reference.md)

## Your own models

Pass `models` to rank with your own estimators instead of the built-in set. Use it when you need a specific estimator or a pipeline with your own preprocessing.

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

- Estimators are cloned at `fit()`, so yours are never modified. A `Pipeline` keeps its preprocessing fit on the training rows only.
- Names must be non-empty strings other than `dummy_baseline`. `models` cannot be combined with a `group` other than `"all"`.
- Every model is fit and scored on read-only views of the shared feature, target and weight arrays. An estimator that writes into its input raises, which is recorded as a `fit` (or `ranking`) failure for that model alone. scikit-learn estimators that copy first, such as `LinearRegression(copy_X=False)`, work. Some, such as `Ridge`, copy a read-only `X` once more while fitting, which can raise peak memory by about a sixth on large data.

> [!IMPORTANT]
> By default your estimators receive a float NumPy array, even when you pass a DataFrame. In a pipeline select columns by position, for example `ColumnTransformer([("scale", StandardScaler(), [0, 1])])`, or use `model_input="dataframe"` (below). Categorical values must be numbers by then: encode them yourself, or pass `categorical=` and your estimators receive integer codes ([Data](data.md#categorical-features)). With `missing="impute"` your estimators receive NaN unchanged ([Data](data.md#missing-values)).

## Families

Each custom model is its own family by default, and each family casts one vote. Three near-identical ridge models are three votes and can outvote a different kind of model. That protection holds only for the built-in families.

Say which models belong together with `families={model_name: family}`. Models mapped to one family vote once, as the median of their members. The report's `family` field shows the mapping.

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

- A name that is not a configured model raises `ValueError` in `fit()` (and in `rank_resampled()` and `ConsensusSelector.fit()`, before any split).
- An empty or non-string name or family, or a value that is not a mapping, raises `ValueError` in the constructor. Any mapping is accepted and copied to a dict.
- `families` also takes built-in model names, to regroup them. The name of a built-in model that failed to initialise is accepted. A built-in name outside the chosen `group` or `task` raises.
- `ConsensusSelector(families=...)` passes it on.

## Columns by name

`model_input="dataframe"` fits, scores and permutes every model (built-in and custom) on a pandas DataFrame. Its columns are the feature names in order, then, with `shadows=True`, `shadow__<name>` for each shadow column. That is the normal scikit-learn way to preprocess mixed data: `ColumnTransformer([("num", StandardScaler(), ["age", "income"])])`.

- Importances, scores, train scores and the SHAP summary equal those of `"array"` mode for models that do not depend on column names. The DataFrame is rebuilt from the permuted array for each scoring call, which costs a little time on wide data.
- The report records `model_input: "dataframe"`; the key is absent in the default mode.
- `rank_resampled()` and `ConsensusSelector(model_input=...)` use it in every split. The CLI keeps array input.
- Without pandas, `model_input="dataframe"` raises `ImportError` at construction.
- Feature names may not start with `shadow__` when shadows are on.

### The shadow contract

With shadows on, a custom pipeline must keep the shadow columns. A model that never sees them has shadow importances of zero, so every real feature "beats" the shadows.

After fitting, FeatRanker works out which input columns each model uses, in both input modes. It follows a `Pipeline` into its first step, a `ColumnTransformer` into each branch, a `FeatureUnion` into each part, and a fitted selector (`SelectKBest`, `SelectFromModel`, ...) to the columns it keeps. A later `Pipeline` step that is or contains a selector, a `FeatureUnion`, a `ColumnTransformer` or an opaque transformer is checked only when every earlier step keeps the columns one for one (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, a `SimpleImputer` that adds no indicator and drops no column). After any other step it counts as unverified.

If the transformers and `remainder` leave a shadow column unused, or the columns cannot be worked out, the model is left out of every shadow comparison in that report: no `above_shadow` flags, no `shadow_max`, not counted in `family_above_shadow` or `above_shadow_share`. `failures` gets an entry with stage `"shadows"` and `error_type` `"ShadowColumnsDropped"`. Its importances and its ranking vote are unaffected. The message says whether the model "does not use every shadow column" or "could not be verified". Use `remainder="passthrough"`, or list the `shadow__` columns, to keep it in the shadow evidence.

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

Both models vote in the consensus, but only `keeps_shadows` supplies shadow evidence. `summarize()` reports the `"shadows"` stage as a warning.

> [!WARNING]
> The check cannot see every way of dropping columns.
> - A `FunctionTransformer` with a function, or a custom transformer used to transform input columns, is opaque. An intermediate pipeline step still acts as a transformer when it also exposes `predict()`. It may drop columns, so the model counts as unverified.
> - A selector, `FeatureUnion` or `ColumnTransformer` wrapped in a meta-estimator (`TransformedTargetRegressor`, `BaggingRegressor`, `GridSearchCV`, ...) is unverified.
> - scikit-learn's own transformers (scalers, encoders, imputers) and final predictors from any library are assumed to use every column they receive. Any other estimator that ignores some inputs is assumed to use all of them, and is your responsibility.
> - XGBoost rejects feature names containing `[`, `]` or `<` in dataframe mode; those fit failures appear in `failures`. Rename the columns or use array mode.

## Parallel models

`n_jobs=2` runs the per-model work in `fit()` and `rank_features()` in parallel with joblib. `None` or `1` is sequential and `-1` uses all cores. Anything else raises an error naming `n_jobs`.

- Reports match `n_jobs=None`, with models in the same order, when every estimator is deterministic or seeded, the scorer is deterministic and `random_state` is a fixed integer. Every built-in model with randomness is seeded (`random_state=42`).
- An unseeded custom estimator, such as `RandomForestRegressor()` without `random_state`, can give different reports on every run, parallel or not.
- With `shadows=True`, reproducibility also needs a fixed integer `shadow_seed`. Fixing `random_state` alone does not seed shadow construction.
- Per-model failures are recorded as in the sequential path.
- With any parallel setting, custom scorers and estimators are sent to worker processes, so everything they hold must be serializable. Lambdas and local functions are fine; locks and connections are not. Use `n_jobs=None` if you get a pickling error.
