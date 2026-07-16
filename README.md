# FeatRanker

Leakage-safe permutation feature ranking across multiple scikit-learn and optional gradient-boosting models.

FeatRanker fits configured models on data supplied to `fit()` and ranks features only on separate data supplied to `rank_features()`. It does not split, preprocess, select, or cross-validate data.

> Feature rankings are evidence for feature selection and ablation experiments. They are not causal effects, biological mechanisms, or clinical importance.

## Installation

Core installation:

```bash
pip install featranker
```

Optional estimator libraries:

```bash
pip install 'featranker[xgboost]'
pip install 'featranker[lightgbm]'
pip install 'featranker[catboost]'
pip install 'featranker[all]'
```

LightGBM is available as an installation extra, but no LightGBM model is configured by default. Missing optional libraries skip only their configured models and produce warnings plus structured failure records.

Development installation:

```bash
pip install -e '.[test]'
```

Requires Python 3.10 or newer.

## Held-out regression workflow

The caller owns splitting and preprocessing. Pass only training data to `fit()` and separate evaluation data to `rank_features()`:

```python
from featranker import FeatureRanker

feature_names = [
    "age",
    "weight",
    "CYP2C9_*1/*1",
    "CYP2C9_*1/*2",
    "CYP2C9_other",
]

ranker = FeatureRanker(task="reg", group="tree")
ranker.fit(X_train, y_train, feature_names=feature_names)

report = ranker.rank_features(
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    n_repeats=20,
    random_state=42,
)
```

NumPy training matrices require `feature_names`. pandas DataFrames retain their column names. Evaluation DataFrames must have the same columns in the same order as training. Evaluation NumPy arrays reuse fitted names and must have the same column count.

FeatRanker deliberately leaves imputation, encoding, scaling, target transformation, and splitting to the caller.

## Grouped permutation

Semantic one-hot or genotype blocks can be permuted jointly:

```python
report = ranker.rank_features(
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    feature_groups={
        "CYP2C9": [
            "CYP2C9_*1/*1",
            "CYP2C9_*1/*2",
            "CYP2C9_other",
        ],
        "VKORC1": [
            "VKORC1_GG",
            "VKORC1_GA",
            "VKORC1_AA",
        ],
    },
    n_repeats=20,
    random_state=42,
)
```

Every column in a multi-column group receives the same row permutation during a repeat. This preserves relationships among group columns while breaking the group's relationship with the target. Features omitted from `feature_groups` become singleton groups.

Unknown features, duplicate membership, overlapping groups, empty groups, and group-name collisions raise clear errors.

## Validation placement and leakage

Ranking belongs inside the training portion of each validation split. For nested validation:

1. Split an outer training and outer test fold.
2. Within the outer training fold, fit FeatRanker on inner-training data.
3. Rank on inner-validation data.
4. Make feature-selection or ablation decisions using only those inner results.
5. Keep the outer test fold untouched until final evaluation.

Do not rank on final test or outer-fold data and then select features using those rankings. That leaks evaluation information into model development. FeatRanker intentionally does not implement or hide cross-validation or site-grouped splitting.

## Scoring and importance

`scoring` accepts a scikit-learn scorer name or a scorer-protocol callable:

```python
score = scorer(estimator, X, y)
```

The same scorer computes baseline and permuted scores. Importance for every repeat is:

```text
baseline_score - permuted_score
```

Positive importance means permutation reduced predictive performance. Each model's held-out baseline score is recorded as `evaluation_score`.

## Report schema

```python
{
    "evaluation_mode": "held_out",
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 20,
    "random_state": 42,
    "feature_groups": {"CYP2C9": ["CYP2C9_*1/*1", "CYP2C9_*1/*2"]},
    "models": {
        "random_forest_regressor": {
            "evaluation_score": -8.2,
            "importance": {
                "CYP2C9": {
                    "values": [1.1, 1.3],
                    "mean": 1.2,
                    "std": 0.1,
                    "rank": 1,
                }
            },
        }
    },
    "consensus": [
        {
            "feature_group": "CYP2C9",
            "median_rank": 1.0,
            "mean_rank": 1.0,
            "rank_std": 0.0,
            "n_models": 1,
        }
    ],
    "failures": [],
}
```

Raw importance values remain per model. Consensus uses ranks because raw magnitudes from heterogeneous estimators are not directly comparable. Exact equal importance means share the minimum rank; output ties are ordered by group name. Consensus sorts by median rank, mean rank, then group name.

Initialization, fit, and ranking failures contain `model`, `stage`, `error_type`, and `message`. Failed models do not receive zero importance and do not enter consensus. `fit()` raises if no model fits. `rank_features()` raises if every fitted model fails during ranking; partial success returns successful reports and failures together.

Set `random_state` for repeatable row permutations. Estimators can still be stochastic unless their own configuration also fixes a seed.

## Legacy preparation-file migration

The preparation-file factory and CLI remain available:

```python
from featranker import build_ranker

ranker = build_ranker(
    task="reg",
    group="tree",
    prep_file="./featureCalc.py",
    prep_class="prepFeature",
)
legacy_report = ranker.rankFeatures()
```

`rankFeatures()` emits `DeprecationWarning`, ranks retained training data in-sample, and records `evaluation_mode: "in_sample"`. Warnings can be suppressed, so always inspect this field. Migrate to explicit `fit(X_train, y_train, ...)` followed by `rank_features(X_eval, y_eval, scoring=...)`.

Normal `fit()` does not retain raw training data; only the legacy preparation-file path retains it for `rankFeatures()`. `run_ML()` is also deprecated.

CLI usage remains for migration:

```bash
featranker --task reg --group tree \
    --prep-file ./featureCalc.py --prep-class prepFeature \
    --output results.json
```

CLI ranking is legacy in-sample ranking and must not be used for leakage-safe feature selection.

## Model configuration

Models are declared in `featranker/importance_config.yaml` by task (`classification` or `regression`) and family (`linear` or `tree`). Each entry supplies output name, import module, estimator class, and optional constructor parameters. Missing XGBoost or CatBoost installations skip only those entries.

## Scope

FeatRanker does not provide nested cross-validation, site-grouped splitting, imputation, encoding, scaling, automatic feature selection, SHAP, a UI, or clinical-model-specific logic.

## License

MIT
