# Leakage-Safe Feature Ranking Design

## Goal

Refactor Featranker so callers explicitly fit configured models on training data and compute permutation rankings on separate evaluation data. Keep validation, preprocessing, and feature selection outside the package.

## Public API

```python
ranker = FeatureRanker(task="reg", group="tree")
ranker.fit(X_train, y_train, feature_names=feature_names)
report = ranker.rank_features(
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    feature_groups={"CYP2C9": ["CYP2C9_*1/*1", "CYP2C9_*1/*2"]},
    n_repeats=20,
    random_state=42,
)
```

`FeatureRanker` initializes configured estimators but does not load data or train. `fit()` accepts only training data and returns `self`. `rank_features()` always requires evaluation data; it never falls back to training data.

NumPy training input requires `feature_names`. pandas DataFrame training input uses column names and rejects a conflicting explicit list. Evaluation input must have the fitted column count; DataFrame columns must exactly match fitted names and order. Targets must be one-dimensional, non-empty, and match row counts. Featranker converts feature values to numeric arrays but does not impute, encode, scale, split, or select data.

## Legacy Compatibility

`build_ranker()` and CLI preserve preparation-file loading: they load the feature dictionary, split out `label`, construct `FeatureRanker`, and call `fit()`.

Legacy `run_ML()` remains as a deprecated alias that fits stored preparation-file data. Legacy `rankFeatures()` remains available and ranks stored training data through `rank_features()`, emitting a prominent `DeprecationWarning` that identifies the result as in-sample and shows the held-out replacement. Bare `FeatureRanker(task=...)` changes to the safe unfitted workflow; migration is documented.

## Model Lifecycle and Optional Dependencies

YAML model selection remains task/group based. Model initialization, fitting, and ranking failures are retained as structured records with stage, exception type, and message. Missing optional imports warn and skip only affected XGBoost or CatBoost models. Core models continue without those packages. LightGBM is removed from core dependencies and no LightGBM estimator is added.

If at least one model ranks successfully, return its results plus all failures. Failed models never contribute zero importance. If no model ranks successfully, raise an informative `RuntimeError` containing failure details.

## Scoring and Permutation

`scoring` accepts a scikit-learn scorer name or scorer-protocol callable with signature `(estimator, X, y) -> float`. `sklearn.metrics.get_scorer()` resolves names. Each model's baseline and permutations use the same scorer. Each repeat records:

```text
importance = baseline_score - permuted_score
```

Positive values therefore mean permutation harmed predictive performance, including for negated loss scorers.

Without `feature_groups`, every feature is a singleton. Explicit groups reject empty groups, unknown names, repeated names within a group, and membership in multiple groups. Unmentioned features become singleton groups named after the feature. Group names must be unique strings and cannot collide with an implicit singleton.

Each group/repeat draws one row permutation and applies it to every member column, preserving within-row one-hot/genotype relationships while breaking association with the target. A seeded NumPy generator makes results reproducible.

## Ranking and Report

Per-model group importance stores all repeat values, mean, population standard deviation, and rank. Exact equal means share minimum rank; otherwise descending importance determines rank. Group name provides deterministic output ordering without changing tied rank.

Consensus aggregates successful per-model ranks only and reports `median_rank`, `mean_rank`, population `rank_std`, and `n_models`. Consensus sorts by median rank, mean rank, then group name. Raw heterogeneous importance magnitudes are never averaged into the primary consensus.

```python
{
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 20,
    "random_state": 42,
    "feature_groups": {"CYP2C9": ["..."], "age": ["age"]},
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

## Tests

Fast pytest tests use synthetic arrays/DataFrames and lightweight custom estimators/scorers. Tests directly verify held-out data flow, scorer propagation, shared group permutations, group validation, singleton completion, rank consensus, deterministic seeds, partial/all failures, shape/name validation, legacy warning behavior, and core-only imports. Optional libraries are not imported by default tests.

## Documentation and Packaging

README documents held-out and grouped workflows, nested-validation placement, leakage warnings, non-causal interpretation, report schema, failures, reproducibility, optional dependencies, and migration. Add `CHANGELOG.md`, version `0.2.0`, optional dependency extras, and minimal GitHub Actions core-test CI. Build wheel and source distribution locally; do not push or publish.

## Scope

No cross-validation, grouped splitting, preprocessing, automatic selection, SHAP, UI, or clinical-model logic. Existing model configuration stays intact except dependency handling required for core-only operation.
