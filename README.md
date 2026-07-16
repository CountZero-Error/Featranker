<div align="center">
  <h1>FeatRanker</h1>
  <p><strong>Leakage-safe, model-agnostic feature ranking for trustworthy ML workflows.</strong></p>
  <p>Rank feature importance across multiple ML models using permutation importance.</p>
  <p>Supports 30+ scikit-learn, XGBoost, and CatBoost classifiers &amp; regressors with CLI and Python API.</p>
  <p>
    <a href="https://pypi.org/project/featranker/"><img src="https://img.shields.io/pypi/v/featranker?style=flat-square" alt="PyPI"></a>
    <a href="https://pypi.org/project/featranker/"><img src="https://img.shields.io/pypi/pyversions/featranker?style=flat-square" alt="Python"></a>
    <a href="https://github.com/CountZero-Error/Featranker/actions/workflows/tests.yml"><img src="https://github.com/CountZero-Error/Featranker/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="MIT License"></a>
  </p>
</div>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#leakage-safe-validation">Validation safety</a> ·
  <a href="#available-models">Models</a> ·
  <a href="#result-schema">Result schema</a>
</p>

---

FeatRanker trains configurable estimators on explicit training data, measures permutation importance on separate evaluation data, and combines model-specific rankings into a robust consensus.

Built for feature-selection and ablation workflows where leakage control, semantic feature groups, and model failures must remain visible.

> [!NOTE]
> Feature rankings are predictive evidence. They are not causal effects, biological mechanisms, or clinical importance.

## Why FeatRanker?

| Held-out by design | Semantic feature groups | Rank consensus |
|---|---|---|
| `fit()` receives training data. `rank_features()` requires separate evaluation data. | Jointly permute one-hot, encoded category, or related feature blocks with one row permutation. | Aggregate per-model ranks instead of incomparable raw importance magnitudes. |

Also included:

- 41 configured model entries: 23 for classification and 18 for regression;
- classification and regression model families;
- scikit-learn scorer names or scorer-protocol callables;
- per-model evaluation scores and raw repeat importances;
- structured initialization, fit, and ranking failures;
- deterministic permutations and tie handling.

## Installation

Install the core package:

```bash
pip install featranker
```

Core dependencies are NumPy, scikit-learn, PyYAML, and tqdm. Python 3.10 or newer is required.

Install optional estimator libraries only when needed:

```bash
pip install 'featranker[xgboost]'
pip install 'featranker[lightgbm]'
pip install 'featranker[catboost]'
pip install 'featranker[all]'
```

## Quick start

The caller owns splitting and preprocessing. Fit on training data, then rank on a separate evaluation set.

```python
from featranker import FeatureRanker

feature_names = [
    "feature_a",
    "feature_b",
    "segment_red",
    "segment_blue",
    "segment_other",
]

feature_groups = {
    "segment": [
        "segment_red",
        "segment_blue",
        "segment_other",
    ]
}

ranker = FeatureRanker(task="reg", group="tree")
ranker.fit(X_train, y_train, feature_names=feature_names)

report = ranker.rank_features(
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    feature_groups=feature_groups,
    n_repeats=20,
    random_state=42,
)

print(report["consensus"])
```

NumPy training matrices require `feature_names`. pandas DataFrames retain their column names.

## Leakage-safe validation

> [!IMPORTANT]
> Ranking belongs inside the training portion of validation. Never use outer-fold or final-test rankings to choose features.

For nested validation:

1. Create outer training and outer test folds.
2. Fit FeatRanker on inner-training data from the outer training fold.
3. Rank features on inner-validation data.
4. Make selection or ablation decisions from inner results only.
5. Keep the outer test fold untouched until final evaluation.

FeatRanker does not implement cross-validation or site-grouped splitting. Those decisions remain explicit in downstream projects.

## How it works

1. **Initialize** configured models for `task="clf"` or `task="reg"`.
2. **Fit** each model on data supplied to `fit()`.
3. **Score** each successful model on held-out evaluation data.
4. **Permute** each feature group repeatedly and measure score degradation.
5. **Rank** groups within each model using mean permutation importance.
6. **Aggregate** model ranks into median- and mean-rank consensus.

Importance is always calculated with the same scorer:

```text
importance = baseline_score - permuted_score
```

Positive importance means permutation reduced predictive performance.

## Grouped permutation

Related columns can be permuted as one semantic unit:

```python
feature_groups = {
    "segment": [
        "segment_red",
        "segment_blue",
        "segment_other",
    ],
    "channel": [
        "channel_web",
        "channel_store",
        "channel_partner",
    ],
}
```

Every column in a multi-column group receives the same row permutation during a repeat. This preserves relationships within the block while breaking its relationship with the target.

Features omitted from `feature_groups` become singleton groups. Empty groups, unknown features, duplicate membership, overlaps, and group-name collisions raise errors.

## Data and scoring contract

| Input | Contract |
|---|---|
| `X_train` | Two-dimensional NumPy array or pandas DataFrame. |
| `y_train` | One-dimensional target with the same row count. |
| `feature_names` | Required for NumPy training input; inferred from DataFrame columns. |
| `X_eval` | Must match fitted feature count, names, and order. |
| `y_eval` | One-dimensional evaluation target with matching rows. |
| `scoring` | scikit-learn scorer name or callable `(estimator, X, y) -> float`. |
| `feature_groups` | Optional mapping of group names to fitted feature names. |

FeatRanker does not impute, encode, scale, split, transform targets, or select features.

Non-finite baseline, permuted, or derived importance values fail that model's ranking. They never enter successful reports or consensus.

## Result schema

The report preserves raw model evidence and keeps aggregation rank-based.

<details>
<summary><strong>View complete report structure</strong></summary>

```python
{
    "evaluation_mode": "held_out",
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 20,
    "random_state": 42,
    "feature_groups": {
        "segment": ["segment_red", "segment_blue", "segment_other"]
    },
    "models": {
        "random_forest_regressor": {
            "evaluation_score": -8.2,
            "importance": {
                "segment": {
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
            "feature_group": "segment",
            "median_rank": 1.0,
            "mean_rank": 1.0,
            "rank_std": 0.0,
            "n_models": 1,
        }
    ],
    "failures": [],
}
```

</details>

<details>
<summary><strong>Ranking, failures, and reproducibility</strong></summary>

- Raw importance values remain model-specific and are never averaged into primary consensus.
- Exact equal mean importances share the minimum rank. Group names provide deterministic output order.
- Consensus sorts by `median_rank`, `mean_rank`, then group name.
- Failures contain `model`, `stage`, `error_type`, and `message`.
- Failed models receive no zero placeholder and do not enter consensus.
- `fit()` raises if no model fits; `rank_features()` raises if every model fails ranking.
- `random_state` controls permutation reproducibility. Estimator randomness remains estimator-specific.

</details>

## Available models

Model entries come directly from [`importance_config.yaml`](featranker/importance_config.yaml). Names in the **Key** column are used in reports and structured failures.

<details>
<summary><strong>Classification models — 23 configurations</strong></summary>

| Family | Key | Estimator |
|---|---|---|
| Linear | `logistic_regression` | `LogisticRegression` |
| Linear | `logistic_regression_l1` | `LogisticRegression` |
| Linear | `logistic_regression_l2` | `LogisticRegression` |
| Linear | `logistic_regression_elasticnet` | `LogisticRegression` |
| Linear | `linear_svm` | `LinearSVC` |
| Linear | `sgd_classifier` | `SGDClassifier` |
| Linear | `ridge_classifier` | `RidgeClassifier` |
| Linear | `perceptron` | `Perceptron` |
| Linear | `passive_aggressive` | `SGDClassifier` |
| Linear | `lda` | `LinearDiscriminantAnalysis` |
| Linear | `qda` | `QuadraticDiscriminantAnalysis` |
| Linear | `naive_bayes_gaussian` | `GaussianNB` |
| Linear | `naive_bayes_bernoulli` | `BernoulliNB` |
| Linear | `pls_da` | `PLSRegression` |
| Tree | `decision_tree` | `DecisionTreeClassifier` |
| Tree | `random_forest` | `RandomForestClassifier` |
| Tree | `extra_trees` | `ExtraTreesClassifier` |
| Tree | `bagging_tree` | `BaggingClassifier` |
| Tree | `adaboost` | `AdaBoostClassifier` |
| Tree | `gradient_boosting` | `GradientBoostingClassifier` |
| Tree | `hist_gradient_boosting` | `HistGradientBoostingClassifier` |
| Tree | `xgboost` | `XGBClassifier` (optional) |
| Tree | `catboost` | `CatBoostClassifier` (optional) |

</details>

<details>
<summary><strong>Regression models — 18 configurations</strong></summary>

| Family | Key | Estimator |
|---|---|---|
| Linear | `linear_regression` | `LinearRegression` |
| Linear | `ridge_regression` | `Ridge` |
| Linear | `lasso_regression` | `Lasso` |
| Linear | `elasticnet_regression` | `ElasticNet` |
| Linear | `elasticnet_cv_regression` | `ElasticNetCV` |
| Linear | `pls_regression` | `PLSRegression` |
| Linear | `huber_regression` | `HuberRegressor` |
| Linear | `ransac_regression` | `RANSACRegressor` |
| Linear | `kernel_ridge_regression` | `KernelRidge` |
| Linear | `svr_regression` | `SVR` |
| Tree | `decision_tree_regressor` | `DecisionTreeRegressor` |
| Tree | `random_forest_regressor` | `RandomForestRegressor` |
| Tree | `extra_trees_regressor` | `ExtraTreesRegressor` |
| Tree | `adaboost_regressor` | `AdaBoostRegressor` |
| Tree | `gradient_boosting_regressor` | `GradientBoostingRegressor` |
| Tree | `hist_gradient_boosting_regressor` | `HistGradientBoostingRegressor` |
| Tree | `xgboost_regressor` | `XGBRegressor` (optional) |
| Tree | `catboost_regressor` | `CatBoostRegressor` (optional) |

</details>

## Optional estimators

| Extra | Installation | Default model configured? |
|---|---|---|
| XGBoost | `pip install 'featranker[xgboost]'` | Yes |
| LightGBM | `pip install 'featranker[lightgbm]'` | No |
| CatBoost | `pip install 'featranker[catboost]'` | Yes |

Missing or broken optional libraries skip only affected models. Each skipped model produces a warning and structured initialization failure.

Model definitions live in [`featranker/importance_config.yaml`](featranker/importance_config.yaml).

## Legacy migration

The preparation-file factory and CLI remain available for migration.

```python
from featranker import build_ranker

ranker = build_ranker(
    task="reg",
    group="tree",
    prep_file="./featureCalc.py",
    prep_class="prepFeature",
)

legacy_report = ranker.rankFeatures()
assert legacy_report["evaluation_mode"] == "in_sample"
```

> [!WARNING]
> `rankFeatures()` performs deprecated in-sample ranking. Use `fit()` and `rank_features()` with separate data for leakage-safe work.

Warnings can be suppressed, so inspect `evaluation_mode`. Normal `fit()` does not retain training data; only the legacy prep-file path retains it for `rankFeatures()`.

`run_ML()` is deprecated. The CLI also uses legacy in-sample ranking.

## Development

```bash
git clone https://github.com/CountZero-Error/Featranker.git
cd Featranker
pip install -e '.[test]'
python -m pytest -q
python -m build
```

Continuous integration runs the core test suite on supported Python versions without optional estimator libraries.

## Scope and interpretation

FeatRanker provides evidence for predictive feature selection and ablation. Results depend on the evaluation sample, scorer, estimator behavior, correlated predictors, and permutation design.

FeatRanker intentionally does not provide:

- nested cross-validation or site-grouped splitting;
- imputation, encoding, scaling, or target transformation;
- automatic feature selection;
- SHAP or causal interpretation;
- clinical-model-specific logic or a user interface.

## License

Released under the [MIT License](LICENSE).
