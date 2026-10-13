# FEATRANKER ▍ MODELS

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <a href="usage.md">Usage</a> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <b>Models</b></sub>

Model entries come from [`importance_config.yaml`](../featranker/importance_config.yaml). Names in the **Key** column appear in reports and structured failures. Select a family with `group="linear"`, `group="tree"`, or `group="all"` (the default).

Built-in linear models are standardized internally (a `StandardScaler` fit on training data only, inside a pipeline); tree models are not.

## ▍ Classification: 22 models

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
| Tree | `decision_tree` | `DecisionTreeClassifier` |
| Tree | `random_forest` | `RandomForestClassifier` |
| Tree | `extra_trees` | `ExtraTreesClassifier` |
| Tree | `bagging_tree` | `BaggingClassifier` |
| Tree | `adaboost` | `AdaBoostClassifier` |
| Tree | `gradient_boosting` | `GradientBoostingClassifier` |
| Tree | `hist_gradient_boosting` | `HistGradientBoostingClassifier` |
| Tree | `xgboost` | `XGBClassifier` (optional) |
| Tree | `catboost` | `CatBoostClassifier` (optional) |

## ▍ Regression: 18 models

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

## ▍ Optional estimators

| Extra | Install | Default model configured? |
|---|---|:---:|
| XGBoost | `pip install 'featranker[xgboost]'` | ✔ |
| LightGBM | `pip install 'featranker[lightgbm]'` | ✘ |
| CatBoost | `pip install 'featranker[catboost]'` | ✔ |

Missing or broken optional libraries skip only the affected models. Each skipped model produces a warning and a structured initialization failure.

> [!WARNING]
> **Known issue (CatBoost):** CatBoost 1.2.8 estimators do not define `__sklearn_tags__`. On scikit-learn 1.7, every scorer call on a CatBoost model emits a `DeprecationWarning` saying this becomes an error in scikit-learn 1.8. If that happens, CatBoost models appear in `failures` and the other models still rank. Install `featranker[catboost]` with scikit-learn < 1.8 if you need the CatBoost models.
