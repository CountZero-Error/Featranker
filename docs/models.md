# FEATRANKER ▍ MODELS

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <a href="usage.md">Usage</a> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <b>Models</b> &nbsp;│&nbsp; <a href="api.md">API</a></sub>

Model entries come from [`importance_config.yaml`](../featranker/importance_config.yaml). Names in the **Key** column appear in reports and structured failures. Select a **Group** with `group="linear"` (the Linear rows: linear, discriminant, naive Bayes, and kernel models), `group="tree"`, or `group="all"` (the default). `group="fast"` runs only the models marked in the **Fast** column: one representative per candidate family, 7 for classification and 5 for regression. Normally the same families vote as in `"all"`, but a representative can fail or score below the dummy baseline and then its family does not vote, while in `"all"` another member of that family might (see [Speed](usage.md#speed)). The **Family** column groups close variants; each family casts one vote in the consensus (see [Report](report.md#reading-the-report)).

Built-in linear models are standardized internally (a `StandardScaler` fit on training data only, inside a pipeline); tree models are not.

## ▍ Classification: 22 models

| Group | Family | Key | Estimator | Fast |
|---|---|---|---|:---:|
| Linear | `logistic` | `logistic_regression` | `LogisticRegression` | ✔ |
| Linear | `logistic` | `logistic_regression_l1` | `LogisticRegression` |  |
| Linear | `logistic` | `logistic_regression_l2` | `LogisticRegression` |  |
| Linear | `logistic` | `logistic_regression_elasticnet` | `LogisticRegression` |  |
| Linear | `linear_margin` | `linear_svm` | `LinearSVC` |  |
| Linear | `linear_margin` | `sgd_classifier` | `SGDClassifier` |  |
| Linear | `linear_margin` | `ridge_classifier` | `RidgeClassifier` | ✔ |
| Linear | `linear_margin` | `perceptron` | `Perceptron` |  |
| Linear | `linear_margin` | `passive_aggressive` | `SGDClassifier` |  |
| Linear | `discriminant` | `lda` | `LinearDiscriminantAnalysis` | ✔ |
| Linear | `discriminant` | `qda` | `QuadraticDiscriminantAnalysis` |  |
| Linear | `naive_bayes` | `naive_bayes_gaussian` | `GaussianNB` | ✔ |
| Linear | `naive_bayes` | `naive_bayes_bernoulli` | `BernoulliNB` |  |
| Tree | `single_tree` | `decision_tree` | `DecisionTreeClassifier` | ✔ |
| Tree | `bagging` | `random_forest` | `RandomForestClassifier` | ✔ |
| Tree | `bagging` | `extra_trees` | `ExtraTreesClassifier` |  |
| Tree | `bagging` | `bagging_tree` | `BaggingClassifier` |  |
| Tree | `boosting` | `adaboost` | `AdaBoostClassifier` |  |
| Tree | `boosting` | `gradient_boosting` | `GradientBoostingClassifier` |  |
| Tree | `boosting` | `hist_gradient_boosting` | `HistGradientBoostingClassifier` | ✔ |
| Tree | `boosting` | `xgboost` | `XGBClassifier` (optional) |  |
| Tree | `boosting` | `catboost` | `CatBoostClassifier` (optional) |  |

## ▍ Regression: 18 models

| Group | Family | Key | Estimator | Fast |
|---|---|---|---|:---:|
| Linear | `linear` | `linear_regression` | `LinearRegression` |  |
| Linear | `linear` | `ridge_regression` | `Ridge` | ✔ |
| Linear | `linear` | `lasso_regression` | `Lasso` |  |
| Linear | `linear` | `elasticnet_regression` | `ElasticNet` |  |
| Linear | `linear` | `elasticnet_cv_regression` | `ElasticNetCV` |  |
| Linear | `linear` | `pls_regression` | `PLSRegression` |  |
| Linear | `linear` | `huber_regression` | `HuberRegressor` |  |
| Linear | `linear` | `ransac_regression` | `RANSACRegressor` |  |
| Linear | `linear` | `kernel_ridge_regression` | `KernelRidge` |  |
| Linear | `kernel` | `svr_regression` | `SVR` | ✔ |
| Tree | `single_tree` | `decision_tree_regressor` | `DecisionTreeRegressor` | ✔ |
| Tree | `bagging` | `random_forest_regressor` | `RandomForestRegressor` | ✔ |
| Tree | `bagging` | `extra_trees_regressor` | `ExtraTreesRegressor` |  |
| Tree | `boosting` | `adaboost_regressor` | `AdaBoostRegressor` |  |
| Tree | `boosting` | `gradient_boosting_regressor` | `GradientBoostingRegressor` |  |
| Tree | `boosting` | `hist_gradient_boosting_regressor` | `HistGradientBoostingRegressor` | ✔ |
| Tree | `boosting` | `xgboost_regressor` | `XGBRegressor` (optional) |  |
| Tree | `boosting` | `catboost_regressor` | `CatBoostRegressor` (optional) |  |

## ▍ Optional estimators

| Extra | Install | Default model configured? |
|---|---|:---:|
| XGBoost | `pip install 'featranker[xgboost]'` | ✔ |
| CatBoost | `pip install 'featranker[catboost]'` | ✔ |

Missing or broken optional libraries skip only the affected models. Each skipped model produces a warning and a structured initialization failure.

> [!WARNING]
> **Known issue (CatBoost):** CatBoost 1.2.8 estimators do not define `__sklearn_tags__`. On scikit-learn 1.7, every scorer call on a CatBoost model emits a `DeprecationWarning` saying this becomes an error in scikit-learn 1.8. If that happens, CatBoost models appear in `failures` and the other models still rank. Install `featranker[catboost]` with scikit-learn < 1.8 if you need the CatBoost models.
