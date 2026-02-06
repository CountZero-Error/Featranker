# FeatRanker

Rank feature importance across multiple ML models using permutation importance.

FeatRanker trains a configurable set of scikit-learn, XGBoost, and
CatBoost models on your data, computes permutation importance for every trained
model, and returns per-model rankings plus an aggregated average ranking.

| Item | Value |
| --- | --- |
| Package name | `featranker` |
| Import module | `featranker` |
| CLI command | `featranker` |
| Model config | `featranker/importance_config.yaml` |
| Default prep file | `featureCalc.py` (project root) |

---

## Table of Contents

- [Installation](#installation)
- [How It Works](#how-it-works)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Model Configuration](#model-configuration)
- [Available Models](#available-models)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package in editable (development) mode:

```bash
pip install -e .
```

Or install from PyPI:

```bash
pip install featranker
```

### Requirements

- Python ≥ 3.10
- numpy, scikit-learn, pyyaml, tqdm, xgboost, lightgbm, catboost

---

## How It Works

1. **Load data** — A user-defined prep class returns a feature dict with a
   `"label"` key.
2. **Initialize models** — Model definitions are read from
   `importance_config.yaml` and instantiated for the requested task and group.
3. **Train models** — Every initialized model is fitted on the feature matrix.
4. **Rank features** — Permutation importance is computed per model, and an
   overall average ranking is produced.

---

## Data Preparation

Before running FeatRanker you need a **prep class** — a Python class with a
`_calc_features()` method that returns your data as a dict.

### Expected return format

```python
{
    "feature_1": [v1, v2, v3, ...],
    "feature_2": [v1, v2, v3, ...],
    ...
    "label":     [y1, y2, y3, ...],
}
```

- Every feature key maps to a list of numeric values.
- All lists (including `"label"`) must have the same length.
- The `"label"` key is required.

### Where to put it

**Option A — Edit the default file (simplest)**

Define your class in `featureCalc.py` at the project root. The default class
name is `prepFeature`, but you can name it anything and select it with
`--prep-class`.

**Option B — Use a separate file (no reinstall needed)**

Keep your prep logic in any Python file and point to it at runtime:

```bash
featranker --prep-file ./my_features.py --prep-class MyPrepClass --task clf
```

### Example prep class

```python
from sklearn.datasets import load_iris

class IrisPrep:
    def _calc_features(self):
        data = load_iris()
        features = {
            data.feature_names[i]: data.data[:, i].tolist()
            for i in range(data.data.shape[1])
        }
        features["label"] = data.target.tolist()
        return features
```

---

## Quick Start

1. Implement `_calc_features()` in `featureCalc.py` (or your own file).
2. Run the CLI:

```bash
# Classification with all model families, using the default prepFeature class
featranker --task clf --group all

# Regression with tree models only, custom prep file and class
featranker --task reg --group tree \
    --prep-file ./my_features.py --prep-class DiabetesPrep

# Save results to a JSON file
featranker --task clf --group linear --output results
```

---

## CLI Reference

```
featranker --task {clf,reg} [--group {linear,tree,all}]
           [--prep-file PATH] [--prep-class NAME]
           [--output PATH]
```

| Flag | Description | Default |
| --- | --- | --- |
| `--task` | `clf` (classification) or `reg` (regression) | *required* |
| `--group` | `linear`, `tree`, or `all` (both) | `all` |
| `--prep-class` | Name of the prep class to instantiate | `prepFeature` |
| `--prep-file` | Path to the Python file containing the prep class | `featureCalc.py` in the current working directory |
| `--output` | File path for JSON output (`.json` appended if missing) | print to stdout |

---

## Python API

### Using `FeatureRanker` directly (default prep file)

When your default `prepFeature` class lives in `featureCalc.py` at the project
root:

```python
from featranker import FeatureRanker

ranker = FeatureRanker(task="clf", group="all")
results = ranker.rankFeatures()
```

### Using `build_ranker` with a custom prep file

`build_ranker` is a convenience factory that returns a fully initialized
`FeatureRanker` instance (features loaded, models trained, ready to rank):

```python
from featranker import build_ranker

ranker = build_ranker(
    task="reg",
    group="tree",
    prep_file="./my_features.py",
    prep_class="DiabetesPrep",
)
results = ranker.rankFeatures()
```

### Constructor parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `task` | `"clf"` \| `"reg"` | Classification or regression |
| `group` | `"linear"` \| `"tree"` \| `"all"` | Which model family to use |
| `prep_file` | `str` or `None` | Path to prep file (defaults to `featureCalc.py`) |
| `prep_class` | `str` | Name of the prep class (defaults to `"prepFeature"`) |

---

## Model Configuration

Models are defined in `featranker/importance_config.yaml`, organized by task
and group:

```yaml
classification:
  linear:
    - name: logistic_regression
      import: sklearn.linear_model
      class: LogisticRegression
      params:
        max_iter: 2000
  tree:
    - name: random_forest
      import: sklearn.ensemble
      class: RandomForestClassifier
      params:
        random_state: 42

regression:
  linear:
    - ...
  tree:
    - ...
```

Each entry has four fields:

| Field | Description |
| --- | --- |
| `name` | Display name used in output |
| `import` | Python module to import (e.g., `sklearn.ensemble`) |
| `class` | Class name to instantiate from that module |
| `params` | Dict of keyword arguments passed to the constructor (optional) |

Edit this file to add, remove, or tune models. Changes take effect on the next
run — no reinstall required.

---

## Available Models

### Classification — Linear

| Name | Class |
| --- | --- |
| `logistic_regression` | `LogisticRegression` |
| `logistic_regression_l1` | `LogisticRegression` (L1) |
| `logistic_regression_l2` | `LogisticRegression` (L2) |
| `logistic_regression_elasticnet` | `LogisticRegression` (ElasticNet) |
| `linear_svm` | `LinearSVC` |
| `sgd_classifier` | `SGDClassifier` |
| `ridge_classifier` | `RidgeClassifier` |
| `perceptron` | `Perceptron` |
| `passive_aggressive` | `PassiveAggressiveClassifier` |
| `lda` | `LinearDiscriminantAnalysis` |
| `qda` | `QuadraticDiscriminantAnalysis` |
| `naive_bayes_gaussian` | `GaussianNB` |
| `naive_bayes_bernoulli` | `BernoulliNB` |
| `naive_bayes_multinomial` | `MultinomialNB` |
| `pls_da` | `PLSRegression` |

### Classification — Tree

| Name | Class |
| --- | --- |
| `decision_tree` | `DecisionTreeClassifier` |
| `random_forest` | `RandomForestClassifier` |
| `extra_trees` | `ExtraTreesClassifier` |
| `bagging_tree` | `BaggingClassifier` |
| `adaboost` | `AdaBoostClassifier` |
| `gradient_boosting` | `GradientBoostingClassifier` |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` |
| `xgboost` | `XGBClassifier` |
| `catboost` | `CatBoostClassifier` |

### Regression — Linear

| Name | Class |
| --- | --- |
| `linear_regression` | `LinearRegression` |
| `ridge_regression` | `Ridge` |
| `lasso_regression` | `Lasso` |
| `elasticnet_regression` | `ElasticNet` |
| `elasticnet_cv_regression` | `ElasticNetCV` |
| `pls_regression` | `PLSRegression` |
| `huber_regression` | `HuberRegressor` |
| `ransac_regression` | `RANSACRegressor` |
| `kernel_ridge_regression` | `KernelRidge` |
| `svr_regression` | `SVR` |

### Regression — Tree

| Name | Class |
| --- | --- |
| `decision_tree_regressor` | `DecisionTreeRegressor` |
| `random_forest_regressor` | `RandomForestRegressor` |
| `extra_trees_regressor` | `ExtraTreesRegressor` |
| `adaboost_regressor` | `AdaBoostRegressor` |
| `gradient_boosting_regressor` | `GradientBoostingRegressor` |
| `hist_gradient_boosting_regressor` | `HistGradientBoostingRegressor` |
| `xgboost_regressor` | `XGBRegressor` |
| `catboost_regressor` | `CatBoostRegressor` |

---

## Output Format

The result is a dict (or JSON object) keyed by model name, with an additional
`"average"` entry that aggregates across all models. Each value is a list of
single-entry dicts sorted by score in descending order. Scores are rounded to
four decimal places.

```json
{
  "logistic_regression": [
    {"feature_a": 0.1234},
    {"feature_b": 0.0567},
    {"feature_c": 0.0012}
  ],
  "random_forest": [
    {"feature_b": 0.0890},
    {"feature_a": 0.0745},
    {"feature_c": 0.0023}
  ],
  "average": [
    {"feature_a": 0.0990},
    {"feature_b": 0.0729},
    {"feature_c": 0.0018}
  ]
}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Prep file not found` | FeatRanker can't locate `featureCalc.py` | Run the command from the directory that contains `featureCalc.py`, or pass an explicit path with `--prep-file` |
| `AttributeError: … has no attribute 'X'` | The prep class name doesn't match what's in the file | Check spelling of `--prep-class` against the class defined in your prep file |
| `'label' key missing` | `_calc_features()` didn't include a `"label"` entry | Add `features["label"] = ...` to your return dict |
| Feature length mismatch | Feature lists have different lengths | Ensure every feature list and `"label"` have the same number of elements |
| Model training errors (printed, not fatal) | A model failed to converge or doesn't support the data | Check the printed warning; consider removing or tuning that model in `importance_config.yaml` |
