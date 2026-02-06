# FeatureRanker

FeatureRanker ranks feature importance across many ML models using permutation importance.

| Item | Value |
| --- | --- |
| Package name | `featureRanker` |
| Import module | `featureRanker` |
| CLI command | `featureRanker` |
| Model config | `featureRanker/importance_config.yaml` |
| Default prep module | `featureCalc.py` (project root) |

---

## Table Of Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Data Prep Classes](#data-prep-classes-required)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Model Configuration](#model-configuration)
- [Output Format](#output-format)
- [Python API (Basic)](#python-api-basic)
- [Troubleshooting](#troubleshooting)

---

## What It Does

- Trains models defined in `featureRanker/importance_config.yaml`.
- Computes permutation feature importance per model.
- Returns per-model rankings plus an `average` ranking.
- Supports custom data pipelines through user-defined prep classes.

---

## Installation

```bash
pip install -r requirements.txt
```

For local package development:

```bash
pip install -e .
```

---

## Data Prep Classes (Required)

Before running CLI or API, define at least one prep class that returns your data in the expected shape.

Expected return shape:

```python
{
  "feature1": [value1, value2, ...],
  "feature2": [value1, value2, ...],
  "label": [label1, label2, ...]
}
```

Default workflow:
- Edit `featureCalc.py` in your project root.
- Define one or more prep classes there.
- Select class via `--prep-class`.

One-time-install workflow (no reinstall after data-prep edits):
- Keep your prep classes in any file, e.g. `my_feature_calc.py`.
- Load it with `--prep-file` + `--prep-class`.

---

## Quick Start

1. Define your prep class(es) in `featureCalc.py` or your own prep file.
2. Run FeatureRanker with the class name.

```bash
# Classification using class from project-root featureCalc.py
featureRanker --prep-class IrisPrepFeature --task clf --group all

# Regression using external prep file, with JSON output
featureRanker --prep-file ./my_feature_calc.py --prep-class DiabetesPrepFeature --task reg --group tree --output results
```

---

## CLI Reference

| Flag | Description | Default |
| --- | --- | --- |
| `--prep-class` | Class name to load (from `--prep-file` or `featureCalc.py`) | `prepFeature` |
| `--prep-file` | Optional user prep Python file path | Not set |
| `--task` | `clf` or `reg` | Required |
| `--group` | `linear`, `tree`, or `all` | `all` |
| `--output` | Output file path (`.json` auto-added if missing) | Print to stdout |

Output behavior:
- No `--output`: pretty JSON printed to stdout.
- With `--output`: JSON written to file.

---

## Model Configuration

`featureRanker/importance_config.yaml` groups models by:
- Task: `classification`, `regression`
- Group: `linear`, `tree`

Each model entry supports:
- `name`
- `import` (module path)
- `class` (class name)
- `params` (constructor kwargs)

---

## Output Format

Returned structure:
- Keys: model names plus `average`
- Values: sorted list of `{feature_name: score}` dictionaries
- Scores: rounded to 4 decimal places

Example:

```json
{
  "some_model": [
    {"feature_a": 0.1234},
    {"feature_b": 0.0456}
  ],
  "average": [
    {"feature_a": 0.1111},
    {"feature_b": 0.0555}
  ]
}
```

---

## Python API (Basic)

If your default class in project-root `featureCalc.py` is `prepFeature`:

```python
from featureRanker.importance import FeatureRanker

ranker = FeatureRanker(task="clf", group="all")
result = ranker.rankFeatures()
```

For external prep files, use:

```python
from featureRanker import build_ranker

RankerClass = build_ranker(
    prep_class="IrisPrepFeature",
    prep_file="./my_feature_calc.py",
)
ranker = RankerClass(task="clf", group="all")
result = ranker.rankFeatures()
```

---

## Troubleshooting

- Error: `prepFeature class 'X' not found`
  - If using external file: verify `--prep-file` path and class name
  - If not using external file: define class `X` in project-root `featureCalc.py`
- Error about missing `label`
  - Ensure `_calc_features()` returns a `label` key
- Feature length mismatch
  - Ensure every feature array has the same sample count as `label`
