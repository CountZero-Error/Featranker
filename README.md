# FeatureRanker

FeatureRanker ranks feature importance across many ML models using permutation importance.

| Item | Value |
| --- | --- |
| Package name | `featureRanker` |
| Import module | `featureRanker` |
| CLI command | `featureRanker` |
| Model config | `featureRanker/importance_config.yaml` |

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

Before running CLI or API, define at least one prep class in `featureRanker/featureCalc.py`.

Expected return shape:

```python
{
  "feature1": [value1, value2, ...],
  "feature2": [value1, value2, ...],
  "label": [label1, label2, ...]
}
```

You can define multiple classes in the same file and choose one at runtime via `--prep-class`.

Reference example:
- `example_featureCalc.py`
- Includes `IrisPrepFeature` and `DiabetesPrepFeature`

---

## Quick Start

1. Add your prep class(es) to `featureRanker/featureCalc.py`.
2. Run FeatureRanker with the class name.

```bash
# Classification with your custom prep class
featureRanker --prep-class IrisPrepFeature --task clf --group all

# Regression with your custom prep class and JSON output
featureRanker --prep-class DiabetesPrepFeature --task reg --group tree --output results
```

Without package install:

```bash
python featureRanker/importance.py --prep-class IrisPrepFeature --task clf --group all
```

---

## CLI Reference

| Flag | Description | Default |
| --- | --- | --- |
| `--prep-class` | Class name in `featureRanker/featureCalc.py` | `prepFeature` |
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

If your default class in `featureRanker/featureCalc.py` is `prepFeature`:

```python
from featureRanker.importance import FeatureRanker

ranker = FeatureRanker(task="clf", group="all")
result = ranker.rankFeatures()
```

For non-default prep classes, prefer CLI with `--prep-class`.

---

## Troubleshooting

- Error: `prepFeature class 'X' not found`
  - Define class `X` in `featureRanker/featureCalc.py`
  - Re-run with `--prep-class X`
- Error about missing `label`
  - Ensure `_calc_features()` returns a `label` key
- Feature length mismatch
  - Ensure every feature array has the same sample count as `label`
