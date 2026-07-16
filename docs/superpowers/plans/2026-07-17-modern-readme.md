# Modern README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite FeatRanker's README as a polished developer-tool landing page without changing approved 0.2.0 behavior or safety guidance.

**Architecture:** Modify only `README.md`. Reorder existing information so value, installation, and safe quick-start usage appear first; keep advanced schema, failures, and legacy details in compact tables and collapsible sections.

**Tech Stack:** GitHub-flavored Markdown, small GitHub-supported HTML blocks, Shields.io badges.

## Global Constraints

- Preserve `FeatureRanker.fit()` and `rank_features()` usage exactly.
- Preserve leakage, non-causal interpretation, failure, finite-score, and legacy warnings.
- Do not modify package code, tests, dependencies, estimator configuration, version, or public behavior.
- Do not add logos, screenshots, generated assets, decorative emoji, or undocumented claims.

---

### Task 1: Rewrite and Validate README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: public behavior documented by `featranker/importance.py` and package metadata in `pyproject.toml`
- Produces: GitHub-rendered project landing page with accurate install, usage, safety, report, migration, and development guidance

- [ ] **Step 1: Replace README structure**

Use this exact top-level order:

```markdown
<div align="center">
  <h1>FeatRanker</h1>
  <p><strong>Leakage-safe, model-agnostic feature ranking for trustworthy ML workflows.</strong></p>
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
  <a href="#result-schema">Result schema</a>
</p>

## Why FeatRanker?
## Installation
## Quick start
## Leakage-safe validation
## How it works
## Grouped permutation
## Data and scoring contract
## Result schema
## Optional estimators
## Legacy migration
## Development
## Scope and interpretation
## License
```

The hero links badges to PyPI, the GitHub Actions workflow, Python classifiers on PyPI, and `LICENSE`. “Why FeatRanker?” uses a three-column Markdown table for held-out evaluation, semantic feature groups, and rank consensus.

- [ ] **Step 2: Preserve safe API examples**

Keep the held-out workflow:

```python
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
```

State that ranking occurs inside training validation, final/outer test data cannot guide feature selection, and callers own splitting and preprocessing.

- [ ] **Step 3: Compact advanced documentation**

Use a table for input/scoring rules. Put the full report object and failure semantics inside `<details>` blocks. Preserve `evaluation_mode`, per-model score/importances/ranks, consensus fields, structured failures, finite-value exclusion, deterministic minimum-rank ties, and reproducibility controls.

- [ ] **Step 4: Validate Markdown and factual consistency**

Run:

```bash
rtk rg -n '^#{1,3} |FeatureRanker|rank_features|evaluation_mode|rankFeatures|outer|causal' README.md
rtk git diff --check
rtk git status --short --branch
```

Expected: all required concepts present, no whitespace errors, and only `README.md` plus this committed plan/design history changed.

- [ ] **Step 5: Commit**

```bash
rtk git add README.md
rtk git commit -m "docs: modernize README"
```
