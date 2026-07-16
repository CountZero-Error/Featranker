# Modern README Design

## Goal

Turn `README.md` into a polished developer-tool landing page while preserving FeatRanker 0.2.0 API accuracy, leakage guidance, failure semantics, and biomedical interpretation limits.

## Visual Direction

Use a centered HTML hero with project name, concise tagline, and badges for PyPI, supported Python, tests, and MIT license. Keep the page text-native: no new logo, screenshots, generated assets, decorative emoji, or external site dependency beyond standard badge images.

## Information Order

1. Hero, badges, and anchor navigation.
2. Three-column value summary: held-out safety, grouped permutation, rank consensus.
3. Core and optional installation commands.
4. Held-out regression quick start.
5. Leakage warning and validation placement.
6. Compact workflow explanation.
7. Grouped genotype example.
8. Input contract and scoring semantics.
9. Report schema and failure behavior in collapsible detail sections.
10. Reproducibility, optional estimators, legacy migration, scope, development, and license.

This order introduces value and safe usage before implementation detail. Advanced material stays available without dominating the first screen.

## Content Rules

- Preserve the approved `FeatureRanker.fit()` and `rank_features()` interface exactly.
- Keep `evaluation_mode`, finite-score/failure behavior, deterministic tie handling, and rank-consensus documentation.
- State that callers own splitting, preprocessing, and cross-validation.
- Warn against using outer-fold or final-test rankings for feature selection.
- Describe rankings as selection/ablation evidence, not causal, biological, or clinical importance.
- Keep legacy `rankFeatures()` and CLI marked as deprecated in-sample workflows.
- Keep paragraphs short and examples directly runnable once caller data exists.
- Do not change package code, dependencies, version, estimator configuration, or public behavior.

## Validation

Check heading hierarchy, anchor targets, badge URLs, code fences, install extras, API names, report fields, and line-level Markdown whitespace. Run `git diff --check`; package tests are unnecessary because only documentation changes.
