## Context

`FeatureRanker.init_models()` reads `importance_config.yaml` (task → `linear`/`tree` → model list, each with a `family`) and builds `models[task][group]`. `fit()` collects candidates from the requested groups (`["linear", "tree"]` for `"all"`). Families come from the YAML `family` key and drive the one-vote-per-family consensus. `rank_resampled()` and `ConsensusSelector` construct rankers with the same `group`. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:** the same families vote as with `"all"`, at a fraction of the cost; one source of truth for which models are fast (the YAML).

**Non-Goals:** per-call speed knobs; changing any family.

## Decisions

### D1. Tag in the YAML, not a list in code
Each representative gets `fast: true` in `importance_config.yaml`. `init_models()` still builds the `linear` and `tree` groups; with `group="fast"` it initializes both groups but keeps only entries tagged `fast`. Alternative: a hard-coded name list in `importance.py`. Rejected: two places to edit when the registry changes. A test pins the exact sets from the spec and checks family coverage, so a YAML edit that breaks coverage fails CI.

### D2. Representatives
Chosen for reliability, speed, and no optional dependency:
| Family | Classification | Regression |
|---|---|---|
| logistic | logistic_regression | — |
| linear_margin | ridge_classifier (closed form, no convergence warnings) | — |
| discriminant | lda | — |
| naive_bayes | naive_bayes_gaussian | — |
| linear | — | ridge_regression |
| kernel | — | svr_regression (the family's only model) |
| single_tree | decision_tree | decision_tree_regressor |
| bagging | random_forest | random_forest_regressor |
| boosting | hist_gradient_boosting | hist_gradient_boosting_regressor |

`svr_regression` is quadratic in rows; it is kept so the regression family set matches `"all"`. Docs note it.

### D3. Validation
The allowed set becomes `{"linear", "tree", "all", "fast"}` in `FeatureRanker.__init__`, the `Literal` types in `FeatureRanker` and `ConsensusSelector`, and the CLI `choices`. The existing rule "group selects built-in models; omit it with models" covers `"fast"` unchanged.

### D4. Report
Unchanged. The report already lists the models that ran; no `group` key is added (YAGNI).

## Risks / Trade-offs

- [One model per family is a less robust family vote than a median over several] → documented; `"all"` stays the default.
- [svr_regression is slow on large data] → documented; users can pass `models=` for full control.
- [A future registry edit forgets a `fast` tag] → the family-coverage test fails.
