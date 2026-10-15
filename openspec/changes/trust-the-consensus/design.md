## Context

- `_build_consensus` in `featranker/importance.py` pools the ranks of all voting models per feature group and takes `np.median`, `np.mean`, and `np.std`.
- The registry (`importance_config.yaml`) has two groups, `linear` and `tree`, which are too coarse to serve as families.
- Model reports are built in `_rank_model`.
- Progress bars come from `tqdm`:
  - in `init_models`, once per registry group;
  - in `_run_models`, on the sequential path only (the joblib path prints nothing).
- Requirements are in `specs/`. Motivation is in `proposal.md`.

## Goals / Non-Goals

**Goals:**
- Families are declared once, in the registry, and are visible in every report.
- Existing tests with one model per family keep passing unchanged.
- `summarize` works on a report loaded from CLI JSON, not only on an in-memory result.

**Non-Goals:**
- Making family thresholds or `summarize` rules configurable.
- Back-filling families into reports produced by 0.4.

## Decisions

**1. Family taxonomy, declared as a `family:` key per registry entry.**

| Task | Family | Members |
|---|---|---|
| clf | `logistic` | logistic_regression, _l1, _l2, _elasticnet |
| clf | `linear_margin` | linear_svm, sgd_classifier, ridge_classifier, perceptron, passive_aggressive |
| clf | `discriminant` | lda, qda |
| clf | `naive_bayes` | naive_bayes_gaussian, naive_bayes_bernoulli |
| clf | `single_tree` | decision_tree |
| clf | `bagging` | random_forest, extra_trees, bagging_tree |
| clf | `boosting` | adaboost, gradient_boosting, hist_gradient_boosting, xgboost, catboost |
| reg | `linear` | linear_regression, ridge, lasso, elasticnet, elasticnet_cv, pls, huber, ransac, kernel_ridge (default `kernel="linear"`) |
| reg | `kernel` | svr_regression (RBF) |
| reg | `single_tree` | decision_tree_regressor |
| reg | `bagging` | random_forest_regressor, extra_trees_regressor |
| reg | `boosting` | adaboost, gradient_boosting, hist_gradient_boosting, xgboost, catboost regressors |

The grouping follows how the estimators learn, not their library or class. For example, `passive_aggressive` is an `SGDClassifier` with a hinge loss, so it sits in `linear_margin`.

Alternative considered: one family per registry group (`linear`/`tree`). Rejected because two votes total would make the consensus fragile.

**2. Combine with a median of family medians.**

The family rank is the median of the member ranks. The consensus then takes the median, mean, and std of the family ranks.

Alternative considered: a weighted per-model median with weight 1/family size. Rejected because it is harder to explain and to audit. The two-level median also yields `family_ranks`, which users can inspect directly.

**3. Families travel with the ranker.**

- `init_models` records `name -> family` in a `model_families_` dict.
- For custom models, the family is the model's own name.
- An entry missing `family` also falls back to its own name, but a registry test forbids that for shipped entries.
- `_rank_model` copies the family into the model report.
- `_build_consensus` reads `family` from each model report, so it needs no extra argument.

Alternative considered: wrapping estimators in a holder object. Rejected because it changes what `trained_models` holds, which is public-ish.

**4. `verbose` becomes `tqdm(disable=not verbose)`.**

- `verbose` is passed to `_run_models` and used in `init_models`.
- It is validated with `isinstance(verbose, bool)`.
- No logging framework is added.

Alternative considered: honoring `TQDM_DISABLE` only. Rejected because it is not discoverable.

**5. `summarize` lives in its own module, `featranker/summary.py`, as a pure function over the dict.**

- It is not a method, so it works on reports loaded from JSON.
- Thresholds are named module constants: `MIN_FAMILIES = 3`, and "within noise" means `mean <= std` in more than half of the voting models.
- For reports older than 0.5, which have no `n_families`, the `few_families` rule is skipped silently.
- The CLI does not gain a summary flag in this change.

Alternative considered: adding the findings to the report itself. Rejected because it would grow the schema, and findings are derived data.

## Risks / Trade-offs

- [The consensus order changes for built-in rankers compared with 0.4] → Call it out in the changelog and release notes. Update `docs/report.md` with the new meaning. Add a test that pins the diabetes quick-start output, and update the README example if its output changes.
- [The taxonomy is a judgment call] → Show it in a family column in `docs/models.md`. It is editable in the YAML, and the registry test guards completeness.
- [With few families, the median is coarse; for example, `group="tree"` on regression gives 3 families] → `few_families` flags this directly. Docs recommend `group="all"`.
- [`tied_groups` is noisy when many groups tie] → Emit one finding per tie set, not per pair.
- [`within_noise` with `n_repeats=1` has std 0, so it fires only when mean <= 0] → Acceptable. Document that more repeats make the check meaningful.

## Migration Plan

- Release as 0.5.0. The report changes are additive.
- To keep 0.4 behavior, users can pin `featranker<0.5`. With custom `models`, they get per-model voting unchanged.
- Rollback means reverting the release. Reports carry no persisted state.
