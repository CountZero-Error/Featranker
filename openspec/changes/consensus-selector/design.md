## Context

- `FeatureRanker.rank_resampled` (0.6) already gives everything selection needs: a leakage-safe ranking on the data passed, a stability order, and, from 0.7, `above_shadow_share`.
- scikit-learn selectors subclass `SelectorMixin` and implement `_get_support_mask()`. `transform`, `get_support`, `inverse_transform`, and `get_feature_names_out` then come for free.
- Requirements are in specs/. Motivation is in proposal.md.

## Goals / Non-Goals

**Goals:**
- No change to `FeatureRanker`. The selector is a thin adapter in its own module.
- Nested-CV safety comes from structure: the selector can only see what `fit()` receives.

**Non-Goals:**
- Full `check_estimator` compliance (see the proposal).

## Decisions

**1. Rank with `rank_resampled`, not a single split.**

`fit()` must produce a training/evaluation split from the data it is given. A single internal holdout would select on one noisy split, while resampling gives the stability order and the shadow shares.

Alternative considered: an internal train/test split in `fit()`. Rejected: noisier, and it duplicates logic that already exists.

**2. Two rules, `k` and `min_shadow_share`; shadows follow from the rule.**

These map to the two questions FeatRanker answers: "the top k" and "beats noise".
- Shadows turn on exactly when `min_shadow_share` is set, so users can't ask for a shadow filter without shadows, and don't pay double cost for `k` alone.
- `shadow_seed` is passed through.

Alternatives considered:
- a `rule="..."` string;
- separate `shadows=` and threshold parameters.

Rejected because each adds a parameter that can contradict the others.

**3. Stability order is the only ordering.**

`k` takes the first `k` rows of `report_["stability"]`, which is already sorted by median place, then `top_k_share`, then name. That is deterministic, with no extra tie rule. `rank_resampled` is called with `top_k=k` when `k` is set, else its default, so `top_k_share` in `report_` reflects the user's `k`.

**4. Validation in `fit()`, parameters stored raw.**

This is scikit-learn convention, needed for `clone` and `set_params` and for `GridSearchCV` to set `k` after construction. A private `_validate_rule()` runs first, so invalid rules fail before any ranking.

**5. Names.**

- DataFrame columns become feature names and set `feature_names_in_`, using scikit-learn's `_check_feature_names` semantics via `validate_data` when available, otherwise set manually.
- Arrays get `x0..x{n-1}`, which matches what `get_feature_names_out()` produces for arrays.
- `feature_groups` refer to these names.
- The selector converts the input to a float array for `transform`, through scikit-learn's own `_transform` path.

**6. Default `verbose=False`.**

Pipelines fit the selector many times, and progress bars per fold are noise. `FeatureRanker` keeps its own default of `True`.

**7. Empty selection.**

Warn and keep zero columns, as `SelectFromModel` does. Raising would abort `cross_val_score`; silently keeping everything would hide the result.

## Risks / Trade-offs

- [Cost: `outer_folds * cv` rankings, doubled with shadows; `GridSearchCV` multiplies that again] → Docs give the formula and suggest `group="tree"`, `cv=3`, and fewer `n_repeats` for search. Caching is a non-goal.
- [Small outer folds can make inner `cv` fail; for example, stratification with rare classes] → Failed splits are recorded in `report_`, and if every split fails, `fit()` raises with the split errors.
- [Users read `report_` from one outer fold as "the" ranking] → Docs say each fitted selector reflects its own training data; for the overall ranking, use `rank_resampled` directly.

## Migration Plan

- Additive. Release as 0.8.0.
