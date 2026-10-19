## Context

`FeatureRanker.fit()` validates `X`/`y`, encodes labels, optionally appends a shadow block (`_with_shadows`), and fits every model on that matrix. Built-in linear models are `make_pipeline(StandardScaler(), estimator)`. `rank_features()` scores each model on evaluation data, runs grouped permutation importance (`_rank_model`, possibly in parallel via `_run_models`), and assembles a plain dict wrapped in `Report`. `rank_resampled()` builds a fresh ranker per split from this ranker's settings. `summarize()` is a list of rule functions. The report shape is public API: additive changes only (docs/api.md stability policy). See proposal.md for motivation.

## Goals / Non-Goals

**Goals:**
- Evidence is opt-in, costs nothing when off, and leaves 1.0 reports byte-identical.
- Every evidence number is reproducible from the report's inputs and JSON-safe.
- SHAP stays an optional dependency.

**Non-Goals:**
- Cross-split aggregation of evidence; evidence inside `ConsensusSelector`.
- Explaining models SHAP cannot explain fast (no KernelExplainer).

## Decisions

### D1. Constructor flag, training data kept on the ranker
`evidence` is a constructor flag, like `shadows`, because the training data is only available in `fit()`. With `evidence=True`, `fit()` keeps the validated training matrix as the models saw it (with the shadow block, if any) and the encoded `y` (`self.train_X_`, `self.train_y_`). `rank_features()` scores each model on it with the ranking scorer. Alternative: `rank_features(..., X_train=, y_train=)`. Rejected: users would have to pass the same data twice and could pass different data.

`__setstate__` defaults `evidence` to `False` for 1.0 pickles.

### D2. Train score inside the per-model job
`train_score` is computed in `_rank_model` (same worker, same scorer) so it parallelises with ranking. A scoring error there becomes a failure with stage `"evidence"`; the model's ranking stays. `overfit_gap = train_score - evaluation_score`, both "higher is better" scorer values.

### D3. Correlation once per report, on training data
`evidence.correlation` is computed once in `rank_features()` from the real (non-shadow) training columns: `scipy.stats.spearmanr` on the column matrix, then for each pair of groups the member-column pair with the largest `|rho|`. Training data, not evaluation data, so the later `auto-groups` change can reuse the same rule without letting evaluation data shape the groups. `threshold = 0.5` is a module constant, written into the report. NaN correlations (constant columns) are skipped.

### D4. SHAP in a new module, two explainers only
New `featranker/evidence.py` holds the correlation and SHAP helpers; `importance.py` calls them. `shap` is imported lazily inside the helper.
- Unwrap once for both explainers: a `Pipeline` is reduced to its final step, and the rows are transformed through the earlier steps, but only if every earlier step is a per-column scaler (`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`); a scaler is per-column affine, so each scaled column's SHAP value is the original column's SHAP value. Any other earlier step (PCA, polynomial features, ...) puts the model in `unsupported` (`ValueError: pipeline step PCA is not a per-column scaler`). A one-step pipeline is the bare estimator.
- Linear: if the final step has `coef_` and `intercept_`, use `shap.LinearExplainer(final, shap.maskers.Independent(background, max_samples=len(background)))`, where `background` is evenly spaced rows of the **training** data (`train_X_`, shadow block included, passed through the same scaler steps). The reference is thus the training distribution, as for `TreeExplainer`'s `tree_path_dependent` mode.
- Otherwise try `shap.TreeExplainer(final)`. Any exception (unsupported model type, SHAP error) puts the model in `unsupported` with `reason = type(error).__name__ + message[:200]`.
- Background (linear, training rows) and explained rows (evaluation rows): each at most `SHAP_ROWS = 1000`, evenly spaced indices (`np.linspace(0, n - 1, k).astype(int)`), so results do not depend on `random_state` or SHAP's own sampling.
- Output normalisation: SHAP returns `(n, p)`, `(n, p, k)`, or a list of `k` arrays depending on model and version. Normalise to `(k, n, p)`. `k == 1` → regression or single-output; `k == 2` → use class 1; `k > 2` → average `mean_abs` over classes, `direction = null`.
- Groups: sum member columns per row; shadow columns (indices ≥ the real column count) are dropped.

SHAP runs in the main process after ranking, not in the parallel jobs: fitted SHAP explainers and numba do not pickle reliably across joblib workers.

### D5. Missing shap
Any exception on `import shap` (`ImportError`, or a broken shap/numba/numpy install raising something else) → one `UserWarning` ("install featranker[shap]"), `evidence.shap = {"available": false}`. Correlation and train scores are still computed.

### D6. Summary rules
`_overfit` uses a scale-free share: `(train - eval) / (train - baseline)`. A raw gap is meaningless across scorers (R² vs negative MAE). `baseline_score` is the dummy's evaluation score, the best available reference. Threshold `MIN_OVERFIT_SHARE = 0.5`. `_correlated_groups` restates `evidence.correlation.pairs`. `FAILURE_STAGES` gains `"evidence"` (info severity, as for initialization).

### D7. Deprecation
`warnings.warn(..., FutureWarning, stacklevel=2)` at the top of `to_prompt()` and `explain()`. `explain()` already calls the private `_prepare()` directly, not `to_prompt()`, so no double warning arises and no private `_prompt()` was needed. FutureWarning, not DeprecationWarning, because end users (not only developers) must see it by default.

### D8. Dependencies and CI
`shap = ["shap"]` extra; `all` unchanged (shap pulls numba and is heavy). The `test` extra gains `shap` so CI runs the SHAP tests; tests use `pytest.importorskip("shap", exc_type=ImportError)` (so the `test` extra pins `pytest>=8.2`) and a plain install still passes.

## Risks / Trade-offs

- [SHAP output shapes differ across versions and model types] → normalise in one function; tests cover regression, binary, and multiclass for one linear and one tree model.
- [Pickled rankers with `evidence=True` carry the training data] → documented; off by default.
- [Memory: the training matrix is kept] → off by default; same size as the data the user already holds.
- [SHAP on 1000 rows of a big forest can take seconds per model] → row cap; only when `evidence=True`.
- [TreeExplainer default output differs by model (probability vs log-odds)] → `mean_abs` is comparable within a model, not across models; docs say so.
- [Correlation on training data can differ from evaluation data] → stated in the report (`data: "train"`).

## Migration Plan

Additive. 1.0 code paths and reports are unchanged unless `evidence=True`. `explain()`/`to_prompt()` keep working through 1.x with a `FutureWarning`; removal in 2.0.
