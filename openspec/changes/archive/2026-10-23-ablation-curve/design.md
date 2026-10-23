## Context

`FeatureRanker.fit()` already handles imputation, categorical coding, named model input and sample weights; `rank_features()` scores every fitted model on evaluation data (`models[name]["evaluation_score"]`). importance.py is 2060 lines, so new logic goes in its own module.

## Decisions

**D1. Refit through a sub-ranker.** For each `k`, build a `FeatureRanker` with the same `task`, `group`, `models`, `n_jobs`, `missing`, `model_input`, the original `categorical` setting restricted to the kept columns, `shadows=False`, `evidence=False`, `verbose=False`, `shuffle="rows"`, limited to the report's voting models. Fit it on the kept columns of the training data and call `rank_features` with one group holding every kept column and `n_repeats=1`; read each model's `evaluation_score`. Every preprocessing path stays identical to the ranking run with no duplicated code. The one permutation per model is the price; it is small next to a refit.
Alternative: clone fitted pipelines and slice columns. Rejected: one-hot steps hold column indices.

**D2. Which models.** The models with `in_consensus: true` in `report["models"]`. Empty consensus → `ValueError` (nothing to ablate).

**D3. Which groups.** Top-k rows of `report["consensus"]` in list order; columns are the union of their `feature_groups` entries, in training column order.

**D4. Default ks.** `{1, 2, 3, 5, 10, 20, 50, 100, ...}` up to `n` (number of consensus groups), plus `n`. Given `ks`: positive ints ≤ `n`, deduplicated and sorted; anything else `ValueError`.

**D5. Same training data.** `ablate()` requires a fitted ranker, a held-out report (resampled → `ValueError`), and training data whose fingerprint equals `report["run"]["train_data_sha256"]` when that is not null; mismatch → `ValueError`. The training and evaluation weights are checked the same way against `train_weights_sha256` and (when `X_eval`/`y_eval` are the report's) `weights_sha256`. This stops ablating a report with other data or weights.

**D6. Scoring.** `scoring=None` uses `report["scoring"]` (the primary scorer name); a report whose scorer was a callable needs `scoring=` (the name is not resolvable → `ValueError` from the existing resolver).

**D7. Block shape.**
```
"ablation": {
  "scoring": str, "models": [voting names], "baseline_score": float|null (the sub-ranker's own dummy score),
  "n_groups": n,
  "curve": [{"k", "feature_groups": [...], "n_features", "scores": {model: float|null}, "median_score": float|null}],
  "failures": [{"model", "stage", "message", "k"}]
}
```
A model that fails at some `k` gets `null` there and a failure entry. `median_score` is over non-null scores.

**D8. Summary.** One `info` finding `ablation_curve`: the `k` with the highest `median_score` (smallest k on ties) and the `median_score` at the largest scored `k`. Higher is better (sklearn scorers).

**D9. CLI.** `--drop a,b` drops columns before categorical inference and feature loading; unknown names, the target, or the weight column → error. `--ablate` calls `ablate()` with the training and evaluation CSVs; `--ablate-ks` parses ints and needs `--ablate`.

**D10. Baseline.** `baseline_score` is the first non-null `baseline_score` the sub-rankers report: the dummy fit on the same training rows and weights and scored on all of `X_eval` with the eval weights and the scorer used. The report's own value is not comparable (`max_samples`, other weights, a third `X_eval`). It is therefore valid with a `scoring` override too. `ablate()` scores every `X_eval` row and ignores the report's `max_samples`.

## Risks

- Scoring the curve on the evaluation data that produced the ranking is optimistic. Documented; users can pass a third data set as `X_eval`.
- Cost: `len(ks)` × voting models refits.
