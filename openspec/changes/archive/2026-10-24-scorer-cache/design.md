## Context

scikit-learn ≥ 1.5 (our floor) provides `sklearn.metrics._scorer._MultimetricScorer(scorers=..., raise_exc=...)`, which caches the estimator's response per method across `_BaseScorer` instances (those from `get_scorer` and `make_scorer`). It is private; `permutation_importance` uses it.

## Decisions

**D1. One scoring helper.** A small helper scores a model on one matrix for the primary and all secondary scorers and returns `{name: float | Exception}`. When there is more than one scorer, it puts the cacheable ones (`_BaseScorer` instances) into one `_MultimetricScorer(raise_exc=False)` and calls the rest directly. Used for the baseline (once directly, once shared, for the self-check in D4) and for each permuted copy.

**D2. Failure semantics unchanged.** A string (traceback) result for a secondary scorer becomes that scorer's failure as today; the message must be today's message. Simplest way to guarantee it: on any non-numeric result, re-call that scorer directly and let its own exception propagate or be recorded. The primary scorer's exception therefore keeps type and message.

**D3. Fallback.** Import failure of `_MultimetricScorer`, or any unexpected exception from it, falls back to per-scorer calls for that call (no state). Covered by a test that monkeypatches the import away.

**D4. Values identical, under stated conditions.** scikit-learn caches a response per method name only, ignoring `pos_label`, and (1.7+) passes `sample_weight` only to score functions that name the parameter. So scorers share a prediction only if they agree on response method and resolved `pos_label` (one `_MultimetricScorer` per `pos_label` group of two or more; regressors and `predict` scorers ignore it; a scorer whose `pos_label` cannot be read is called directly). With weights, a scorer whose score function lacks an explicit `sample_weight` parameter (a `**kwargs`-only function counts as lacking) is called directly. As a last guard, `_rank_model` scores the baseline per scorer directly and once through the shared path; sharing stays on for that model only if every shared value equals the direct value exactly and the same scorers failed. Under these conditions cached and uncached scores are equal bit for bit; tests assert report equality with and without caching and count `predict` calls with a recording estimator. The check covers the baseline only, so it cannot catch a divergence that appears on permuted data alone.

**D5. Weights.** `sample_weight` is passed to the multi-metric scorer as today's scorers receive it, inside `_unrouted`.
