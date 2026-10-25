## Why

Signal's pre-push review (2026-10-25) of 1.17–1.21 found remaining correctness defects and two failures on newer dependencies (scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1), and gave NO-GO for pushing and tagging until they are fixed:

1. A selector (or union, or nested column-aware transformer) after a column step whose mapping is not tracked is not inspected, so a `SelectKBest` after a passthrough `ColumnTransformer` can drop every shadow and the model is still certified "kept".
2. Shared scoring sends any `_BaseScorer` subclass through scikit-learn's multi-metric path, which calls `_score` and bypasses an overridden `__call__`; the baseline self-check can agree while permuted scores differ (7.52 cached vs 0.01 uncached).
3. `ablate()` checks the training-row hash only when the report has one, so a full-data report ablated with a later subsampled refit of the same ranker passes and refits on the subsample.
4. Old (scheme 1) rankers fingerprint evaluation shuffle labels with scheme 2.
5. A test uses the scorer name `"max_error"`, removed in scikit-learn 1.9.
6. `_spearman_matrix` promotes float32 ranks (kept by SciPy 1.18) to float64, so it is not identical to 1.20 on newer SciPy.
7. The resampled `constant_within_groups` summary says "zero by construction" for a union over splits, though a group can be constant in only some splits.
8. The docs say unseen levels share the infrequent column only when the column has more than `max_categories` levels; scikit-learn creates the bucket at equality too.
9. The installed-wheel smoke test accepts an ablation curve in which every refit failed.

## What Changes

- Shadow check: once a step's column mapping is unknown, any later step that is or contains a selector, `FeatureUnion`, or `ColumnTransformer` makes the model unverified (`shadows` failure).
- Scorer sharing only for scorers whose class is one of scikit-learn's own scorer classes (exact type, not subclasses); everything else is called directly.
- `ablate()` compares the ranker's training-row hash with the report's, treating absent as "full data", in both directions.
- Shuffle-label fingerprints use the ranker's `fingerprint_scheme_`.
- Test uses `make_scorer(max_error, greater_is_better=False)`.
- Spearman buffers use the dtype of `rankdata()` output.
- Resampled `constant_within_groups` summary names the affected splits and says "in those splits".
- Docs/spec: "at least `max_categories` levels"; equality-boundary test.
- Smoke test requires scored models and finite paired scores.
- Version 1.21.1.

Report schema: unchanged except the resampled summary message and evidence.

## Non-goals

- Tracking column mappings through arbitrary transformers.

## Capabilities

### New Capabilities
- `prepush-fixes`: the corrections above.

### Modified Capabilities
- `memory-limits`: the category-cap boundary is "at least max_categories levels".

## Impact

- Code: `featranker/importance.py`, `featranker/ablation.py`, `featranker/evidence.py`, `featranker/summary.py`; tests; `tests/smoke_installed.py`; docs/usage.md; CHANGELOG (local).
