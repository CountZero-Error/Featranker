## Decisions

**D1. Unknown mapping is sticky.** In `_kept_inputs`, after the first Pipeline step whose mapping is not tracked (anything but the column-preserving allowlist, a mapped CT, or the first step), every later step is scanned recursively (`get_params(deep=True)` values plus the step itself) for `SelectorMixin`, `FeatureUnion`, or `ColumnTransformer`; any hit → UNVERIFIED. The same scan applies to any estimator that is not itself a Pipeline, selector, `FeatureUnion` or `ColumnTransformer` (a meta-estimator such as `TransformedTargetRegressor` or `BaggingRegressor`), wherever it sits (whole model, after a scaler, in a union part or a column-transformer branch). Signal's reproduction is a test. Built-in pipelines must stay verified (existing test).

**D2. Native scorers only.** Sharing eligibility: `type(scorer)` is one of `sklearn.metrics._scorer`'s own scorer classes (`_Scorer`, and `_PredictScorer`/`_ProbaScorer`/`_ThresholdScorer` where they exist), not `isinstance`. Test: a deterministic subclass overriding `__call__` (clipping) — full cached vs uncached reports equal.

**D3. Symmetric row check.** In `ablation._prepare`, always compare `report.run.get("train_rows_sha256")` with the ranker's hash, both `None` meaning full data; mismatch → `ValueError`. No version before 1.21 could subsample training rows, so a report without the hash was ranked on every row. Tests both directions, including a pre-1.21 report with a full ranker (accepted) and with a subsampled ranker (refused).

**D4. Label scheme.** `_labels_fingerprint(labels, scheme=FINGERPRINT_SCHEME)`; `rank_features` passes `self.fingerprint_scheme_`. Old-pickle test with `shuffle="within_groups"` and integer labels: all run hashes scheme 1.

**D5. Spearman dtype.** Rank buffer and norm use the dtype `rankdata` returns (probe once on the first block, or rank the first block before allocating). Equality test on float32 input passes on both old and new SciPy behaviour.

**D6. Resampled summary.** Resampled reports with `constant_within_groups` get a finding whose evidence lists `{group: [split indices]}` and message "constant within shuffle groups in at least one split; their importance was zero by construction in those splits". Held-out message unchanged.

**D7. Boundary.** Docs, docstrings, spec: "at least max_categories levels". Test at equality.

**D8. Smoke.** Smoke ablation asserts each row's `n_models_scored > 0` and finite non-null `paired_median_score`, printing the ablation failures on error.
