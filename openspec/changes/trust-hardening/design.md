## Context

See proposal.md. The relevant code: `FeatureRanker.fit()` stores `self.train_X_ = X_models`, `self.train_y_ = y_values` (importance.py ~778), which alias the caller's float arrays when no copy happened during validation. `ConsensusSelector` has no `__setstate__`. `rank_resampled()` builds the `shadow_test` block.

## Decisions

### D1. Relabel, not remove
The p/q keys and `max_shadow_q` stay (1.1–1.4 are unreleased, so either choice is free); they are honest once labelled. `shadow_test` gains `calibrated: false` and a `note`. Docs replace the "conservative null" sentence with: "A 0.5 null probability and false-discovery-rate control are not established for arbitrary estimators or overlapping splits; these binomial reference values should not be interpreted as calibrated significance." `max_shadow_q` docs call it a heuristic filter. 1.6 studies calibration on fresh seeds.

### D2. Copies at validation
Root cause: `_validate_X` (`np.asarray(X, dtype=float)`) and the target helpers return the caller's own arrays when no conversion is needed, so models (and the kept evidence data) can alias them; custom estimators that store training data (KNN, a bare KernelRidge) are affected, not only evidence. Fix once where every path goes through: `_validate_X` uses `np.array(X, dtype=float)` (always a copy) and the target validation copies too. The `.copy()` calls added in `fit()` become redundant and are removed.

### D3. Selector pickles
`ConsensusSelector.__setstate__` sets `max_shadow_q` to `None` when absent, then delegates to `BaseEstimator.__setstate__`. The test builds an old-state fixture by deleting `max_shadow_q` from a fitted selector's `__dict__` before pickling (equivalent to a 1.3 pickle).

### D4. Small items
- `init_models()`: a model counts as fast only if `model.get("fast") is True`.
- `_unwrap` reason for `"passthrough"`/`None` steps: name the step value, e.g. `pipeline step 'passthrough' is not a per-column scaler`, or treat them as identity; choose identity (sklearn's documented meaning) and test it.
- CLI `--group` help: "'linear' (linear, discriminant, naive Bayes and kernel models)", matching docs/models.md.

### D5. Docs only
The SHAP scale, direction, overfit gap, fast-preset voting, benchmark reproduction (`pip install -e '.[all]'` plus the recorded versions), the shuffling runtime sentence, and a new "Time series and grouped data" recipe (`TimeSeriesSplit`/`GroupKFold` with `rank_resampled` and the selector; note that shadows and permutations still shuffle rows, see roadmap). Roadmap list in openspec/config.yaml updated as listed in tasks.

## Risks / Trade-offs

- [Users may still read q < 0.05 as significance] → the report itself now says `calibrated: false`.
- [Every `fit`, `rank_features`, and `rank_resampled` call holds one extra copy of X and y] → documented; the copy is the price of never aliasing caller data.
