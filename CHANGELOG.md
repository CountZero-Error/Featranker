# Changelog

## 0.2.0 - 2026-07-16

### Added

- Explicit `fit(X_train, y_train)` and held-out `rank_features(X_eval, y_eval)` workflow.
- Named/callable scoring, grouped permutation, per-model evaluation scores, uncertainty, and rank consensus.
- Structured model failures and machine-readable `evaluation_mode` provenance.
- NumPy and pandas input validation, deterministic permutation seeds, tests, and core CI.
- Optional dependency extras for XGBoost, LightGBM, and CatBoost.

### Changed

- Consensus aggregates model ranks instead of heterogeneous raw importance values.
- Core installation now requires only NumPy, scikit-learn, PyYAML, and tqdm.
- `FeatureRanker` starts unfitted unless an explicit legacy preparation file is supplied.

### Deprecated

- `rankFeatures()` performs marked in-sample ranking and warns; use held-out `rank_features()`.
- `run_ML()` warns; use `fit()`.
