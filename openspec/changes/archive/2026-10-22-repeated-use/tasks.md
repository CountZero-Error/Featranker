## 1. Run record

- [x] 1.1 Test: `run` block in rank_features and rank_resampled reports (task, group, data_sha256, train_data_sha256, cv, splits_sha256, per-split run); same-inputs, one-value-changed, different-splits, row-order, name-change scenarios; categorical and NaN data hash; old pickles load (`train_data_sha256` null); strict JSON
- [x] 1.2 Implement (D1, D2)

## 2. Grouping threshold and selector warning

- [x] 2.1 Test: `auto_groups_threshold` (looser threshold scenario, invalid value raises, clone/get_params); `run["auto_groups"]` present only for automatic grouping (selector and CLI); CLI `--auto-groups-threshold`; split warning (one failed split, empty-consensus split, all vote → no warning)
- [x] 2.2 Implement (D3, D5)

## 3. Resampled summary

- [x] 3.1 Test: each resampled rule and its order; clean resampled report has no warnings; report not modified; `Report.summary()` on resampled reports; single-split findings unchanged
- [x] 3.2 Implement (D4); version 1.12.0
- [x] 3.3 Docs: docs/report.md (`run`), docs/usage.md (comparing runs; summarising resampled reports; threshold option), README; CHANGELOG 1.12.0 (local); typing consumer
- [x] 3.4 Full suite passes; mypy clean; `openspec validate repeated-use --strict` passes
