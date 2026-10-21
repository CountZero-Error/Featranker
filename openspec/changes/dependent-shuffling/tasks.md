## 1. Shuffle modes

- [ ] 1.1 Test: option validation; default bit-identical; within-group scenario for permutations and shadows (joint and independent); single-row groups keep values; all-singleton raises; missing/mismatched labels raise; labels with another mode raise; circular is a roll by 1..n−1 (permutations and shadows, joint shares offset); fewer than 2 rows raises; old pickles; `shuffle` in report; strict JSON
- [ ] 1.2 Implement (D1, D2)

## 2. Resampling, selector, docs

- [ ] 2.1 Test: rank_resampled slices groups; missing groups raises; selector scenario; selector clone/get_params; typing consumer
- [ ] 2.2 Implement; version 1.15.0
- [ ] 2.3 Docs: docs/usage.md (time-series recipe with `"circular"`, grouped-data recipe), docs/report.md, README; CHANGELOG 1.15.0 (local); roadmap L22 note
- [ ] 2.4 Full suite passes; mypy clean; `openspec validate dependent-shuffling --strict` passes
