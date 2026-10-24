## 1. Families and within-group record

- [x] 1.1 Test: families validation (type, empty value, unknown name in fit); custom models share a vote; built-in remap; report `family`; old pickle loads; selector passes through and clones
- [x] 1.2 Implement D1
- [x] 1.3 Test: patient-level predictor listed + UserWarning; varying predictor not listed; key absent in other modes; max_samples subsample used; resampled split reports carry it; summary finding; strict JSON
- [x] 1.4 Implement D2, D3

## 2. Docs

- [x] 2.1 docs/usage.md: grouped-data recipe estimand; sample-weights donor law (cost vs population weights); ablation validation wording and self-contained runnable example (run it; paste real output); custom-family risk near the consensus claim and in family evidence; docs/report.md (`constant_within_groups`, families), docs/api.md; CHANGELOG 1.18.0 (local); roadmap 1.18 in openspec/config.yaml; version 1.18.0
- [x] 2.2 Review fixes: families validated once before the split loop (rank_resampled, selector), any Mapping accepted, resampled `constant_within_groups` union and summary finding, ablation example prints validation and test curves for every k and is run by a test, donor law and nested-CV wording
- [x] 2.3 Full suite passes; mypy clean; `openspec validate statistical-contracts --strict` passes
