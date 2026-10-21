## 1. Study code

- [x] 1.1 Test (`benchmarks/test_calibration.py`, fake rankers): null data has no dependence on the target (shuffling y changes nothing structural); rules are applied to one report per configuration; decision functions Q, F, S return the right outcomes on hand-made summaries, including borderline cases
- [x] 1.2 Implement `benchmarks/calibration.py` (D1–D3), reusing `benchmarks/run.py` for the signal generator; result JSON with environment and the script's SHA-256
- [x] 1.3 Freeze: record the SHA-256 in `docs/calibration.md` before the full run; smoke run on one seed

## 2. Run and write-up

- [x] 2.1 Full run in the scratch virtual environment; save `benchmarks/results/<date>-calibration.json`
- [ ] 2.2 `docs/calibration.md`: method, frozen hash, tables, decisions Q/F/S as computed by the script, deviations; link from docs/usage.md (shadow hits test and `min_shadow_families` sections) and docs/benchmark.md; update docs wording and, if decision Q allows, the `shadow_test.note` text (still stating `calibrated: false` for arbitrary estimators)

## 3. Checks

- [ ] 3.1 Package suite passes; `pytest benchmarks` passes; `openspec validate calibration-study --strict` passes
