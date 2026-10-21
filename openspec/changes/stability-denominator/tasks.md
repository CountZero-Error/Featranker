## 1. Definition

- [x] 1.1 Test: the spec scenarios (1.0, 0.5, null → 0.5; one lucky split of 10 → 0.1); a failed split (absent from `splits`) counts 0; when every split votes, values equal HEAD's; reports without shadows unchanged; `ConsensusSelector(min_shadow_share=0.5)` no longer selects a group whose only consensus split gave 1.0 out of 10
- [x] 1.2 Implement in `_build_stability`; version 1.8.0

## 2. Docs and checks

- [x] 2.1 Docs: docs/report.md and docs/usage.md definition of stability `above_shadow_share`; a dated note in docs/benchmark.md and docs/calibration.md that their tables use the earlier definition (not re-run); CHANGELOG 1.8.0 (local)
- [x] 2.2 Full suite passes; mypy clean; `openspec validate stability-denominator --strict` passes
