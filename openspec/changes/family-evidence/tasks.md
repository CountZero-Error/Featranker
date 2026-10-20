## 1. Per-family evidence

- [x] 1.1 Test: `family_above_shadow` on consensus rows (family fractions scenario; mean equals `above_shadow_share`; sorted keys); stability `family_above_shadow_share` divided by n_splits (non-voting, failed and empty splits count 0; changed after review) and `shadow_families` strict > 0.5 (stability scenario); no keys without shadows; strict JSON; reports otherwise identical to HEAD
- [x] 1.2 Implement in `_build_consensus` and `_build_stability` (D1, D2)

## 2. Shadow mode

- [x] 2.1 Test: `shadow_mode` validation; default joint reproduces HEAD exactly; independent mode shadows are per-column permutations of their own column and break an exact-copy pair; seeded reproducibility; report records `shadow_mode` only with shadows; 1.5 pickles default to joint; `rank_resampled` passes it to splits
- [x] 2.2 Implement (D4)

## 3. Selector

- [x] 3.1 Test: `min_shadow_families` and `shadow_mode` round-trip clone/get_params; validation (positive int; bad mode); no-rule error names all four rules; family filter scenario; family filter + other filters + k; family rule alone turns shadows on; old pickles default both
- [x] 3.2 Implement (D3); version 1.6.0; typing consumer

## 4. Docs and checks

- [x] 4.1 Docs: docs/usage.md (reading family evidence; `min_shadow_families` with N=2 advice and the overfitting-family risk, pointing to the calibration study; `shadow_mode`), docs/report.md (new keys), README; CHANGELOG 1.6.0 (local); one runnable example on planted interaction data showing `shadow_families` and the family rule keeping the interaction pair
- [x] 4.2 Full suite passes; mypy clean; `openspec validate family-evidence --strict` passes
