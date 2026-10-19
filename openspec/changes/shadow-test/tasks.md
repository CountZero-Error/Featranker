## 1. Hits test

- [x] 1.1 Test: hits, p-value (9 of 10 → 0.0107), strict-majority rule, missing split counts as a non-hit with `n_splits` trials, BH example (0.01, 0.04, 0.03 → 0.03, 0.04, 0.04), `shadow_test` block and `min_p_value` for cv=5, no keys without shadows, strict JSON; a planted-signal integration run with `RepeatedKFold` (10+ splits) gives the signal group a small q-value and pure noise a large one
- [x] 1.2 Implement in `_build_stability` and the resampled report (D1–D4)

## 2. Selector

- [x] 2.1 Test: `max_shadow_q` round-trips through clone/get_params; validation of `(0, 1]` and the "no rule" error now naming all three; q filter scenario; q filter then k; `max_shadow_q` alone turns shadows on
- [x] 2.2 Implement `max_shadow_q` (D5); version 1.4.0

## 3. Docs and checks

- [x] 3.1 Docs: docs/usage.md (significance section: what a hit is, conservative null, dependent splits, low power and `min_p_value`, recommend 10+ splits, selector `max_shadow_q`), docs/report.md (new keys), README, CHANGELOG.md 1.4.0 entry (local); examples run and output pasted
- [x] 3.2 Full suite passes; mypy clean; `openspec validate shadow-test --strict` passes
