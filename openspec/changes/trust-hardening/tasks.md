## 1. Bugs

- [x] 1.1 Test: mutating NumPy X/y and a float DataFrame after `fit(evidence=True)` leaves the report unchanged; a custom KNN that stores its training data is unaffected; validated inputs share no memory with the caller
- [x] 1.2 Implement copies at validation for every call (D2; replaces the evidence-only copy after review)
- [x] 1.3 Test: a selector pickled without `max_shadow_q` loads, transforms identically, has `get_params()["max_shadow_q"] is None`, clones and refits
- [x] 1.4 Implement `ConsensusSelector.__setstate__` (D3)

## 2. Uncalibrated label

- [x] 2.1 Test: `shadow_test.calibrated is False` and `note` present with shadows; absent without
- [x] 2.2 Implement the keys (D1); update docs (usage, report) with the replacement sentence and `max_shadow_q` as a heuristic

## 3. Small items

- [x] 3.1 Tests: `fast: "false"` string in a registry copy is not treated as fast; every fast linear representative (clf and reg) is a StandardScaler pipeline; CLI `--group bogus` errors; a `"passthrough"` pipeline step is treated as identity in the SHAP unwrap
- [x] 3.2 Implement D4; version 1.5.0

## 4. Docs and roadmap

- [x] 4.1 Docs corrections (D5): SHAP output scale and direction wording; overfit gap wording; fast-preset voting wording (docs/usage.md and docs/models.md); benchmark reproduction with `.[all]` and the recorded versions, and the SelectByShuffling runtime sentence; time-series and grouped-CV recipe; CHANGELOG 1.5.0 (local)
- [x] 4.2 Roadmap (openspec/config.yaml): mark the fixed items; reword L2 ("the shadow block preserves collinearity; large shadow importances were observed for one correlated pair in one seed; whether shared shuffling caused them remains untested"), L3 (null and FDR control not established for arbitrary estimators; label added in 1.5), L5 (output scale may be a link scale; direction is the coefficient sign when defined, else null); add raw/transformed feature mapping, selection with too few ranked splits, and run identity beyond L13
- [x] 4.3 Full suite passes; mypy clean; `openspec validate trust-hardening --strict` passes
