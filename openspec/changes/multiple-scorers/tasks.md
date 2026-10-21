## 1. Scorers

- [ ] 1.1 Test: validation (empty, duplicate, bad entry, one-element list unchanged); primary-unchanged scenario (bit-identical consensus and importances); secondary values match manual scoring; same permutations (recording scorer sees identical permuted arrays); secondary failure; dummy secondary scores; strict JSON
- [ ] 1.2 Implement (D1–D3)

## 2. Resampling, selector, CLI, docs

- [ ] 2.1 Test: rank_resampled split reports carry secondary, stability unchanged; selector with a list (clone/get_params); CLI comma-separated; typing consumer
- [ ] 2.2 Implement; version 1.14.0
- [ ] 2.3 Docs: docs/usage.md (several metrics), docs/report.md, README; CHANGELOG 1.14.0 (local)
- [ ] 2.4 Full suite passes; mypy clean; `openspec validate multiple-scorers --strict` passes
