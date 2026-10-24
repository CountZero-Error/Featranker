## 1. Detection and fingerprints

- [x] 1.1 Test: nested-transformer reproduction (outer passes all, inner keeps real) gives a `shadows` failure; nested passthrough none; Pipeline inside a branch; unverifiable → failure message; existing named-input tests still pass
- [x] 1.2 Implement D1
- [x] 1.3 Test: 2**53 label swap differs; float/NaN unchanged behaviour within float columns; `fingerprint_scheme` in held-out, resampled, and selector-produced runs; old-scheme report accepted by `ablate()`
- [x] 1.4 Implement D2; `scikit-learn>=1.5` (D6)

## 2. Ablation, subsample, docs

- [x] 2.1 Test: provenance keys and `same_holdout` (same holdout true, third set false, other weights false); paired curve and cohort warning reproduction; summary uses paired median
- [x] 2.2 Implement D3, D4
- [x] 2.3 Test + implement D5 (held-out and resampled)
- [x] 2.4 Docs: usage.md (shadow check wording, ablation provenance/paired curve, comparing runs scheme note), report.md, CHANGELOG 1.17.0 (local, note fingerprint change), roadmap 1.17 in openspec/config.yaml; version 1.17.0
- [x] 2.5 Full suite passes; mypy clean; `openspec validate trust-fixes --strict` passes
