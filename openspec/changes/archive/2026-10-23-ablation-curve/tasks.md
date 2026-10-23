## 1. Ablation

- [x] 1.1 Test: signal scenario; report unchanged; default ks; ks validation; resampled/empty consensus/unfitted raise; fingerprint mismatch raises; only voting models refit; categorical, missing="impute", model_input="dataframe", sample weights pass through; failure scenario; strict JSON; Report type
- [x] 1.2 Implement in featranker/ablation.py with `FeatureRanker.ablate` delegating (D1–D7)

## 2. Summary, CLI, docs

- [x] 2.1 Test: `ablation_curve` finding; CLI `--drop` (normal, unknown, target, weight), `--ablate`, `--ablate-ks` (parse, needs `--ablate`)
- [x] 2.2 Implement (D8, D9); version 1.16.0
- [x] 2.3 Docs: docs/usage.md (ablation section with the evaluation-reuse caveat; CLI `--drop`, `--ablate`), docs/report.md (`ablation` block), docs/api.md, README; CHANGELOG 1.16.0 (local); roadmap 1.16 in openspec/config.yaml
- [x] 2.4 Full suite passes; mypy clean; `openspec validate ablation-curve --strict` passes
