## 1. Validation and fitting

- [x] 1.1 Test: weight validation (length, negative, NaN, inf, zero sum) before any fit; weights are copied; unweighted reports byte-identical to HEAD; built-in pipelines receive weights at the final step (recording estimator inside a pipeline); unsupported models (`lda`, `qda`, `pls_regression`) become `SampleWeightNotSupported` fit failures; weighted dummy scenario
- [x] 1.2 Implement validation and weighted fitting (D1, D3)

## 2. Scoring and record

- [x] 2.1 Test: baseline, permuted, dummy and train scores weighted (compare with manual weighted scores); "weights change the ranking" scenario; `sample_weight` record; resampled split reports carry it; strict JSON; evidence unweighted
- [x] 2.2 Implement (D2)

## 3. Resampling, selector, CLI, docs

- [x] 3.1 Test: rank_resampled slices (recording estimator); selector fit with weights; routed pipeline scenario; selector clone/get_params unchanged; CLI `--sample-weight` (missing column errors, column excluded from features); typing consumer
- [x] 3.2 Implement (D4, D5); version 1.11.0
- [x] 3.3 Docs: docs/usage.md (weights section: what is weighted, unsupported models, unweighted evidence, routing in pipelines), docs/report.md (`sample_weight`), README; CHANGELOG 1.11.0 (local); runnable example with real output
- [x] 3.4 Full suite passes; mypy clean; `openspec validate sample-weights --strict` passes
