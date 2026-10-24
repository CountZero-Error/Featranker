## 1. Copies

- [x] 1.1 Test: shadows equal old construction (joint, independent, within_groups, circular, object dtype); Spearman equals old implementation bit for bit (with/without NaN, constant column, object-free numeric)
- [x] 1.2 Implement D2, D3

## 2. Options

- [x] 2.1 Test: max_categories validation, encoder width, unseen → infrequent, default byte-identical, pickles, selector/CLI pass-through, run key
- [x] 2.2 Implement D1
- [x] 2.3 Test: fit max_samples (row count seen by a recording estimator, n_train_rows, reproducible, within_groups whole groups, circular order, zero-weight error, evidence train copy is the subsample, fingerprint of full data, ablate still accepts the report), rank_resampled/selector/CLI pass-through
- [x] 2.4 Implement D4

## 3. Measure and document

- [x] 3.1 D5 measurements appended to spikes/memory/FINDINGS.md
- [x] 3.2 Docs: docs/usage.md large-data section; report.md; README; CHANGELOG 1.21.0 (local); roadmap 1.21 and limitations L11/L12 in openspec/config.yaml; version 1.21.0
- [x] 2.5 Fixes after review: Spearman blocks by cells with an F-ordered output (peak never above 1.20, bit-identical at large n); fit(max_samples) raises when a class is missing; run records train_max_samples, train_random_state, train_rows_sha256 and ablate() checks the hash; docs for unseen levels and subsample-counted levels/missing
- [x] 3.3 Full suite passes; mypy clean; `openspec validate memory-limits --strict` passes
