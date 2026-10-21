## 1. Subsampling

- [ ] 1.1 Test: validation scenarios; unchanged default; same rows for every score (recording scorer); reproducible; int above n uses all rows; float floor with minimum 1; `max_samples` and `n_eval_rows` in the report; strict JSON
- [ ] 1.2 Implement (D1, D2)

## 2. Resampling, selector, CLI, docs

- [ ] 2.1 Test: rank_resampled per split; selector scenario; selector clone/get_params; CLI `--max-samples` int and float; typing consumer
- [ ] 2.2 Implement (D3); version 1.13.0
- [ ] 2.3 Docs: docs/usage.md (speed: `max_samples`, `group="fast"`, `n_jobs`), docs/report.md, README; CHANGELOG 1.13.0 (local)
- [ ] 2.4 Full suite passes; mypy clean; `openspec validate eval-subsampling --strict` passes
