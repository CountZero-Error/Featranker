## 1. Script

- [x] 1.1 Test (`benchmarks/test_run.py`, fake selectors): data generation is deterministic per seed and permutes roles; scoring counts recall, noise selected, and treats the copy as neutral; a selector whose import fails is recorded as missing and the run continues
- [x] 1.2 Implement `benchmarks/run.py` (D1–D3): generator, selectors, scoring, JSON result with environment versions, Markdown table output
- [x] 1.3 Confirm `python -m build` sdist and wheel contain no `benchmarks/` files (update MANIFEST.in or packaging config if needed)

## 2. Run and write-up

- [x] 2.1 Run the benchmark in a scratch virtual environment with `boruta` and `feature-engine` (D4); store `benchmarks/results/<date>.json`
- [x] 2.2 Write `docs/benchmark.md` (method, environment, table, honest reading including losses and limitations) and link it from README and docs/README.md

## 3. Post-hoc diagnostic (D6)

- [x] 3.1 Add `featranker_share_tree` (by name only) and per-role, per-family shadow-share instrumentation; warm-up before timing; store roles per run
- [x] 3.2 Run the diagnostic on all seeds, tasks and variants; save `benchmarks/results/<date>-posthoc.json`
- [x] 3.3 Correct the docs mechanism and add the dated "Added after seeing the results" section; fairness facts and timing wording
- [x] 3.4 Re-review fixes: three docs numbers, `_shadow_diagnostics` unit test, `PREREGISTERED` as programmatic default, docstring and table label

## 4. Checks

- [x] 4.1 Package suite still passes; `openspec validate benchmark --strict` passes
