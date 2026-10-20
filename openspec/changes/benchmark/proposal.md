## Why

FeatRanker claims that a multi-family, held-out, shadow-checked ranking is more trustworthy than single-model selectors, but it has no public evidence. A reproducible benchmark on data with known ground truth, against the tools users would otherwise pick, either backs the claim or shows where it fails. Both outcomes get published.

## What Changes

- New `benchmarks/` directory (not part of the package, not run by CI), like `spikes/`: one script that generates planted-signal data, runs each selector, scores it against ground truth, and writes a JSON result file and a Markdown table.
- Data: synthetic regression and classification tasks with known roles: linear, threshold, an interaction pair, a correlated copy of a signal feature, and pure-noise features; 5 seeds; a larger-noise variant.
- Selectors compared:
  - FeatRanker `ConsensusSelector` with `min_shadow_share=0.5` and with `max_shadow_q=0.05` (1.4), each with `group="fast"` and `group="all"`;
  - BorutaPy (random forest);
  - scikit-learn `permutation_importance` on a random forest with a "mean − 2·std > 0" rule;
  - feature-engine `SelectByShuffling` (random forest, default threshold).
- Metrics per seed: recall of signal features, number of noise features selected, runtime. The correlated copy is neutral (neither credited nor penalised).
- `docs/benchmark.md` with the method, the table, the environment (package versions), and an honest reading, including where FeatRanker loses (expected: runtime). README links it.

## Non-goals

- Real-world datasets without ground truth.
- Tuning competitors beyond their documented defaults, or FeatRanker beyond its documented rules.
- CI integration; adding competitor packages to any extra.

## Capabilities

### New Capabilities

### Modified Capabilities

None: no package behaviour changes (`skip_specs: true`).

## Impact

- New: `benchmarks/run.py`, `benchmarks/results/`, `docs/benchmark.md`; README link.
- Benchmark-only dependencies (`boruta`, `feature-engine`) installed in a throwaway virtual environment, never in `pyproject.toml`. The sdist/wheel must not include `benchmarks/`.
