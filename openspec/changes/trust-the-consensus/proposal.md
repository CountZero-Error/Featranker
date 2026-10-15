## Why

The consensus counts every built-in model as an independent vote, but the registry holds close variants: four logistic regressions, a cluster of SGD-style margin classifiers, and several boosted tree ensembles. With `group="all"`, a family with many members outvotes the others, so "median of 20 models" reads as stronger evidence than it is. Users also have no programmatic way to apply the five-step reading guide in `docs/report.md`, and the always-on progress bars clutter notebooks, logs, and tests. These are the cheapest fixes that make the existing output more honest before 0.6 adds resampling.

## What Changes

- **Family-aware consensus.** Each built-in model gets a family tag in `importance_config.yaml`, such as `logistic`, `linear_margin`, `bagging`, or `boosting`. The consensus first takes the median rank within each family, then combines the family medians. A family with many members gets one vote, the same as a family with one member.
- **Custom models.** A model passed through `models={...}` forms its own one-model family. For custom-only rankers the consensus is therefore identical to 0.4.
- **Report schema, additive only.**
  - Each model report gains `family`.
  - Each consensus row gains `n_families` and `family_ranks`, a map from family to that family's median rank.
  - Existing keys keep their names and types.
- **Changed meaning of the consensus numbers.** For rankers using built-in models, `median_rank`, `mean_rank`, and `rank_std` are now computed over family medians, not over individual models. Their values and the consensus order can change compared with 0.4. `n_models` keeps its meaning: the number of voting models.
- **`verbose` flag.** `FeatureRanker(..., verbose=True)` controls the progress bars. The CLI gains `--quiet`. The default stays `True`, so current output is unchanged.
- **`summarize(report)`.** A new public function, exported from `featranker`, returns a list of structured findings that turns the reading guide into code:
  - whether the dummy gate was active;
  - whether the consensus is empty;
  - failures, grouped by stage;
  - a consensus that rests on few families;
  - consensus groups with tied median ranks;
  - groups whose importance stays within shuffle noise in most voting models.

  It is pure and deterministic, needs no I/O, and does not change the report.

## Non-goals

- Resampling or cross-split stability (0.6).
- Significance testing or shadow features (0.7).
- A feature-selection transformer (0.8).
- LLM interpretation (0.9).
- A report object or plots (1.0); the report stays a plain dict.
- User-assigned families for custom models. Each custom model is its own family until a real need appears.
- Weighting votes by model score, or adding a margin to the dummy gate.
- Changing the registry's model list, including adding LightGBM models.

## Capabilities

### New Capabilities

- `family-consensus`: how voting models are grouped into families and how the consensus combines per-family ranks, including the new report fields.
- `progress-output`: when progress bars are shown, controlled by `verbose` and `--quiet`.
- `report-summary`: the deterministic `summarize(report)` findings, their codes, severities, and trigger rules.

### Modified Capabilities

(none; no specs exist yet)

## Impact

- **Code:**
  - `featranker/importance.py`: `_build_consensus`, `_rank_model`, `init_models`, `_run_models`, and the `FeatureRanker` constructor.
  - `featranker/importance_config.yaml`: a `family` key on each entry.
  - `featranker/cli.py`: `--quiet`.
  - A new `featranker/summary.py`.
  - `featranker/__init__.py`: export `summarize`.
- **Report schema:** additive fields only (`family`, `n_families`, `family_ranks`). The consensus values change meaning for built-in models, as described above. Minor version bump to 0.5.0.
- **Docs:**
  - `docs/report.md`: consensus fields, the reading guide, and pointers to `summarize`.
  - `docs/usage.md`: the `verbose` parameter, `--quiet`, and `summarize`.
  - `docs/models.md`: a family column.
  - `README.md`: the consensus card wording.
- **Dependencies:** none added.
- **Tests:**
  - new tests for family consensus, verbose output, and summary findings;
  - existing consensus tests updated where built-in model voting changes values.
