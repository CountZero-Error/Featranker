## Why

FeatRanker is slower than single-model tools: a run fits and permutes about 20 built-in models per task, but the consensus gives each model family one vote (9 families across both tasks). Many models are close variants inside one family, so they cost compute without adding votes. A preset with one representative per family keeps every family's vote with about a third of the models (measured: 0.31 of the time for classification, 0.45 for regression, where SVR dominates).

## What Changes

- `group` accepts a fourth value, `"fast"`: one built-in representative per family, across the linear and tree groups. Classification: 7 models (one per family, 22 in `"all"`); regression: 5 models (18 in `"all"`). The preset uses only scikit-learn models, so it needs no optional dependency.
- The registry marks representatives with `fast: true`.
- `ConsensusSelector(group="fast")` and CLI `--group fast` pass the preset through.
- Version 1.2.0.

Report schema: unchanged. Reports list whichever models ran, as today.

## Non-goals

- Skipping permutation for models that lose to the dummy (listed in the roadmap). Dropped: it would leave `importance` empty for those models under the new option, adding schema edge cases to save a small share of the work.
- Adaptive repeats, row subsampling, or other speed knobs.
- Changing the `"all"` default or any family assignment.

## Capabilities

### New Capabilities
- `fast-preset`: the `"fast"` group value, its representatives, and its pass-through to `ConsensusSelector` and the CLI.

### Modified Capabilities

## Impact

- Code: `featranker/importance_config.yaml` (`fast: true` tags), `featranker/importance.py` (group validation and selection, `Literal` type), `featranker/selection.py` (type and pass-through), `featranker/cli.py` (choice).
- Docs: README, docs/models.md (Fast column), docs/usage.md (speed section), CHANGELOG (local).
- Tests: preset selection, family coverage, pass-through.
