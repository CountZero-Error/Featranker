## Why

Users who run FeatRanker repeatedly (tuning, monitoring, comparing reruns) cannot tell whether two reports came from the same data, splits, and preset (L13, L21); the `auto_groups` threshold is fixed at 0.7 in the selector and CLI (L4); `ConsensusSelector` silently selects when some splits failed or produced no consensus (L20); and `summarize()` refuses resampled reports, so the most robust mode has no reading guide (A9).

## What Changes

- Run record: every report from `rank_features()` and `rank_resampled()` gains `run`:
  - `task` and `group` (the preset);
  - `data_sha256`: a fingerprint of the evaluation data (single split) or of all data (resampled);
  - `train_data_sha256`: the training data's fingerprint, single split only;
  - resampled only: `cv`, the splitter's `repr`, and `splits_sha256`, a fingerprint of the split indices.
- `auto_groups` threshold:
  - `ConsensusSelector(auto_groups_threshold=0.7)` and CLI `--auto-groups-threshold` (default 0.7).
  - When grouping is automatic, the report records `run["auto_groups"] = {"threshold": t, "method": "spearman-complete-linkage"}`.
- Selector warning: when any split failed or produced an empty consensus, `ConsensusSelector.fit` warns how many of the splits did not vote; those splits already count as misses in the shares.
- `summarize()` accepts resampled reports and returns resampled findings:
  - `split_failures` (warning)
  - `failures` (aggregated over splits)
  - `empty_consensus` (warning, splits without a consensus)
  - `gate_off` (warning, splits without a dummy gate)
  - `below_shadow` (info, from the stability table)
  - `insufficient_splits` (info, when the shadow test cannot reach q ≤ 0.05 with this many splits)
- Version 1.12.0.

Report schema: additive (`run` block; resampled findings). The `ValueError` for resampled reports in `summarize()` is removed.

## Non-goals

- Fingerprints that survive library upgrades or row reordering (equal inputs under the same versions give equal fingerprints; nothing more).
- Evaluation subsampling or speed work (1.13).
- A minimum-ranked-splits parameter: the 1.8 denominators already penalise missing splits; a warning is enough.

## Capabilities

### New Capabilities
- `run-record`: the `run` block, fingerprints, and the `auto_groups` record and threshold option.

### Modified Capabilities
- `report-summary`: resampled reports are summarised instead of rejected.
- `consensus-selector`: warning when splits did not vote; `auto_groups_threshold`.

## Impact

- Code: `featranker/importance.py` (`run` block), `featranker/summary.py` (resampled rules), `featranker/selection.py`, `featranker/cli.py`, `featranker/report.py` (docstring).
- Docs: docs/report.md (`run`), docs/usage.md (comparing runs; summarising resampled reports), README; CHANGELOG (local).
