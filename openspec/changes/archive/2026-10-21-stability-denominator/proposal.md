## Why

The resampled stability `above_shadow_share` averages a group's per-split share over the splits that produced a consensus only. When most splits have no consensus (for example on weak or pure-noise data, where most models lose to the dummy baseline), one lucky split gives a noise group a share of 1.0. The calibration study's review measured the effect: `min_shadow_share=0.5` selected 11 % of pure-noise groups in regression with `group="fast"`, mostly from runs where only 1–2 of 10 splits voted. 1.6 fixed the same selection bias for per-family shares by dividing by `n_splits`; `shadow_hits` already counts missing splits as non-hits. The stability share should follow the same rule. Shadows have never been released (last release 0.4.1), so the meaning can be corrected before 1.0 ships.

## What Changes

- Stability `above_shadow_share` becomes the sum of the per-split consensus shares divided by `n_splits`: failed splits, empty-consensus splits, and splits where the group has no consensus row count as 0.
- `ConsensusSelector(min_shadow_share=...)` therefore becomes stricter on data where many splits lack a consensus; on data where every split votes, values are unchanged.
- Docs: report and usage text; the benchmark and calibration write-ups note that their numbers were computed with the earlier definition (their result files are not re-run).
- Version 1.8.0 (the planned categorical change moves to 1.9.0).

Report schema: no new keys; the meaning of stability `above_shadow_share` changes (unreleased).

## Non-goals

- Changing single-split consensus `above_shadow_share`, `min_shadow_share` defaults, or any threshold.
- Re-running the benchmark or calibration study.

## Capabilities

### New Capabilities

### Modified Capabilities
- `shadow-significance`: the "Resampled shadows" requirement's stability share definition.

## Impact

- Code: `featranker/importance.py` (`_build_stability`).
- Docs: docs/report.md, docs/usage.md, docs/benchmark.md and docs/calibration.md (a note each), README if it states the definition; CHANGELOG (local).
