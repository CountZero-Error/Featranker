## ADDED Requirements

### Requirement: Shadow hits test
With shadows on, `rank_resampled()` SHALL add to every stability row:
- `shadow_hits`: the number of splits whose consensus row for the group has `above_shadow_share > 0.5`; splits where the group has no consensus row count as non-hits;
- `shadow_p_value`: the one-sided binomial probability of at least `shadow_hits` successes in `n_splits` trials with success probability 0.5;
- `shadow_q_value`: the Benjamini–Hochberg adjusted p-value across all stability rows, capped at 1 and monotone in `shadow_p_value`.

The resampled report SHALL include `shadow_test` with `method` `"binomial"`, `hit` `"above_shadow_share > 0.5"`, `null_probability` `0.5`, `correction` `"benjamini-hochberg"`, `n_trials` (equal to `n_splits`), and `min_p_value` (`0.5 ** n_splits`). Without shadows, none of these keys SHALL appear. All values SHALL be JSON-safe.

#### Scenario: Hits and p-value
- **WHEN** a group's per-split shares over 10 splits are 1.0 nine times and 0.0 once
- **THEN** `shadow_hits == 9` and `shadow_p_value` equals the binomial tail `P(X >= 9 | n=10, p=0.5)` (about 0.0107)

#### Scenario: Missing split is a non-hit
- **WHEN** a group has no consensus row in one of 5 splits and a share of 1.0 in the other 4
- **THEN** `shadow_hits == 4` and the test uses 5 trials

#### Scenario: Strict majority
- **WHEN** a group's share is exactly 0.5 in a split
- **THEN** that split is not a hit

#### Scenario: BH adjustment
- **WHEN** three groups have p-values 0.01, 0.04, and 0.03
- **THEN** their q-values are 0.03, 0.04, and 0.04

#### Scenario: Few splits
- **WHEN** `cv=5` with shadows
- **THEN** `shadow_test.min_p_value == 0.03125`

#### Scenario: No shadows
- **WHEN** `rank_resampled()` runs without shadows
- **THEN** stability rows have no `shadow_hits` and the report has no `shadow_test`
