## MODIFIED Requirements

### Requirement: Resampled shadows
`rank_resampled()` SHALL pass `shadows` and `shadow_mode` to every split ranker. Split `i` SHALL use `shadow_seed + i` when `shadow_seed` is an int, and `None` otherwise. With shadows on, each stability row SHALL include `above_shadow_share`: the sum of that group's per-split consensus `above_shadow_share` over all splits divided by `n_splits`, where failed splits, splits with an empty consensus, and splits without a consensus row for the group contribute 0. The resampled report SHALL record `shadow_mode`.

#### Scenario: Stability share
- **WHEN** a group's split consensus shares are 1.0, 0.5, and `null` (an empty consensus)
- **THEN** its stability `above_shadow_share == 0.5`

#### Scenario: One lucky split
- **WHEN** over 10 splits only one split has a consensus and the group's share there is 1.0
- **THEN** its stability `above_shadow_share == 0.1`

#### Scenario: Mode reaches every split
- **WHEN** `rank_resampled()` runs on a ranker with `shadows=True, shadow_mode="independent"`
- **THEN** every split report and the resampled report record `shadow_mode == "independent"`
