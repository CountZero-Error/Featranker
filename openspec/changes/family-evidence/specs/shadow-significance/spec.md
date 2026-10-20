## MODIFIED Requirements

### Requirement: Shadow options
`FeatureRanker` SHALL accept `shadows` (bool, default `False`), `shadow_seed` (non-negative int or `None`, default `None`), and `shadow_mode` (`"joint"` or `"independent"`, default `"joint"`). A non-bool `shadows` SHALL raise `TypeError`. A `shadow_seed` that is neither an int nor `None` (bool included) SHALL raise `TypeError`, and a negative `shadow_seed` SHALL raise `ValueError` at construction. Any other `shadow_mode` SHALL raise `ValueError` at construction. With `shadows=False`, behavior and reports SHALL be identical to 0.6, whatever `shadow_mode` is. With shadows on, every report SHALL record `shadow_mode`. A ranker unpickled without `shadow_mode` SHALL behave as `"joint"`.

#### Scenario: Default off is unchanged
- **WHEN** a ranker is built without `shadows`
- **THEN** its reports contain no `shadows`, `shadow_seed`, `shadow_mode`, `shadow_importance`, `shadow_max`, `above_shadow`, or `above_shadow_share` keys

#### Scenario: Invalid options
- **WHEN** `FeatureRanker(task="reg", shadows="yes")` or `FeatureRanker(task="reg", shadows=True, shadow_seed=1.5)` is called
- **THEN** `TypeError` is raised

#### Scenario: Negative seed
- **WHEN** `FeatureRanker(task="reg", shadows=True, shadow_seed=-1)` is called
- **THEN** `ValueError` is raised before any fit

#### Scenario: Invalid mode
- **WHEN** `FeatureRanker(task="reg", shadow_mode="per-column")` is called
- **THEN** `ValueError` is raised

### Requirement: Shadow construction
With `shadows=True`:
- `fit()` SHALL train every model, but not the dummy baseline, on the training features followed by a shadow block, giving twice the columns. With `shadow_mode="joint"`, the shadow block is the training rows in one random order, applied to all columns. With `shadow_mode="independent"`, each shadow column is that column's training values in its own random order.
- `rank_features()` SHALL build the evaluation shadow block the same way from the evaluation rows, using independent random orders.
- Shadow columns SHALL never use rows from outside the data passed to that call.
- Feature-name validation SHALL use the real feature names only.

#### Scenario: Shadows come from the same call's rows
- **WHEN** a recording estimator is fit with `shadows=True` on training rows `R`
- **THEN** it receives `2 * n_features` columns, and every shadow row equals some row of `R` in joint mode

#### Scenario: Independent mode breaks collinearity
- **WHEN** column `f` equals column `e` and `shadow_mode="independent"`
- **THEN** each shadow column is a permutation of its real column, and the shadows of `e` and `f` are not identical

#### Scenario: Seeded shadows are reproducible
- **WHEN** two rankers with `shadows=True, shadow_seed=3`, the same `shadow_mode`, deterministic estimators and scorer, and the same integer permutation `random_state` are fit and ranked identically
- **THEN** their reports are equal

### Requirement: Resampled shadows
`rank_resampled()` SHALL pass `shadows` and `shadow_mode` to every split ranker. Split `i` SHALL use `shadow_seed + i` when `shadow_seed` is an int, and `None` otherwise. With shadows on, each stability row SHALL include `above_shadow_share`, the mean of that group's per-split consensus `above_shadow_share` over ranked splits, and the resampled report SHALL record `shadow_mode`.

#### Scenario: Stability share
- **WHEN** a group's split consensus shares are 1.0, 0.5, and `null` (an empty consensus)
- **THEN** its stability `above_shadow_share == 0.75`

#### Scenario: Mode reaches every split
- **WHEN** `rank_resampled()` runs on a ranker with `shadows=True, shadow_mode="independent"`
- **THEN** every split report and the resampled report record `shadow_mode == "independent"`

## ADDED Requirements

### Requirement: Per-family shadow evidence
With shadows on, every consensus row SHALL include `family_above_shadow`: for each voting family, the fraction of its voting members with `above_shadow == True`. `above_shadow_share` SHALL equal the mean of these values. In resampled reports, every stability row SHALL include `family_above_shadow_share` (for each family that voted in at least one split, the sum of its per-split fractions divided by `n_splits`, so splits where it did not vote, failed splits, and empty splits count as 0, as for `shadow_hits`) and `shadow_families` (the number of families whose `family_above_shadow_share` is greater than 0.5). Without shadows, none of these keys SHALL appear.

#### Scenario: Family fractions
- **WHEN** family `F1` has 2 of 2 voting members above the shadow and family `F2` has 0 of 1
- **THEN** `family_above_shadow == {"F1": 1.0, "F2": 0.0}` and `above_shadow_share == 0.5`

#### Scenario: Stability family shares
- **WHEN** over three splits family `F1` has fractions 1.0, 1.0, 0.0 and family `F2` voted only in the first split with 1.0
- **THEN** `family_above_shadow_share == {"F1": 0.6667, "F2": 0.3333}` (rounded) and `shadow_families == 1`

#### Scenario: A family that rarely votes cannot pass alone
- **WHEN** over 10 splits a family passes the dummy gate in only 2 splits and is above the shadows in both
- **THEN** its `family_above_shadow_share` is 0.2 and it does not count toward `shadow_families`
