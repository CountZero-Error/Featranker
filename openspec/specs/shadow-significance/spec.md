# shadow-significance Specification

## Purpose
Show whether each feature group carries more signal than noise. Each model is given a shuffled "shadow" copy of every feature, and each real group is compared with the strongest shadow, model by model and family by family.
## Requirements
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

### Requirement: Shadow importance and comparison
For each successfully ranked model:
- the report SHALL include `shadow_importance`, giving each feature group's shadow (the same columns in the shadow block) with `values`, `mean`, and `std`, measured with the same scorer, repeats, and permutation procedure as the real groups;
- the report SHALL include `shadow_max`, the largest shadow mean;
- each real group's importance entry SHALL gain `above_shadow`, which is `mean > shadow_max`.

Real-group `rank` values SHALL be computed among real groups only.

#### Scenario: Strong signal beats every shadow
- **WHEN** `y` is a near-linear function of feature `signal`, and `shadows=True` with a linear model
- **THEN** `signal` has `above_shadow == True` in that model

#### Scenario: Ranks ignore shadows
- **WHEN** the same data is ranked with `shadows=True` and with `shadows=False` using a model unaffected by extra noise columns
- **THEN** the real groups have the same rank order in both reports

### Requirement: Family-weighted share
With shadows on, each consensus row SHALL include `above_shadow_share`:
- for each voting family, take the fraction of its voting members with `above_shadow == True`;
- `above_shadow_share` is the mean of those fractions across voting families.

#### Scenario: Family weighting
- **WHEN** family `F1` has members beating the shadow in 2 of 2 models, and family `F2` in 0 of 1
- **THEN** `above_shadow_share == 0.5`

### Requirement: Resampled shadows
`rank_resampled()` SHALL pass `shadows` and `shadow_mode` to every split ranker. Split `i` SHALL use `shadow_seed + i` when `shadow_seed` is an int, and `None` otherwise. With shadows on, each stability row SHALL include `above_shadow_share`, the mean of that group's per-split consensus `above_shadow_share` over ranked splits, and the resampled report SHALL record `shadow_mode`.

#### Scenario: Stability share
- **WHEN** a group's split consensus shares are 1.0, 0.5, and `null` (an empty consensus)
- **THEN** its stability `above_shadow_share == 0.75`

#### Scenario: Mode reaches every split
- **WHEN** `rank_resampled()` runs on a ranker with `shadows=True, shadow_mode="independent"`
- **THEN** every split report and the resampled report record `shadow_mode == "independent"`

### Requirement: Pickled 0.6 rankers
A ranker pickled without the shadow attributes SHALL load with `shadows=False` and `shadow_seed=None`.

#### Scenario: Load old pickle
- **WHEN** a fitted ranker without `shadows` and `shadow_seed` in its state is unpickled and ranked
- **THEN** ranking succeeds and the report has no shadow keys

### Requirement: Shadow hits test
With shadows on, `rank_resampled()` SHALL add to every stability row:
- `shadow_hits`: the number of splits whose consensus row for the group has `above_shadow_share > 0.5`; splits where the group has no consensus row count as non-hits;
- `shadow_p_value`: the one-sided binomial probability of at least `shadow_hits` successes in `n_splits` trials with success probability 0.5;
- `shadow_q_value`: the Benjamini–Hochberg adjusted p-value across all stability rows, capped at 1 and monotone in `shadow_p_value`.

The resampled report SHALL include `shadow_test` with `method` `"binomial"`, `hit` `"above_shadow_share > 0.5"`, `null_probability` `0.5`, `correction` `"benjamini-hochberg"`, `n_trials` (equal to `n_splits`), `min_p_value` (`0.5 ** n_splits`), `calibrated` (`false`), and `note`, a string stating that the values are binomial reference values whose null and false-discovery control are not established for arbitrary estimators or overlapping splits. Without shadows, none of these keys SHALL appear. All values SHALL be JSON-safe.

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

#### Scenario: Uncalibrated label
- **WHEN** `rank_resampled()` runs with shadows
- **THEN** `shadow_test.calibrated is False` and `shadow_test.note` mentions that the values are not calibrated

#### Scenario: No shadows
- **WHEN** `rank_resampled()` runs without shadows
- **THEN** stability rows have no `shadow_hits` and the report has no `shadow_test`

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

