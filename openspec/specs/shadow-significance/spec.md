# shadow-significance Specification

## Purpose
Show whether each feature group carries more signal than noise. Each model is given a shuffled "shadow" copy of every feature, and each real group is compared with the strongest shadow, model by model and family by family.
## Requirements
### Requirement: Shadow options
`FeatureRanker` SHALL accept `shadows` (bool, default `False`) and `shadow_seed` (non-negative int or `None`, default `None`). A non-bool `shadows` SHALL raise `TypeError`. A `shadow_seed` that is neither an int nor `None` (bool included) SHALL raise `TypeError`, and a negative `shadow_seed` SHALL raise `ValueError` at construction. With `shadows=False`, behavior and reports SHALL be identical to 0.6.

#### Scenario: Default off is unchanged
- **WHEN** a ranker is built without `shadows`
- **THEN** its reports contain no `shadows`, `shadow_seed`, `shadow_importance`, `shadow_max`, `above_shadow`, or `above_shadow_share` keys

#### Scenario: Invalid options
- **WHEN** `FeatureRanker(task="reg", shadows="yes")` or `FeatureRanker(task="reg", shadows=True, shadow_seed=1.5)` is called
- **THEN** `TypeError` is raised

#### Scenario: Negative seed
- **WHEN** `FeatureRanker(task="reg", shadows=True, shadow_seed=-1)` is called
- **THEN** `ValueError` is raised before any fit

### Requirement: Shadow construction
With `shadows=True`:
- `fit()` SHALL train every model, but not the dummy baseline, on the training features followed by a shadow block, giving twice the columns. The shadow block is the training rows in one random order, applied to all columns.
- `rank_features()` SHALL build the evaluation shadow block the same way from the evaluation rows, using an independent random order.
- Shadow columns SHALL never use rows from outside the data passed to that call.
- Feature-name validation SHALL use the real feature names only.

#### Scenario: Shadows come from the same call's rows
- **WHEN** a recording estimator is fit with `shadows=True` on training rows `R`
- **THEN** it receives `2 * n_features` columns, and every shadow row equals some row of `R`

#### Scenario: Seeded shadows are reproducible
- **WHEN** two rankers with `shadows=True, shadow_seed=3`, deterministic estimators and scorer, and the same integer permutation `random_state` are fit and ranked identically
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
`rank_resampled()` SHALL pass `shadows` to every split ranker. Split `i` SHALL use `shadow_seed + i` when `shadow_seed` is an int, and `None` otherwise. With shadows on, each stability row SHALL include `above_shadow_share`, the mean of that group's per-split consensus `above_shadow_share` over ranked splits.

#### Scenario: Stability share
- **WHEN** a group's split consensus shares are 1.0, 0.5, and `null` (an empty consensus)
- **THEN** its stability `above_shadow_share == 0.75`

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

