## Purpose

Measure how stable a feature ranking is across different training/evaluation splits of the caller's development data, so users can tell a robust ranking from an artifact of one split.

## ADDED Requirements

### Requirement: Resampled ranking method
`FeatureRanker.rank_resampled(X, y, *, scoring, cv=5, groups=None, feature_names=None, feature_groups=None, n_repeats=10, random_state=None, top_k=5)` SHALL rank feature groups on each split of `X`/`y` and return a resampled report.
- `X`, `y`, `feature_names`, `scoring`, `feature_groups`, and `n_repeats` SHALL follow the same rules as `fit()` and `rank_features()`.
- The method SHALL NOT change the ranker's own fitted state.
- It SHALL work whether or not the ranker has been fitted.

#### Scenario: Ranker state untouched
- **WHEN** a fitted ranker calls `rank_resampled(...)`
- **THEN** its `trained_models`, `feature_names_`, and `is_fitted_` are the same objects and values as before the call

#### Scenario: Unfitted ranker
- **WHEN** `FeatureRanker(task="reg").rank_resampled(X, y, scoring="r2", cv=3)` is called
- **THEN** a resampled report with three splits is returned and the ranker remains unfitted

### Requirement: Splitter handling
`cv` SHALL accept the following, and nothing else:
- an integer >= 2, meaning shuffled `StratifiedKFold` for `task="clf"` or shuffled `KFold` for `task="reg"`, seeded with `random_state`;
- an object with a scikit-learn-style `split(X, y, groups)` method;
- an iterable of `(train_indices, eval_indices)` pairs.

`groups` SHALL be passed to the splitter's `split` method. Any other `cv`, a split with an empty training or evaluation side, or a split whose training and evaluation indices overlap SHALL raise `ValueError` before any model is fit. Overlap between different splits' evaluation rows, as with `RepeatedKFold`, is allowed. For integer `cv` with `task="clf"`, folds SHALL be stratified on class codes, so any labels `fit()` accepts, including fractional numbers, can be stratified.

#### Scenario: Integer folds for classification are stratified
- **WHEN** `task="clf"` and `cv=3` on a target with classes in proportion 2:1
- **THEN** every split's evaluation rows contain both classes

#### Scenario: Group splitter
- **WHEN** `cv=GroupKFold(2)` and `groups` assigns rows to two groups
- **THEN** no group appears in both the training and evaluation rows of any split

#### Scenario: Explicit index pairs
- **WHEN** `cv=[(train_idx, eval_idx)]` is passed
- **THEN** the report has exactly one split, with `n_train == len(train_idx)` and `n_eval == len(eval_idx)`

#### Scenario: Overlapping rows within a split
- **WHEN** `cv=[([0, 1, 2, 3], [3, 4])]` is passed
- **THEN** `ValueError` is raised before any fit

#### Scenario: Fractional class labels
- **WHEN** `task="clf"`, `y` alternates 0.1 and 0.2, and `cv=2`
- **THEN** both labels appear in every split's evaluation rows

#### Scenario: Invalid cv
- **WHEN** `cv=1` or `cv="five"` is passed
- **THEN** `ValueError` is raised and no model is fit

### Requirement: Per-split isolation and seeding
- Each split SHALL use a new ranker with the same `task`, `group`, `models`, `n_jobs`, and `verbose`.
- Each split ranker SHALL be fit only on that split's training rows and ranked only on that split's evaluation rows.
- When `random_state` is an integer, split `i` (0-based) SHALL use permutation seed `random_state + i`. When `random_state` is `None`, every split SHALL use `None`.
- With deterministic estimators, an integer `random_state`, and a deterministic splitter, two calls SHALL return equal reports apart from `provenance`.

#### Scenario: Reproducible
- **WHEN** `rank_resampled` is called twice with `cv=3` and `random_state=0` on built-in seeded models
- **THEN** the two `stability` lists and the per-split reports are equal

### Requirement: Resampled report shape
The report SHALL be a strict-JSON dict with:
- `evaluation_mode == "resampled"`;
- `scoring`, `n_repeats`, `random_state`, `feature_groups`, and `provenance`, as in single-split reports;
- `top_k`;
- `n_splits`;
- `splits`: a list of `{"split": i, "n_train": int, "n_eval": int, "report": <single-split report>}` for the splits that ranked;
- `stability`;
- `failures`.

#### Scenario: Strict JSON
- **WHEN** a resampled report is serialized with `json.dumps(report, allow_nan=False)`
- **THEN** serialization succeeds

### Requirement: Stability aggregation
A group's **place** in a split SHALL be its 1-based position in that split's sorted consensus. Groups with equal `median_rank` and equal `mean_rank` SHALL share the better (lower) place, and the next distinct group SHALL take its row position, so places skip after a tie.

For each feature group, a stability row SHALL contain:
- `feature_group`;
- `split_ranks`: one integer place per entry in `splits`, or `null` when that split's consensus is empty;
- `n_splits_ranked`: the number of non-null values;
- `median_rank` (float), `min_rank`, and `max_rank` (integers) over the non-null places;
- `top_k_share`: the number of ranked splits in which the group's place is at most `top_k`, divided by `n_splits`.

Rows SHALL be sorted by `median_rank` ascending, then `top_k_share` descending, then group name. `top_k` SHALL be a positive integer, else `ValueError`.

#### Scenario: Aggregation arithmetic
- **WHEN** group `x` is in places 1, 2, and 4 in three splits, and `top_k=2`
- **THEN** the row for `x` has `split_ranks == [1, 2, 4]`, `median_rank == 2.0`, `min_rank == 1`, `max_rank == 4`, `n_splits_ranked == 3`, and `top_k_share == 2/3`

#### Scenario: Place ignores the consensus score scale
- **WHEN** group `x` is first in a split's consensus with `median_rank == 4.5`
- **THEN** its place in that split is 1

#### Scenario: Tied groups share the better place
- **WHEN** a split's consensus has `x` (1.0, 1.0), `y` (2.0, 2.0), `z` (2.0, 2.0), and `w` (2.0, 3.0) as (`median_rank`, `mean_rank`)
- **THEN** their places are 1, 2, 2, and 4, and with `top_k=2`, both `y` and `z` count as top

#### Scenario: Split with empty consensus
- **WHEN** one of three ranked splits has an empty consensus
- **THEN** that split's `split_ranks` entry is `null`, `n_splits_ranked == 2`, and that split counts as "not in top k" in `top_k_share`

#### Scenario: No split has a consensus
- **WHEN** every ranked split has an empty consensus
- **THEN** `stability == []` and a `UserWarning` is emitted

### Requirement: Split failures
If fitting or ranking a split raises, the method SHALL record a failure with `split` (index), `model` (`null`), `stage` (`"split"`), `error_type`, and `message`, and SHALL continue with the next split. Per-model failures inside a ranked split SHALL also appear in the top-level `failures`, each with its `split` index added. If no split ranks, the method SHALL raise `RuntimeError` listing the failures.

#### Scenario: One split fails
- **WHEN** one of three splits raises during fit
- **THEN** `splits` has two entries, `n_splits == 3`, and `failures` contains an entry with `stage == "split"` and that split's index

#### Scenario: All splits fail
- **WHEN** every split raises
- **THEN** `RuntimeError` is raised and its message includes each split's error
