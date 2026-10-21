## Purpose

Trade evaluation rows for speed while keeping the dummy gate and every score on the same rows.

## ADDED Requirements

### Requirement: Subsample argument
`rank_features()` and `rank_resampled()` SHALL accept `max_samples` (default `None`): a positive int or a float in (0, 1]; otherwise `ValueError` before any scoring. An int larger than the number of evaluation rows SHALL use all rows. A float SHALL use `max(1, floor(max_samples * n))` rows. Without `max_samples`, reports SHALL be unchanged.

#### Scenario: Invalid value
- **WHEN** `max_samples` is 0, -1, 1.5, `True`, or a string
- **THEN** `ValueError` is raised

#### Scenario: Unchanged default
- **WHEN** `max_samples` is not given
- **THEN** reports equal those of the previous version

### Requirement: One subsample for every score
With `max_samples`, one subsample of evaluation rows SHALL be drawn without replacement, using `random_state`, and the baseline score, the dummy's score, and every permuted score SHALL be computed on it. Train scores and evidence SHALL be unchanged. With the same `random_state`, the subsample SHALL be the same. Reports SHALL include `max_samples` (as given) and `n_eval_rows`.

#### Scenario: Same rows everywhere
- **WHEN** a recording scorer is used with `max_samples=50` on 200 evaluation rows
- **THEN** every scorer call sees the same 50 rows

#### Scenario: Reproducible
- **WHEN** ranking runs twice with `max_samples=0.5` and the same `random_state`
- **THEN** the reports are equal

### Requirement: Resampling, selector, and CLI
`rank_resampled` SHALL subsample each split's evaluation rows. `ConsensusSelector(max_samples=None)` SHALL pass it to `rank_resampled`. The CLI SHALL accept `--max-samples` (an int, or a float with a decimal point).

#### Scenario: Selector
- **WHEN** `ConsensusSelector(max_samples=0.5)` is fit
- **THEN** every split report has `n_eval_rows` equal to half its evaluation rows, rounded down
