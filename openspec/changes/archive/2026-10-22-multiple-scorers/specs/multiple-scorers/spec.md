## Purpose

Check a ranking under several metrics in one run, while one primary scorer drives the gate, consensus, shadows, and selection.

## ADDED Requirements

### Requirement: Scorer list
`scoring` SHALL accept a scorer name, a callable, or a non-empty list or tuple of them. Names SHALL be unique (a callable is named by its `__name__`, a `make_scorer` object by its score function's; two with the same name SHALL be told apart by wrapping one in a function); an empty list, a duplicate name, or an invalid entry SHALL raise before any model is ranked. With a single scorer, given alone or in a one-element list, the report SHALL be unchanged. The first entry SHALL be primary: `scoring` in the report SHALL name it, and the dummy gate, consensus, shadow evidence, stability, and selection SHALL use it alone.

#### Scenario: Primary unchanged
- **WHEN** the same data is ranked with `scoring="accuracy"` and with `scoring=["accuracy", "neg_log_loss"]`
- **THEN** both reports have equal `consensus`, primary importances, and `baseline_score`

#### Scenario: Duplicate
- **WHEN** `scoring=["r2", "r2"]`
- **THEN** `ValueError` is raised

### Requirement: Secondary importances
For each secondary scorer, every ranked model report SHALL include `secondary[name]` with `evaluation_score` and `importance` (`mean`, `std` per feature group), computed on the same permuted copies as the primary importances. The report SHALL include `secondary_scoring` (the secondary names in order) and `secondary_baseline_scores` (the fitted dummy's score per secondary scorer, or `null` when unavailable). A secondary scorer that raises for a model SHALL set that model's `secondary[name]` to `null` and add a failure with stage `"evidence"`; the model SHALL stay ranked.

#### Scenario: Secondary values
- **WHEN** a model is ranked with `["r2", "neg_mean_absolute_error"]`
- **THEN** its `secondary["neg_mean_absolute_error"]["evaluation_score"]` equals that scorer applied to the model on the evaluation data

#### Scenario: Secondary failure
- **WHEN** a secondary scorer raises for one model
- **THEN** that model is ranked, its secondary entry is `null`, and `failures` has an `"evidence"` entry for it

### Requirement: Resampling, selector, and CLI
`rank_resampled` and `ConsensusSelector` SHALL accept the list; split reports SHALL carry secondary results and stability SHALL use the primary scorer. The CLI `--scoring` SHALL accept comma-separated names.

#### Scenario: CLI
- **WHEN** the CLI runs with `--scoring accuracy,neg_log_loss`
- **THEN** the report has `scoring == "accuracy"` and `secondary_scoring == ["neg_log_loss"]`
