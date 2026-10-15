## Purpose

Let callers turn off FeatRanker's progress bars, which today always print, so that notebooks, logs, and test output stay clean.

## ADDED Requirements

### Requirement: verbose controls progress bars
`FeatureRanker` SHALL accept a keyword argument `verbose` (bool, default `True`).
- When `verbose` is `True`, model initialization, `fit()`, and `rank_features()` SHALL show progress bars as in 0.4.
- When `verbose` is `False`, none of these steps SHALL write progress output.
- Warnings SHALL be emitted regardless of `verbose`.
- A non-bool `verbose` SHALL raise `TypeError`.

#### Scenario: Quiet run
- **WHEN** a ranker built with `verbose=False` runs `fit()` and `rank_features()`
- **THEN** nothing is written to stderr by FeatRanker's progress reporting

#### Scenario: Default unchanged
- **WHEN** a ranker is built without `verbose`
- **THEN** progress bars are shown during initialization, fit, and ranking

#### Scenario: Warnings still raised when quiet
- **WHEN** `verbose=False` and no model beats the dummy baseline
- **THEN** the `UserWarning` about the empty consensus is still emitted

#### Scenario: Invalid verbose
- **WHEN** `FeatureRanker(task="reg", verbose="no")` is called
- **THEN** a `TypeError` is raised

### Requirement: CLI quiet flag
The CLI SHALL accept `--quiet`, which runs the ranker with `verbose=False`. Without the flag, CLI output SHALL be unchanged.

#### Scenario: CLI quiet
- **WHEN** `featranker ... --quiet --output report.json` runs successfully
- **THEN** no progress bars are written to stderr and `report.json` is written
