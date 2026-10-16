## Purpose

Define what FeatRanker's public API is from 1.0 onward and how it may change, and make the package's type information available to users' tools.

## ADDED Requirements

### Requirement: Public names
`featranker.__all__` SHALL be exactly `ConsensusSelector`, `FeatureRanker`, `Report`, `explain`, `summarize`, and `to_prompt`. `featranker.__version__` SHALL equal the installed package version, or `None` when the package metadata is unavailable.

#### Scenario: Exports
- **WHEN** `featranker.__all__` is read
- **THEN** it lists exactly those six names, each importable from `featranker`

### Requirement: Typed package
The distribution SHALL include `featranker/py.typed`. `mypy featranker` (with `types-PyYAML`) SHALL report no errors, and CI SHALL run it.

#### Scenario: Marker shipped
- **WHEN** the wheel is built
- **THEN** it contains `featranker/py.typed`

### Requirement: Extras match the registry
Every optional-dependency extra SHALL install something FeatRanker uses. The `lightgbm` extra SHALL be removed. `all` SHALL install `xgboost` and `catboost`, and a `plot` extra SHALL install `matplotlib`.

#### Scenario: Extras
- **WHEN** `pyproject.toml` is read
- **THEN** the extras are `xgboost`, `catboost`, `all` (xgboost and catboost), `plot` (matplotlib), and `test`
