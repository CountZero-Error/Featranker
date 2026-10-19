## MODIFIED Requirements

### Requirement: Public names
`featranker.__all__` SHALL be exactly `ConsensusSelector`, `FeatureRanker`, `Report`, `auto_groups`, `explain`, `summarize`, and `to_prompt`. `featranker.__version__` SHALL equal the installed package version, or `None` when the package metadata is unavailable.

#### Scenario: Exports
- **WHEN** `featranker.__all__` is read
- **THEN** it lists exactly those seven names, each importable from `featranker`
