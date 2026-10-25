## MODIFIED Requirements

### Requirement: Category cap
`FeatureRanker(max_categories=None)` SHALL accept None or an int of at least 2; other values SHALL raise `ValueError`. When set, built-in linear models SHALL one-hot code at most that many levels per categorical column, grouping the rest as infrequent; an unseen level SHALL share the infrequent column only for a column with at least that many levels, and otherwise be all zeros. When None, pipelines and reports SHALL be unchanged. `ConsensusSelector` and the CLI (`--max-categories`) SHALL pass it through.

#### Scenario: High-cardinality column capped
- **WHEN** a categorical column has 1000 levels and `max_categories=20`
- **THEN** each built-in linear model's encoder outputs at most 20 columns for it

#### Scenario: Default unchanged
- **WHEN** `max_categories` is not given
- **THEN** reports equal those of the previous version

#### Scenario: Unseen level at equality
- **WHEN** a categorical column has exactly `max_categories` levels and an unseen level is encoded
- **THEN** it falls into the infrequent column
