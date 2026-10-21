## ADDED Requirements

### Requirement: Warning when splits did not vote
When at least one split failed or produced an empty consensus, `ConsensusSelector.fit` SHALL emit a `UserWarning` stating how many of the splits did not vote and that their shares count them as misses. When every split voted, no such warning SHALL be emitted.

#### Scenario: One failed split
- **WHEN** one of five splits fails
- **THEN** fit warns that 1 of 5 splits did not vote and still selects

#### Scenario: All splits vote
- **WHEN** every split produces a consensus
- **THEN** no split warning is emitted
