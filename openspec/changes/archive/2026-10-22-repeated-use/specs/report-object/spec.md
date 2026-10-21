## MODIFIED Requirements

### Requirement: summary
`Report.summary()` SHALL return `summarize(self)`. For a resampled report it SHALL return the same resampled findings as `summarize`.

#### Scenario: Same findings
- **WHEN** `report.summary()` is called on a single-split report
- **THEN** it equals `summarize(report)`
