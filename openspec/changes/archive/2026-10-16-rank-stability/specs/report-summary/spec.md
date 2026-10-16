## ADDED Requirements

### Requirement: Resampled reports are rejected with guidance
`summarize` SHALL raise `ValueError` when given a report with `evaluation_mode == "resampled"`. The message SHALL say to summarize each entry of `report["splits"]` (its `report` field) instead.

#### Scenario: Resampled report
- **WHEN** `summarize(resampled_report)` is called
- **THEN** a `ValueError` mentioning `splits` is raised
