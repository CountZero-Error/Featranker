## ADDED Requirements

### Requirement: Deprecation
`explain()` and `to_prompt()` SHALL each emit a `FutureWarning` on every call, stating that the function is deprecated since 1.1 and will be removed in 2.0. Their behaviour and return values SHALL otherwise be unchanged. Their docstrings and the docs SHALL mark them as deprecated.

#### Scenario: Warning
- **WHEN** `to_prompt(report)` is called
- **THEN** a `FutureWarning` mentioning 2.0 is emitted and the returned prompt equals the 1.0 prompt for the same report

#### Scenario: explain still works
- **WHEN** `explain(report, fake)` is called
- **THEN** a `FutureWarning` is emitted, `fake` is called once, and the result has the 1.0 shape
