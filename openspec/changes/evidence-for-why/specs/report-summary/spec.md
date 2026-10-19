## ADDED Requirements

### Requirement: Overfitting models are reported
When model reports include `train_score` and `baseline_score` is not `None`, `summarize` SHALL emit one `overfit` finding, with severity `info`, listing every model whose `train_score - baseline_score > 0` and whose `(train_score - evaluation_score) / (train_score - baseline_score) >= 0.5`, that is, a model that lost at least half of its training advantage over the dummy on the evaluation data. Each listed model SHALL carry its `train_score`, `evaluation_score`, and that share. Models with a `null` `train_score` SHALL be skipped. The finding SHALL come after `below_shadow`. Without `train_score`, no `overfit` finding SHALL be emitted.

#### Scenario: Overfit tree
- **WHEN** a model has `train_score` 1.0, `evaluation_score` 0.2, and `baseline_score` is 0.0
- **THEN** an `overfit` finding lists it with share 0.8

#### Scenario: Healthy model
- **WHEN** a model has `train_score` 0.56, `evaluation_score` 0.64, and `baseline_score` is 0.0
- **THEN** it is not listed

### Requirement: Correlated groups are reported
When the report has `evidence.correlation.pairs` and it is non-empty, `summarize` SHALL emit one `correlated_groups` finding, with severity `info`, listing the pairs and their `rho`, and saying that permutation importance may be split between correlated groups. The finding SHALL come after `overfit`. Otherwise no `correlated_groups` finding SHALL be emitted.

#### Scenario: Correlated pair
- **WHEN** `evidence.correlation.pairs` is `[{"groups": ["e", "f"], "rho": 0.99}]`
- **THEN** a `correlated_groups` finding lists `e` and `f`

#### Scenario: Evidence stage failures
- **WHEN** `failures` contains a failure with stage `"evidence"`
- **THEN** the `failures` rule emits a finding for that stage with severity `"info"`
