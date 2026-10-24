## ADDED Requirements

### Requirement: Shared predictions across scorers
With several scorers, each baseline and permuted copy SHALL be predicted once per response method for the scorers scikit-learn can cache that agree on the positive label; other scorers (a different `pos_label`, or with weights a score function without a `sample_weight` parameter) SHALL be called directly, and a model's scorers SHALL share only if its baseline scores through the shared path equal the direct ones exactly. Scores, importances, failures, and reports SHALL equal those of per-scorer scoring.

#### Scenario: One prediction for two regression scorers
- **WHEN** `scoring=["r2", "neg_mean_absolute_error"]` and a recording regressor
- **THEN** `predict` is called once per permuted copy, not twice

#### Scenario: Different positive label
- **WHEN** `scoring=["neg_log_loss", make_scorer(average_precision_score, response_method="predict_proba", pos_label=0)]`
- **THEN** the report equals the one without caching

#### Scenario: Same report
- **WHEN** a report is built with caching and with caching disabled
- **THEN** the reports are equal

### Requirement: Scorer failures unchanged
A failing secondary scorer SHALL fail alone with the same failure entry as before; a failing primary scorer SHALL keep its exception type and message. If the multi-metric scorer is unavailable or raises unexpectedly, scoring SHALL fall back to per-scorer calls.

#### Scenario: Secondary scorer raises
- **WHEN** a secondary callable scorer raises on every call
- **THEN** its failure entry equals the one produced without caching and the primary scores are unchanged

#### Scenario: Multi-metric scorer unavailable
- **WHEN** the multi-metric scorer cannot be imported
- **THEN** reports equal those with caching
