## ADDED Requirements

### Requirement: Groups below shadow are reported
When consensus rows include `above_shadow_share`, `summarize` SHALL emit one `below_shadow` finding, with severity `info`, listing every consensus group whose `above_shadow_share < 0.5` along with its share. The finding SHALL come after `within_noise`. Without `above_shadow_share`, no `below_shadow` finding SHALL be emitted.

#### Scenario: Below shadow
- **WHEN** consensus groups `a` and `b` have `above_shadow_share` 1.0 and 0.25
- **THEN** one `below_shadow` finding lists `b` with share 0.25

#### Scenario: No shadows
- **WHEN** the report has no `above_shadow_share`
- **THEN** no `below_shadow` finding is emitted
