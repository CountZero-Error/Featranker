## ADDED Requirements

### Requirement: Older pickles stay usable
A `ConsensusSelector` unpickled from a version without `max_shadow_q` SHALL behave as `max_shadow_q=None`: `transform()`, `get_params()`, `clone()`, and `fit()` SHALL work, and a refit SHALL NOT fail on the missing attribute.

#### Scenario: Pickle from 1.3
- **WHEN** a fitted selector whose state lacks `max_shadow_q` is unpickled
- **THEN** `transform(X)` returns the same columns as before pickling, `get_params()["max_shadow_q"] is None`, and `clone(selector).fit(X, y)` succeeds
