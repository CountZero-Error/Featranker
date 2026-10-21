## Context

`_resolve_scorer` returns one scorer and name. `_rank_model` scores the baseline, calls `_permutation_importances` (which scores each permuted copy), and builds the model report. See proposal.md.

## Decisions

### D1. Resolution
`_resolve_scorers(scoring) -> list[(scorer, name)]`. A string or callable becomes a one-element list. The primary is `[0]`; existing code keeps receiving the primary scorer and name.

### D2. One permutation loop
`_permutation_importances` takes the secondary scorers and, for each permuted copy, scores it with the primary and then each secondary, so the shuffles are shared and the primary results are bit-identical to a single-scorer run. A secondary that raises is caught per scorer: its accumulator is dropped and the error kept for the `evidence` failure.

### D3. Dummy
`secondary_baseline_scores` scores the fitted dummy with each secondary scorer; a failure there is `null` (not a failure entry: the dummy is not ranked).

## Risks / Trade-offs

- [Time grows with the number of scorers] → each scorer re-predicts; documented, and `max_samples` (1.13) offsets it.
