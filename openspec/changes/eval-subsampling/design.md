## Context

`rank_features` validates and encodes the evaluation data, then scores the dummy, each model's baseline, and each permuted copy. See proposal.md.

## Decisions

### D1. Draw once, after validation
After `X_values`/`y_values` are validated and encoded, draw `np.random.default_rng([random_state, 2]).choice(n, size, replace=False)` (seed `None` stays `None`; the `2` keeps it apart from the permutation RNG), sorted to keep row order. The baseline, dummy and permuted scores use only those rows. The shadow columns and the SHAP evidence are still built from every row, and `run.data_sha256` covers every row passed. `overfit_gap` is the train score minus the subsampled baseline. The permutation RNG is seeded as before, so with `max_samples=None` nothing changes.

### D2. Validation
`_resolve_max_samples(value, n)`: reject bools; ints ≥ 1; floats in (0, 1]; return the row count.

### D3. CLI parsing
`--max-samples` parses `"50"` as int and `"0.5"` as float.

## Risks / Trade-offs

- [A small subsample can drop a rare class, and a scorer such as `roc_auc` can then fail] → the failure is recorded as usual; docs advise a subsample large enough for every class.
