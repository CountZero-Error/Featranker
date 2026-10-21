## Context

Reports carry `provenance` (library versions) but no data or split identity. `auto_groups` already validates `threshold`. `rank_resampled` builds one report per split; failed splits appear as `failures` entries with stage `"split"`. `summarize()` raises on resampled reports. See proposal.md.

## Decisions

### D1. Fingerprint
`_fingerprint(names, X_values, y_values)`: SHA-256 over `json.dumps(names)`, then each array's `dtype.str`, `shape`, and bytes (`np.ascontiguousarray(...).tobytes()`); object arrays (categorical columns, string targets) are hashed by `repr` of each element instead of bytes. NaNs are normalised to one bit pattern before hashing. `fit()` stores `train_data_sha256_`; `__setstate__` defaults it to `None` for old pickles, and the report then carries `null`.

### D2. Split fingerprint and cv
`splits_sha256` hashes, per split in order, the train indices followed by the evaluation indices (`int64` bytes). `cv` is `repr(cv)` for splitter objects and the integer itself for `cv=int`.

### D3. Where `auto_groups` is recorded
Ranking functions do not know that groups were proposed automatically, so the caller records it: the selector and the CLI set `report["run"]["auto_groups"]` after ranking. For resampled reports, only the top-level `run` gets it.

### D4. Resampled summary
`summarize()` dispatches on `evaluation_mode`. The resampled rules read `splits`, `failures` (stage `"split"` versus model stages), `stability`, and `shadow_test`. Single-split rules are unchanged. 0.05 is a named constant (`SPLIT_Q_LEVEL`).

### D5. Selector warning
After ranking: `silent = n_splits - sum(1 for s in report["splits"] if s["report"]["consensus"])` counts splits that failed or produced no consensus; warn when `silent > 0`. This is independent of the existing "no group selected" warning.

## Risks / Trade-offs

- [Fingerprints change across NumPy/pandas versions for object arrays] → documented; `provenance` already records the versions.
- [Hashing large data costs time] → one pass over bytes, small next to model fitting.
