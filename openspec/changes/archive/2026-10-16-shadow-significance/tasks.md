## 1. Options and construction

- [x] 1.1 Write tests:
  - option validation (`TypeError` cases);
  - `shadows=False` reports have no shadow keys and equal a 0.6-style run;
  - a recording estimator sees `2 * n_features` columns whose shadow rows all come from the training rows;
  - the dummy sees `n_features` columns;
  - a 0.6 pickle loads with shadows off.

  Confirm they fail.
- [x] 1.2 Add `shadows`/`shadow_seed` to the constructor with validation and `__setstate__` defaults. Build the shadow block in `fit()` (models only, not the dummy).

## 2. Shadow importance

- [x] 2.1 Write tests:
  - the evaluation shadow block comes from the evaluation rows;
  - the report has `shadow_importance` (one entry per group), `shadow_max`, and `above_shadow`;
  - a strong-signal group beats the shadow;
  - real ranks are the same with shadows on and off for `LinearRegression`;
  - seeded runs are reproducible;
  - the report is strict JSON.

  Confirm they fail.
- [x] 2.2 Implement the evaluation shadow block, plus shadow importance and the comparison in `_rank_model`.

## 3. Consensus and stability

- [x] 3.1 Write tests:
  - the family-weighted `above_shadow_share` scenario from the spec;
  - no share key without shadows;
  - the stability share is a mean over ranked splits (1.0, 0.5, `null` gives 0.75);
  - `rank_resampled` uses `shadow_seed + i` per split.

  Confirm they fail.
- [x] 3.2 Implement the share in `_build_consensus` and `_build_stability`, and pass the shadow settings through in `rank_resampled`.

## 4. summarize

- [x] 4.1 Write tests: the `below_shadow` finding (info, lists groups with share < 0.5, after `within_noise`), and no finding without shares. Confirm they fail.
- [x] 4.2 Implement the rule in `featranker/summary.py`.

## 5. Docs

- [x] 5.1 `docs/usage.md`:
  - add `shadows`/`shadow_seed` to the constructor table;
  - add a "Shadow features" section with an example run, the cost, the pipeline-column caveat, the fact that it is descriptive and not a p-value, and the note that ranks can shift when models are sensitive to extra columns;
  - add `below_shadow` to the summarize table.
- [x] 5.2 `docs/report.md`: add the shadow fields to the reading guide, the schema notes, and the resampled stability table.
- [x] 5.3 `README.md` and `docs/README.md`: update the docs-table entries.
- [x] 5.4 Verify the doc claims and example output against the code, then render through the GitHub API and check anchors and escaped tags.

## 6. Release prep

- [x] 6.1 Bump to 0.7.0. Add a CHANGELOG entry.
- [x] 6.2 Run the full suite outside the sandbox.
