## 1. Registry families

- [x] 1.1 Write a registry test: every shipped clf/reg entry has a non-empty `family`, and no family appears under both `linear` and `tree`. Confirm it fails.
- [x] 1.2 Add a `family:` key to every entry in `importance_config.yaml`, following the design.md taxonomy table, so that 1.1 passes.
- [x] 1.3 Record `name -> family` in `model_families_` in `init_models`. Custom models and entries without a family fall back to the model name.

## 2. Family-aware consensus

- [x] 2.1 Write tests:
  - "large family cannot outvote small families", with the exact numbers from the spec scenario;
  - "one model per family matches per-model consensus";
  - "gated-out models don't vote within family";
  - row fields `n_models`, `n_families`, and `family_ranks`;
  - custom models get their own name as family.

  Confirm they fail.
- [x] 2.2 Add `family` to each model report in `_rank_model`.
- [x] 2.3 Rewrite `_build_consensus` as a median of family medians. Add `n_families` and `family_ranks`, and keep the existing keys and sort order. The 2.1 tests pass.
- [x] 2.4 Run the full suite. Update only those existing assertions whose values legitimately change under family voting, and note each one in the PR description.
- [x] 2.5 Re-run the README diabetes quick start and update its shown output if the order or counts change.

## 3. Progress output

- [x] 3.1 Write tests:
  - `verbose=False` writes no progress to stderr (`capsys`) during init, fit, and rank;
  - by default, progress is shown;
  - a non-bool `verbose` raises `TypeError`;
  - the empty-consensus `UserWarning` is still emitted when quiet;
  - CLI `--quiet` writes no progress bars.

  Confirm they fail.
- [x] 3.2 Add the `verbose` parameter with validation. Thread it to the `tqdm(disable=...)` calls in `init_models` and `_run_models`.
- [x] 3.3 Add the CLI `--quiet` flag that passes `verbose=False`.

## 4. Report summary

- [x] 4.1 Write tests for `summarize`:
  - each rule from the spec table: `gate_off`, `empty_consensus`, `failures` with info and warning severities, `few_families`, `tied_groups` with one finding per tie set, and `within_noise` with the majority rule;
  - rule order;
  - purity and determinism;
  - strict-JSON output;
  - a clean report yields no warnings;
  - `TypeError` and `ValueError` on bad input;
  - a 0.4-style report without `n_families` skips `few_families`.

  Confirm they fail.
- [x] 4.2 Implement `featranker/summary.py` with the named constants, and export `summarize` from `featranker/__init__.py`.

## 5. Docs

- [x] 5.1 `docs/report.md`:
  - explain family voting in step 4 of the reading guide;
  - add `n_families` and `family_ranks` to the field table and the schema example;
  - update the Ranking section;
  - point to `summarize`.
- [x] 5.2 `docs/usage.md`:
  - add `verbose` to the constructor table;
  - add `--quiet` to the CLI section;
  - add a short `summarize` section with an example.
- [x] 5.3 `docs/models.md`: add a Family column to both model tables.
- [x] 5.4 `README.md`: update the "Rank consensus" card to say families vote, not models.
- [x] 5.5 Verify that every new doc claim against the code passes, and render all changed docs through the GitHub markdown API, checking for broken anchors and escaped tags.

## 6. Release prep

- [x] 6.1 Bump `pyproject.toml` to 0.5.0. Add a CHANGELOG entry that names the consensus-value change for built-in rankers.
- [x] 6.2 Run the full test suite outside the sandbox (joblib needs process spawning), and confirm everything passes.
