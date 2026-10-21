## Context

`benchmarks/run.py` has the planted-signal generator and selector wrappers. FeatRanker 1.6 adds per-family evidence, `min_shadow_families`, and `shadow_mode`. See proposal.md.

## Decisions

### D1. Grid and data
- Null: `n_rows=1000`, 10 standard-normal features, target independent of them (regression: standard normal; classification: Bernoulli(0.5)). Seeds 100–119.
- Signal: `benchmarks/run.py` generator, standard variant, seeds 100–109.
- Both tasks; `group` ∈ {`fast`, `tree`}; `shadow_mode` ∈ {`joint`, `independent`}; `cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)`; `n_repeats=10`; `random_state=seed`; `shadow_seed=seed`; scorers `r2` / `roc_auc`; `n_jobs=-1`.
- One `rank_resampled` report per (study, task, group, mode, seed); every rule is applied to the same report, so rules are compared on identical rankings.

### D2. Metrics
- Null: per configuration, the fraction of all null groups with `shadow_p_value <= 0.05` (with a Clopper–Pearson 95% interval), and for each rule the fraction of null groups selected and the fraction of runs selecting at least one group.
- Signal: per rule, recall of the 5 signal features, noise features selected, and the copy (neutral); c·d interaction recall reported separately.

### D3. Preregistered decisions (frozen with the script hash)
- **Q (hits test).** For a (task, group, mode) configuration, if the observed fraction of null groups with `p <= 0.05` is at most 0.05 and its 95% upper bound is at most 0.075, docs may state "on this study's null data, the binomial reference values were conservative for built-in `<group>` models"; otherwise docs report the measured excess. `calibrated` stays `false` either way (custom estimators are untested).
- **F (family rule).** The recommended `min_shadow_families` per task is the smallest N in {1, 2, 3} whose null group-selection fraction is at most 0.05 in every configuration of that task **and** whose signal-study interaction recall is at least 0.8 under `group="fast"`. If no N meets both, docs say so and show the trade-off; the 1.6 docs advice (N=2) is replaced by the measured table.
- **S (shadow mode).** Docs recommend `shadow_mode="independent"` for collinear data only if, in the signal study, it raises mean recall of the correlated source `e` or keeps the copy without raising any null selection fraction above 0.05. The package default stays `"joint"` in 1.x regardless.
- No other conclusion is drawn; any change to the script after freezing is listed as a deviation and the full study is re-run.

### D4. Freeze and run
`benchmarks/calibration.py` writes its own SHA-256 into the result file; the hash is recorded in `docs/calibration.md` before the run. Runtime budget about 60 minutes; if a configuration exceeds it, drop `group="tree"` for the null study first, and record the drop.

## Risks / Trade-offs

- [20 null seeds × 10 groups = 200 groups per configuration: the 95% interval on a 5% rate is about ±3%] → the decision uses the upper bound, so a borderline result does not earn a claim.
- [Synthetic null only covers independent rows] → stated; grouped or time-ordered data remains a known limitation.
