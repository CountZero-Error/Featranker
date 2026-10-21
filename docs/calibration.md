# FeatRanker ▍ Calibration study

<sub><a href="README.md">◂ Back to docs</a></sub>

FeatRanker 1.5 labels the [shadow hits test](usage.md#shadow-test) as uncalibrated, and 1.6 added `min_shadow_families` and `shadow_mode` without measuring them. This study measures, on synthetic data with fresh seeds and decision rules written down before the run: how often each shadow rule selects pure noise, how often it finds planted signal, and whether independent shadow shuffling matters for collinear features. The code is [`benchmarks/calibration.py`](https://github.com/CountZero-Error/Featranker/tree/main/benchmarks) (not part of the package, not run by CI); the numbers come from `benchmarks/results/20261020-calibration.json`.

> [!NOTE]
> 2026-10-21: the `share` rows here (`min_shadow_share`) were computed when the resampled stability `above_shadow_share` averaged only over the splits that had a consensus. FeatRanker 1.8 divides by `n_splits` instead (see [the report reference](report.md)), which removes the one-lucky-split mechanism described under Observations; the `q` and family rows are unaffected. This study was not re-run, so the effect on the false-selection rate was not measured.

## Short version

- **Hits test (decision Q): allowed.** On this study's null data (independent features and target), no group in any of the 8 configurations had `shadow_p_value <= 0.05` (0 of 160 to 200 groups per configuration; upper 95% bounds 1.8% to 2.3%). The preregistered rule allows the claim "on this study's null data, the binomial reference values were conservative for built-in `fast` and `tree` models", but read it with the caveats below: with 10 splits the test's real level is 0.0107 (it needs at least 9 hits in 10 splits), so "conservative" is partly built in, and in regression most null runs had too few voting splits for the rule to fire at all (post-hoc observation, see Observations). Classification is the informative null. `calibrated` stays `false`: custom estimators, other data, and dependent rows were not tested.
- **`min_shadow_families` (decision F): the computed rule recommends N = 1 for both tasks.** N = 1 and N = 2 selected almost no null groups (at most 1% of null groups, in classification) and found the `c·d` interaction in every seed with `group="fast"`; N = 3 passes in regression (`c·d` recall 0.90) but fails in classification (0.05 to 0.15). The rule did not look at noise selected in the signal study. Post hoc: there N = 1 selected some noise (up to 0.4 of 4 noise features per run) and N = 2 at most 0.1, with no loss of `c·d` recall; the null cannot separate N = 1 from N = 2 in regression. The study does not single out N = 2 as its recommendation; see Observations.
- **`shadow_mode` (decision S): not recommended.** Independent shadows gave a small pooled gain in recall of `e` (+0.03) and in keeping the copy (+0.045), both within seed noise, and the rule's null guard was tripped by one cell only (`share`, `tree`, regression: 0.037 to 0.067), a cell affected by the empty-consensus effect below. The study therefore gives no evidence for `independent` either way. The default stays `"joint"`.
- **`max_shadow_q=0.05` never selected null noise and has low power at 10 splits**: recall 0.52 to 0.90, and `c·d` recall 0.00 to 0.75 (classification `fast`: 0.00).
- **`min_shadow_share=0.5` selected noise in the regression null** (`fast` models: 7 to 11% of null groups, 7 to 9% counting every group of every run; 35 to 40% of runs selected at least one noise group), but that is largely an artifact of how the stability share is averaged (see Observations), and in the signal study `share` kept 0.00 noise features while `fam1` kept up to 0.4. It is not simply the loosest rule.

## Method

### Data

- **Null study.** `n_rows = 1000`, 10 standard-normal features, a target independent of them (regression: standard normal; classification: Bernoulli(0.5)). Seeds 100 to 119 (20 runs per configuration). A "null group" is one of the 10 features.
- **Signal study.** The `benchmarks/run.py` generator (linear `a`, threshold `b`, interaction `c·d`, linear `e`, near-copy `f` of `e`, four noise features; see [the benchmark](benchmark.md#data)), standard variant, `n_rows = 1000`. Seeds 100 to 109 (10 runs per configuration). Fresh seeds: the benchmark used 0 to 4.

### Grid and rules

Both tasks (`reg`, `clf`); `group` in `fast`, `tree`; `shadow_mode` in `joint`, `independent`. One `FeatureRanker.rank_resampled()` report per (study, task, group, mode, seed), with `RepeatedKFold(5, 2)` (10 splits), `n_repeats=10`, `random_state=seed`, `shadow_seed=seed`, scorer `r2` or `roc_auc`, `n_jobs=-1`. Every rule is applied to the stability table of that one report, with the same comparisons `ConsensusSelector` uses, so rules are compared on identical rankings:

| Rule | Keeps a group when |
|---|---|
| `share` | `above_shadow_share >= 0.5` (`min_shadow_share=0.5`) |
| `q` | `shadow_q_value <= 0.05` (`max_shadow_q=0.05`) |
| `fam1`, `fam2`, `fam3` | `shadow_families >= N` (`min_shadow_families=N`) |

### Metrics

- **Null:** the fraction of null groups with `shadow_p_value <= 0.05` (exact Clopper-Pearson 95% interval, which treats the groups of one run as independent trials; they are not exactly, so read the interval as approximate); per rule, the fraction of null groups selected and the fraction of runs selecting at least one.
- **Signal:** per rule, mean over seeds of recall of the 5 signal features, noise features selected (0 to 4), whether the copy `f` was kept (neutral), the recall of the interaction pair `c`, `d` (0, 0.5, or 1 per run), and whether `e` was selected.

### Decisions, written before the run

- **Q (hits test).** For a (task, group, mode) configuration: if the fraction of null groups with `p <= 0.05` is at most 0.05 and its 95% upper bound is at most 0.075, docs may say "on this study's null data, the binomial reference values were conservative for built-in `<group>` models"; otherwise they report the measured excess. `calibrated` stays `false` either way.
- **F (family rule).** Per task, the smallest N in {1, 2, 3} whose null group-selection fraction is at most 0.05 in every null configuration of that task, **and** whose signal-study `c·d` recall is at least 0.8 under `group="fast"`. Settled in code before freezing, where the design text left it open: "every configuration" is all four group and mode cells; "under `group="fast"`" means both shadow modes, and the smaller recall counts.
- **S (shadow mode).** Recommend `independent` for collinear data only if, pooled over tasks, groups and rules in the signal study, it has a higher mean recall of `e` or a higher rate of keeping the copy than `joint`, and no (task, group, rule) null selection fraction is raised by it above 0.05 (independent above 0.05 and above joint). The default stays `"joint"` in 1.x regardless. Settled in code before freezing, where the design text left it open: the comparison is pooled over tasks, groups and rules; "raised" means independent above 0.05 and above joint; any positive pooled gain counts (no minimum size).

No other conclusion is drawn from the study's numbers; the "Observations" section is marked as such.

## Frozen setup

Fixed before the full run. A result file counts only if its `calibration_sha256` equals this hash; any later change to `benchmarks/calibration.py` is listed under Deviations and the whole study is re-run.

| Item | Value |
|---|---|
| `benchmarks/calibration.py` SHA-256 | `3ea044ade77e9cdaf696e493d9b50b3b9bf3becbf674aac81cf629025886bdd2` |
| Package under test | FeatRanker 1.6.0, commit `a1c7380`, run from a frozen copy of that commit |
| Null study | 1000 rows, 10 standard-normal features, target independent (regression: standard normal; classification: Bernoulli(0.5)); seeds 100 to 119 |
| Signal study | `benchmarks/run.py` generator, standard variant, seeds 100 to 109 |
| Grid | tasks `reg`, `clf`; `group` in `fast`, `tree`; `shadow_mode` in `joint`, `independent` |
| Per report | `rank_resampled`, `RepeatedKFold(5, 2, random_state=seed)`, `n_repeats=10`, `random_state=seed`, `shadow_seed=seed`, scorer `r2` / `roc_auc`, `n_jobs=-1` |
| Rules | `share`: `above_shadow_share >= 0.5`; `q`: `shadow_q_value <= 0.05`; `fam1`, `fam2`, `fam3`: `shadow_families >= N`; all applied to the same report |
| Decision rules | Q, F, S of the design (D3), computed by `calibration.py` |

## Environment

| Item | Value |
|---|---|
| FeatRanker | 1.6.0, commit `a1c7380`, imported from a frozen copy of that commit (package source SHA-256 `1068717ab3e74e2f4426d52bf5f2c6c7917c2094bbd8f1b26a90342c85f3cae0`), so concurrent edits to the working tree could not enter the run |
| `benchmarks/run.py` (generator) | from the repository; identical to the file at `a1c7380` (SHA-256 `be1f53e84090940f619e6789eea240963d6fb23e76862aa6ab9ce4222dab97be`) |
| Python | 3.12.7 |
| numpy / scipy / scikit-learn / pandas | 1.26.4 / 1.13.1 / 1.5.1 / 2.2.2 |
| xgboost / catboost | 3.1.3 / 1.2.8 (used by the `tree` group) |
| Machine | macOS arm64, 10 cores, `n_jobs=-1` |
| Run time | 48.4 minutes in total for 240 reports (signal 15.9, null 32.4); nothing dropped, no errors |

## Results

**Null study: shadow_p_value hits and noise groups selected (fraction of all null groups / fraction of runs with at least one)**

| Task | Group | Mode | Runs | p <= 0.05 | 95% CI | share | q | fam1 | fam2 | fam3 |
|---|---|---|---|---|---|---|---|---|---|---|
| reg | fast | joint | 20 | 0.000 (0/160) | 0.000 to 0.023 | 0.113 / 0.35 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| reg | fast | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.070 / 0.40 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | fast | joint | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.010 / 0.10 | 0.000 / 0.00 | 0.010 / 0.05 | 0.010 / 0.05 | 0.010 / 0.05 |
| clf | fast | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.010 / 0.10 | 0.000 / 0.00 | 0.010 / 0.10 | 0.005 / 0.05 | 0.005 / 0.05 |
| reg | tree | joint | 20 | 0.000 (0/160) | 0.000 to 0.023 | 0.037 / 0.20 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| reg | tree | independent | 20 | 0.000 (0/180) | 0.000 to 0.020 | 0.067 / 0.35 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | tree | joint | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 |
| clf | tree | independent | 20 | 0.000 (0/200) | 0.000 to 0.018 | 0.000 / 0.00 | 0.000 / 0.00 | 0.005 / 0.05 | 0.000 / 0.00 | 0.000 / 0.00 |

**Signal study: mean over seeds**

| Task | Group | Mode | Rule | Recall | Noise selected | Copy kept | c,d recall | e recall | Runs |
|---|---|---|---|---|---|---|---|---|---|
| reg | fast | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | q | 0.90 | 0.00 | 1.00 | 0.75 | 1.00 | 10 |
| reg | fast | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | joint | fam3 | 0.96 | 0.00 | 1.00 | 0.90 | 1.00 | 10 |
| reg | fast | independent | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | q | 0.86 | 0.00 | 1.00 | 0.65 | 1.00 | 10 |
| reg | fast | independent | fam1 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | fast | independent | fam3 | 0.98 | 0.00 | 1.00 | 0.95 | 1.00 | 10 |
| clf | fast | joint | share | 0.60 | 0.00 | 1.00 | 0.00 | 1.00 | 10 |
| clf | fast | joint | q | 0.52 | 0.00 | 0.40 | 0.00 | 0.60 | 10 |
| clf | fast | joint | fam1 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | joint | fam3 | 0.62 | 0.00 | 1.00 | 0.05 | 1.00 | 10 |
| clf | fast | independent | share | 0.66 | 0.00 | 1.00 | 0.15 | 1.00 | 10 |
| clf | fast | independent | q | 0.60 | 0.00 | 0.90 | 0.00 | 1.00 | 10 |
| clf | fast | independent | fam1 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | independent | fam2 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| clf | fast | independent | fam3 | 0.66 | 0.10 | 1.00 | 0.15 | 1.00 | 10 |
| reg | tree | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | q | 0.90 | 0.00 | 1.00 | 0.75 | 1.00 | 10 |
| reg | tree | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | joint | fam3 | 0.74 | 0.00 | 0.90 | 0.35 | 1.00 | 10 |
| reg | tree | independent | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | q | 0.86 | 0.00 | 1.00 | 0.65 | 1.00 | 10 |
| reg | tree | independent | fam1 | 1.00 | 0.10 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| reg | tree | independent | fam3 | 0.70 | 0.00 | 1.00 | 0.25 | 1.00 | 10 |
| clf | tree | joint | share | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | q | 0.64 | 0.00 | 0.70 | 0.10 | 1.00 | 10 |
| clf | tree | joint | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | joint | fam3 | 0.56 | 0.00 | 0.70 | 0.05 | 0.70 | 10 |
| clf | tree | independent | share | 0.98 | 0.00 | 1.00 | 0.95 | 1.00 | 10 |
| clf | tree | independent | q | 0.68 | 0.00 | 0.70 | 0.20 | 1.00 | 10 |
| clf | tree | independent | fam1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | independent | fam2 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 10 |
| clf | tree | independent | fam3 | 0.58 | 0.00 | 1.00 | 0.00 | 0.90 | 10 |

The null table lists, for each rule, the fraction of null groups selected, then the fraction of runs with at least one null group selected. "Runs" counts the reports of the configuration; "groups" in the `p <= 0.05` column counts only groups that have a stability row (see Limits: 10 regression runs had none).

### Decisions as computed by the script

**Decision Q (hits test), per configuration**

- `fast` models, reg, shadow_mode=joint: 0.0% (95% CI 0.0% to 2.3%, 0 of 160) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, reg, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, clf, shadow_mode=joint: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `fast` models, clf, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `fast` models
- `tree` models, reg, shadow_mode=joint: 0.0% (95% CI 0.0% to 2.3%, 0 of 160) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, reg, shadow_mode=independent: 0.0% (95% CI 0.0% to 2.0%, 0 of 180) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, clf, shadow_mode=joint: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models
- `tree` models, clf, shadow_mode=independent: 0.0% (95% CI 0.0% to 1.8%, 0 of 200) -> ALLOWED: on this study's null data, the binomial reference values were conservative for built-in `tree` models

**Decision F (min_shadow_families), per task**

- reg: recommended N = 1
  - N=1: worst null group-selection fraction 0.000, worst fast-group c,d recall 1.00, passes
  - N=2: worst null group-selection fraction 0.000, worst fast-group c,d recall 1.00, passes
  - N=3: worst null group-selection fraction 0.000, worst fast-group c,d recall 0.90, passes
- clf: recommended N = 1
  - N=1: worst null group-selection fraction 0.010, worst fast-group c,d recall 1.00, passes
  - N=2: worst null group-selection fraction 0.010, worst fast-group c,d recall 1.00, passes
  - N=3: worst null group-selection fraction 0.010, worst fast-group c,d recall 0.05, fails

**Decision S (shadow_mode)**

- independent recommended for collinear data: False (pooled gain in e recall +0.030, in copy kept +0.045; null fractions raised above 0.05: 1)
  - reg tree share: joint 0.037 -> independent 0.067

## Observations (not part of the preregistered decisions)

The voting-split counts and the `f8`/`f4` example below come from post-hoc probes in the Opus review of this study (run on the frozen copy, not preregistered, not in the result file); the p-values and every other number are read from the result file.

- **The regression null mostly measures the dummy gate.** Pure noise leaves most models below the dummy baseline, so few splits have a consensus at all. Number of the 10 splits with any consensus, per regression `fast` run (seeds 100 to 119), reviewer's probe: `joint` [0,0,1,2,2,2,1,2,0,1,4,2,2,0,1,3,3,2,3,2]; `independent` [1,1,2,5,3,1,3,2,2,1,3,2,5,1,3,1,4,3,4,1] (median 2). In the result file, 10 of the 160 regression null runs (4 `fast` joint, 4 `tree` joint, 2 `tree` independent) had no consensus in any split, so the report had no stability table; those runs select nothing and have no p-values, which is why some regression configurations show 160 or 180 groups instead of 200. With the missing groups counted (10 per run), `share` is 0.09 / 0.07 (`fast` joint / independent) and 0.03 / 0.06 (`tree`) for regression, and every other cell is unchanged or lower; no decision changes.
- **Consequences for Q and for the count rules in regression.** With at most a few voting splits, `p <= 0.05` (at least 9 hits in 10) and the `q` and family rules are mechanically impossible in most regression null runs. The smallest regression null p-values in the result file are 0.945 (`fast` joint), 0.623 (`fast` independent), 0.999 (`tree` joint) and 0.989 (`tree` independent), so 0% hits there is weak evidence. **Classification is the more informative null**: most splits vote (reviewer probe, minimum voting splits per run: `fast` 4 to 10 of 10, median 9; `tree`, seeds 100 to 105, 8 to 10), and the smallest p-values are 0.17 (`fast`) and 0.62 to 0.95 (`tree`). Q's "conservative" outcome also reflects the test's design: its real level at 10 splits is 0.0107 (at least 9 hits, `p = 11/1024`), and a noise feature has to beat the maximum of 10 shadows to be a hit, so its per-split hit probability is far below the 0.5 the test assumes. The test is conservative largely by construction, not close to calibrated, and a conservative test is also low-power: `q` recovers `c·d` poorly (signal table above).
- **Why `share` reaches 11.3% in regression.** Stability `above_shadow_share` averages only over the splits that have a consensus row, so one voting split can give a share of 1.0. Example (`fast` joint, seed 102): `f8` and `f4` have share 1.0 with a single hit and `shadow_p_value` 0.999. This is a property of the 1.6 package's stability average, not evidence that `share` is a worse rule on real signal (in the signal study it kept 0.00 noise features). FeatRanker 1.8.0 changed the stability share to average over all splits (`stability-denominator`); this study is not rerun for it.
- **N = 1 vs N = 2.** In the signal study, `fam1` selected noise in `fast` regression with independent shadows (0.40 of 4 noise features per run), in `fast` classification (0.20 joint, 0.10 independent) and in `tree` regression with independent shadows (0.10); `fam2` selected none in regression and 0.10 in `fast` classification with independent shadows. On null data both selected 0 to 1%, and the regression null cannot separate them (see above). `fam2` loses no `c·d` recall relative to `fam1` in this study. The computed recommendation of N = 1 is the smallest N that passes the preregistered rule; it ignores signal-study noise. Practical guidance (a judgement, not a study result): N = 1 if recall matters more, N = 2 if false selections matter more; avoid N = 3 with `fast` models in classification. One synthetic design with 10 seeds; still a heuristic.
- **Classification is harder for the count rules than for N = 1 and 2**: `fam3` needs three families to agree and found `c·d` in 0.00 to 0.15 of seeds.
- **No guidance to prefer `max_shadow_q` or `min_shadow_families` over `share` follows from this study.** On pure noise `q` and the family rules selected almost nothing (but see the regression caveat); in the signal study `share` kept 0.00 noise while `fam1` kept up to 0.4 per run, and `q` has low recall.

## Limits

- **Synthetic, independent rows, one size.** 1000 rows, 10 features, Gaussian features. Grouped or time-ordered data, more features, and other sample sizes are untested; overlapping splits make the intervals approximate.
- **Built-in models only** (`fast` and `tree` groups). Custom estimators are not covered, which is why `calibrated` stays `false`. The `linear` and `all` groups were not run.
- **Null power.** Null data tests false selections, not coverage under a real signal; the signal study has 10 seeds, so its means move in steps of 0.1 (recall of one feature) and differences of a few hundredths are noise.
- **The pure-noise regression target is filtered by the dummy gate**, so the null under-tests every rule that needs many voting splits (`p <= 0.05`, `q`, the family rules). The classification null exercises them much more fully, though not in every run.
- **Fewer null groups than planned** in three regression configurations (empty consensus); `frac_groups` excludes those runs' groups by design, which the observations above quantify.
- **Not recorded.** `STABILITY_KEYS` kept `above_shadow_share`, `shadow_p_value`, `shadow_q_value` and `shadow_families`, not `n_splits_ranked` or `shadow_hits`, so the number of voting splits per run can only be re-derived by rerunning the package. Errored runs would not have been counted in the summaries; none occurred. The summary would also fail (not silently mis-report) on a configuration in which every run had an empty consensus; none did.
- **Fixed in 1.8.0** (stability share over all splits, `stability-denominator`); the numbers here describe 1.6.0 and were not re-run.

## Deviations

None. `benchmarks/calibration.py` was not changed after its hash was recorded above; the result file's `calibration_sha256` equals that hash. The smoke run (1 seed, 300 rows) was made with an earlier version of `calibration.py` (SHA-256 `5b7b2e1e394a...`, the frozen file added a separate decision-Q text branch for the case where only the upper bound fails, as the smoke output shows; outcomes are computed the same way); the frozen version itself was not smoke-tested, but the reviewer re-ran 4 of the 240 reports with it and got identical stability tables. The smoke results are not used. Nothing was dropped.

## Reproduce

```bash
# FeatRanker 1.6.0 from commit a1c7380 on PYTHONPATH (not the working tree), plus xgboost and catboost
PYTHONPATH=<copy of the a1c7380 package> python benchmarks/calibration.py
```

This writes `benchmarks/results/<date>-calibration.json` (with `--resume` it continues a stopped run if the script hash matches) and a `.md` with the tables above. Check that `calibration_sha256` in the result equals the hash in "Frozen setup".
