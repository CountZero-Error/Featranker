# FeatRanker ▍ Benchmark

<sub><a href="README.md">◂ Back to docs</a></sub>

A reproducible comparison of FeatRanker's selector against common feature selectors on synthetic data where the true signal is known. It is a check of the claim that a multi-family, held-out, shadow-checked ranking is more trustworthy than a single-model selector, and it reports where the claim fails as well as where it holds.

The code is in [`benchmarks/`](https://github.com/CountZero-Error/Featranker/tree/main/benchmarks) (not part of the package, not run by CI); the preregistered results are in `benchmarks/results/20261019.json`, and every number outside the last section comes from that file, except where marked. The section [Added after seeing the results](#added-after-seeing-the-results-2026-10-20) uses a separate file, `benchmarks/results/20261020-posthoc.json`, and is not part of the headline comparison.

## Short version

- **False selections.** The four FeatRanker configurations selected **no noise feature in any of their 80 runs**. Boruta selected 7 noise features over its 20 runs, the permutation rule 5. Only `SelectByShuffling` also selected none, and it did so by missing most of the signal.
- **Regression.** `min_shadow_share=0.5` found all 5 signal features in all 10 runs of both variants with no noise, as did Boruta and the permutation rule (which each added some noise features, see the tables).
- **Classification: FeatRanker loses.** Boruta recovered all 5 signal features in every run; FeatRanker recovered 0.44 to 0.64 of them. It missed the interaction pair (`c`, `d`) almost every time: the family-majority shadow rule cannot see an effect that only some model families can represent (see the [post-hoc diagnostic](#added-after-seeing-the-results-2026-10-20)). The correlated copy `e`/`f` was also selected less often than a lone linear feature.
- **Runtime: FeatRanker loses.** Mean time per configuration was 3.2 to 19.4 s for FeatRanker and 0.3 to 4.8 s for the others; single runs ranged over 3.1 to 21.2 s and 0.22 to 9.93 s. Machine: 10 cores. Classification `featranker_share_fast` (3.2 to 3.3 s) is close to Boruta's classification means (2.1 and 3.2 s).
- **The `max_shadow_q` rule did not help here.** It scored at or below `min_shadow_share=0.5` on recall in every table, with zero noise either way, and costs about twice the time (10 resamples instead of 5).

## Method

### Data

Synthetic, `n_rows = 1000`, 10 features, drawn per seed from standard normals. Features play fixed roles, and the roles are assigned to randomly permuted columns for each seed (so no method gains from column position):

| Role | Definition | Counted as |
|---|---|---|
| `a` | linear: `2·a` | signal |
| `b` | threshold: `3·[b > 0]` | signal |
| `c`, `d` | interaction: `2·c·d` (neither has a marginal effect) | signal |
| `e` | linear: `1.5·e` | signal |
| `f` | `e + 0.1·noise`, a near-copy of `e` | neutral |
| 4 others | pure noise | noise |

Regression: `y = 2a + 3·[b>0] + 2cd + 1.5e + sd·ε`, with `sd = 1` (standard) or `2` (hard, same features and same ε, doubled noise). Classification: the same latent score minus 1.5 (to centre the threshold term), plus `sd·ε`, passed through a logistic link and sampled to a 0/1 label. Seeds 0 to 4, so each table cell averages 5 runs.

### Scoring

Per run, on the features a selector keeps: `recall = |selected ∩ signal| / 5`, `noise selected = |selected ∩ noise|` (0 to 4), and wall time of the fit/selection call only. The copy `f` is neutral: neither credited nor penalised, but the "Copy kept" column shows how often it was selected. Tables show mean ± sample standard deviation (n = 5) over seeds.

### Selectors and settings

All selectors see the same 1000 rows and are set to their documented defaults unless noted. The benchmark's own random forests use 100 trees; `random_state` is the seed for the selectors' splits and forests. Inside FeatRanker it is not the seed of everything: its built-in models use `random_state=42`, and the seed drives the cv splits, the permutations and the shadows.

| Selector | Settings |
|---|---|
| `featranker_share_fast` / `_all` | `ConsensusSelector(min_shadow_share=0.5, cv=5, group="fast" or "all")`, scorer `r2` (regression) or `roc_auc` (classification), `n_jobs=-1`; fit on the whole sample, it resamples internally |
| `featranker_q_fast` / `_all` | `ConsensusSelector(max_shadow_q=0.05, cv=RepeatedKFold(5, 2), group="fast" or "all")`, same scorer, `n_jobs=-1` |
| `boruta` | `BorutaPy` on a random forest of unlimited depth fit on all rows (its importances are in-sample impurity importances), `perc=100`, `alpha=0.05`, `max_iter=100`; `n_estimators=100`, which overrides BorutaPy's default of 1000 and is a handicap for Boruta; confirmed features only (tentative are not selected) |
| `permutation_rule` | scikit-learn `permutation_importance` of a random forest on a 70/30 split, importance measured on the 30 %, with the forest's default score (`r2` for regression, accuracy for classification) and `n_repeats=5` (scikit-learn's default, few repeats are a handicap for this rule); keep features with `mean − 2·std > 0` |
| `select_by_shuffling` | feature-engine `SelectByShuffling` on a random forest, default `cv=3` and default threshold (mean performance drift), scorer `r2` or `roc_auc` |

`n_jobs=-1` is used wherever a selector supports it (FeatRanker, and the random forests inside Boruta, the permutation rule, and `SelectByShuffling`; `permutation_importance` as well), so the timings are for a 10-core machine and are not single-core numbers. FeatRanker's "all" group included the optional xgboost and catboost models because both were installed. Dropped configurations: none; every configuration ran for every seed.

Fairness choices worth knowing: Boruta, the permutation rule, and `SelectByShuffling` are run without tuning, as are FeatRanker's settings; FeatRanker's `feature_groups="auto"` was not used (it would group `e` with its copy, see below), and no competitor was tuned beyond its defaults. The scorers differ (FeatRanker and `SelectByShuffling` use `r2` / `roc_auc`; the permutation rule uses the forest's default `r2` / accuracy), as each tool's documented default or example does. `run.py` now makes one untimed warm-up call per selector and task before timing; the preregistered run predates that (see Limits).

## Environment

| Item | Value |
|---|---|
| FeatRanker | 1.4.0, commit `e486fbf`, working tree imported from the repository |
| Python | 3.12.7 |
| numpy / scipy / scikit-learn | 1.26.4 / 1.13.1 / 1.5.1 |
| pandas / joblib | 2.2.2 / 1.4.2 |
| xgboost / catboost | 3.1.3 / 1.2.8 |
| boruta / feature-engine | 0.4.3 / 1.9.4 |
| Machine | macOS (arm64), `os.cpu_count()` = 10 |
| Total run time | 15.4 minutes |

`boruta` and `feature-engine` were installed only in a throwaway virtual environment; they are not dependencies of FeatRanker, not in any extra, and `benchmarks/` is excluded from the sdist and wheel.

## Results

Recall of the 5 signal features, noise features selected (of 4), whether the neutral copy was kept, and wall time, mean ± std over 5 seeds.

**Regression, standard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 6.2 ± 1.0 | 5 |
| featranker_share_all | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 10.3 ± 0.8 | 5 |
| featranker_q_fast | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 11.3 ± 0.2 | 5 |
| featranker_q_all | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 19.3 ± 0.8 | 5 |
| boruta | 1.00 ± 0.00 | 0.20 ± 0.45 | 1.00 ± 0.00 | 4.8 ± 4.6 | 5 |
| permutation_rule | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.4 ± 0.1 | 5 |
| select_by_shuffling | 0.44 ± 0.09 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.7 ± 0.0 | 5 |

**Regression, hard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 5.9 ± 0.2 | 5 |
| featranker_share_all | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 9.9 ± 0.8 | 5 |
| featranker_q_fast | 0.88 ± 0.18 | 0.00 ± 0.00 | 1.00 ± 0.00 | 11.6 ± 0.3 | 5 |
| featranker_q_all | 0.92 ± 0.11 | 0.00 ± 0.00 | 1.00 ± 0.00 | 19.4 ± 1.1 | 5 |
| boruta | 1.00 ± 0.00 | 0.60 ± 0.55 | 1.00 ± 0.00 | 3.0 ± 3.1 | 5 |
| permutation_rule | 1.00 ± 0.00 | 0.20 ± 0.45 | 1.00 ± 0.00 | 0.4 ± 0.2 | 5 |
| select_by_shuffling | 0.44 ± 0.09 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.7 ± 0.0 | 5 |

**Classification, standard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 0.56 ± 0.09 | 0.00 ± 0.00 | 0.60 ± 0.55 | 3.2 ± 0.1 | 5 |
| featranker_share_all | 0.64 ± 0.09 | 0.00 ± 0.00 | 0.80 ± 0.45 | 9.4 ± 0.7 | 5 |
| featranker_q_fast | 0.48 ± 0.11 | 0.00 ± 0.00 | 0.20 ± 0.45 | 6.3 ± 0.2 | 5 |
| featranker_q_all | 0.52 ± 0.11 | 0.00 ± 0.00 | 0.40 ± 0.55 | 17.7 ± 0.5 | 5 |
| boruta | 1.00 ± 0.00 | 0.40 ± 0.55 | 1.00 ± 0.00 | 2.1 ± 1.4 | 5 |
| permutation_rule | 0.64 ± 0.09 | 0.80 ± 1.10 | 0.60 ± 0.55 | 0.3 ± 0.2 | 5 |
| select_by_shuffling | 0.40 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.7 ± 0.0 | 5 |

**Classification, hard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 0.60 ± 0.00 | 0.00 ± 0.00 | 0.80 ± 0.45 | 3.3 ± 0.2 | 5 |
| featranker_share_all | 0.56 ± 0.09 | 0.00 ± 0.00 | 0.80 ± 0.45 | 9.3 ± 0.5 | 5 |
| featranker_q_fast | 0.44 ± 0.17 | 0.00 ± 0.00 | 0.20 ± 0.45 | 6.4 ± 0.2 | 5 |
| featranker_q_all | 0.52 ± 0.11 | 0.00 ± 0.00 | 0.40 ± 0.55 | 18.0 ± 0.5 | 5 |
| boruta | 1.00 ± 0.00 | 0.20 ± 0.45 | 1.00 ± 0.00 | 3.2 ± 2.7 | 5 |
| permutation_rule | 0.64 ± 0.26 | 0.00 ± 0.00 | 0.20 ± 0.45 | 0.3 ± 0.2 | 5 |
| select_by_shuffling | 0.48 ± 0.11 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.7 ± 0.0 | 5 |

### Which roles were found

Share of the 5 seeds in which each role was selected. Regression with standard noise is omitted: every selector except `select_by_shuffling` found every role in every seed there.

**Regression, hard: share of seeds in which each role was selected**

| Selector | a (linear) | b (threshold) | c (interaction) | d (interaction) | e (linear) | f (copy of e) |
|---|---|---|---|---|---|---|
| featranker_share_fast | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| featranker_share_all | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| featranker_q_fast | 1.0 | 1.0 | 0.8 | 0.6 | 1.0 | 1.0 |
| featranker_q_all | 1.0 | 1.0 | 1.0 | 0.6 | 1.0 | 1.0 |
| boruta | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| permutation_rule | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| select_by_shuffling | 1.0 | 1.0 | 0.0 | 0.0 | 0.2 | 0.0 |

**Classification, standard: share of seeds in which each role was selected**

| Selector | a (linear) | b (threshold) | c (interaction) | d (interaction) | e (linear) | f (copy of e) |
|---|---|---|---|---|---|---|
| featranker_share_fast | 1.0 | 1.0 | 0.0 | 0.0 | 0.8 | 0.6 |
| featranker_share_all | 1.0 | 1.0 | 0.2 | 0.0 | 1.0 | 0.8 |
| featranker_q_fast | 1.0 | 1.0 | 0.0 | 0.0 | 0.4 | 0.2 |
| featranker_q_all | 1.0 | 1.0 | 0.0 | 0.0 | 0.6 | 0.4 |
| boruta | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| permutation_rule | 1.0 | 1.0 | 0.2 | 0.4 | 0.6 | 0.6 |
| select_by_shuffling | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

**Classification, hard: share of seeds in which each role was selected**

| Selector | a (linear) | b (threshold) | c (interaction) | d (interaction) | e (linear) | f (copy of e) |
|---|---|---|---|---|---|---|
| featranker_share_fast | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.8 |
| featranker_share_all | 1.0 | 1.0 | 0.0 | 0.0 | 0.8 | 0.8 |
| featranker_q_fast | 1.0 | 0.8 | 0.0 | 0.0 | 0.4 | 0.2 |
| featranker_q_all | 1.0 | 1.0 | 0.0 | 0.0 | 0.6 | 0.4 |
| boruta | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| permutation_rule | 1.0 | 1.0 | 0.4 | 0.4 | 0.4 | 0.2 |
| select_by_shuffling | 1.0 | 1.0 | 0.2 | 0.0 | 0.2 | 0.0 |

## Reading the results

**Where FeatRanker wins.** Zero false selections across all four configurations, both tasks, both noise levels. Boruta selected 7 noise features in total over its 20 runs and the permutation rule 5 (4 of them in standard classification, where its recall is also only 0.64; 1 in hard regression). The absolute gap is small, a fraction of a noise feature per run. In regression FeatRanker with `min_shadow_share=0.5` matches the best recall (1.00 in both variants) with no noise. That is the clearest win, though it is a modest one: Boruta adds 0.2 to 0.6 noise features per run there and the permutation rule 0 to 0.2. FeatRanker also kept the copy `f` together with `e` in regression every time.

**Where FeatRanker loses.**

- *Classification recall.* Boruta found all five signals in all 10 classification runs (with 0.4 and 0.2 noise features on average); FeatRanker found 0.44 to 0.64. The role tables show where: `c` and `d`, which matter only together, were selected in at most 1 of 5 seeds by any FeatRanker configuration, while Boruta always found them. The signal is not weak (see the post-hoc section). The cause is the shadow rule: `above_shadow_share` is averaged over model families, and in classification `group="fast"` 4 of the 7 families (logistic regression, the linear margin classifier, linear discriminant analysis, naive Bayes) are additive and almost never rank a pure interaction above the shadows, so `c` and `d` average at most about 3/7 and stay under the 0.5 cut-off. The post-hoc diagnostic measured this. Regression `"fast"` includes an RBF-kernel family that does score `c` and `d`, which is why regression passes. This is a limitation of FeatRanker: a family-majority rule is blind to effects that only some model families can model.
- *`max_shadow_q`.* The BH-adjusted rule was at or below the simpler `min_shadow_share=0.5` rule on recall in every table (for instance 0.52 against 0.64 in standard classification with `group="all"`, 0.92 against 1.00 in hard regression), with the same zero noise, at roughly twice the time. At 5 seeds on 1000 rows it bought nothing here.
- *`group="all"` versus `"fast"`.* No consistent gain from the 18 or 22 models over the 5 or 7 of `"fast"`; recall differs by at most 0.08 in either direction in the share rule, at about 1.7 to 3 times the time per run for the same rule. `"fast"` was sufficient on this data.
- *Runtime.* Expected: FeatRanker fits many models over several resamples. Mean time per configuration was 3.2 to 19.4 s against 0.3 to 4.8 s for the others; single runs ranged over 3.1 to 21.2 s (FeatRanker) and 0.22 to 9.93 s (others), so Boruta's slowest runs overlap FeatRanker's fastest, and classification `featranker_share_fast` (3.2 to 3.3 s) is close to Boruta's means (2.1 and 3.2 s). The permutation rule and `SelectByShuffling` fit one forest, and Boruta stops early once it has decided.
- *The correlated copy.* In classification, FeatRanker kept `e` in 0.4 to 1.0 of seeds and the copy `f` in 0.2 to 0.8, while the lone linear feature `a` was found in every seed. A likely contributor, not established: FeatRanker shuffles the shadow block with one shared row order (`_with_shadows`), so the near-collinear `e`/`f` pair stays collinear among the shadows. In a single-seed inspection during review, linear discriminant analysis scored the shadows of `e` and `f` at about 0.056 and 0.055 AUC drop against at most 0.002 for the other shadows, in 3 of the 5 splits of one seed (splits 0 to 2: 0.056/0.055, 0.061/0.035, 0.034/0.034; splits 3 and 4 at most 0.008), which would raise the bar (`shadow_max`) that linear models set for those two features. This was seen on one seed only and has not been tested across seeds. Permutation importance splitting credit between near-duplicates is another possible contributor and was also not isolated. `feature_groups="auto"` is the documented remedy for the second and was not used here, since the benchmark takes the defaults; its effect is not measured. Boruta kept both `e` and `f` every time, which is correct under its all-relevant definition.

**The other selectors.** `SelectByShuffling` with its default threshold keeps only features whose performance drift is at least the mean drift, which in these runs dropped the interaction pair and, in most seeds, `e` (recall 0.40 to 0.48, no noise). That is a property of the default threshold, not necessarily of the method; a lower threshold would probably raise its recall and its noise, which was not tested.

## Limits

- **Synthetic data.** The roles favour whichever model family fits them; the set (linear, threshold, interaction, correlated copy) was chosen to spread this around, but a different generator could reorder the results. Real data has no ground truth, which is the reason for using synthetic data and also its limit.
- **One size.** 1000 rows, 10 features, 5 seeds. With 5 seeds, differences of a few hundredths are noise; read the standard deviations. Larger feature counts and sample sizes may change both accuracy and runtime.
- **Defaults, not tuning.** Nobody was tuned. A user who tunes Boruta, thresholds, or FeatRanker's `min_shadow_share` will get different numbers.
- **Timings** depend on the machine and on `n_jobs=-1`; compare them with each other, not with other hardware. Boruta's time varies widely because it stops early. The preregistered run had no warm-up, and its first FeatRanker run (`featranker_share_fast`, regression, standard, seed 0: 7.9 s against 5.6 to 5.9 s for the other four seeds) probably includes one-off import and worker start-up, which inflates that row's mean and standard deviation (6.2 ± 1.0 s). The run was not repeated; the post-hoc run, with a warm-up, timed the same configuration at 5.6 to 6.0 s on regression standard.
- **Classification labels** are sampled, so even the true model cannot separate perfectly; during review the oracle AUC was 0.91 to 0.93, and permuting `c·d` dropped it by 0.09 to 0.13, comparable to `b` and `e` (a reviewer measurement, not in the result files). The missed interaction is therefore not a weak-signal artefact.

## Added after seeing the results (2026-10-20)

**Why.** The preregistered table showed FeatRanker missing the `c·d` interaction in classification. Review traced it to the family-averaged shadow share; this section tests that explanation. It was written after the preregistered results were known, so it is a diagnostic, not part of the headline comparison, and nothing in the preregistered file or tables above was changed (design D6).

**What was run.** `featranker_share_fast` again, now recording per-role and per-family `above_shadow_share` (for each family, the fraction of its voting members above their shadows, per split, averaged over the splits), and one extra configuration, `featranker_share_tree`: the same rule (`min_shadow_share=0.5`, `cv=5`, same scorers) with `group="tree"`. Same seeds, tasks and variants, with an untimed warm-up first. File: `benchmarks/results/20261020-posthoc.json` (4.2 minutes). The re-run of `featranker_share_fast` selected exactly the same features as the preregistered run in all 20 runs.

**Result: the tree group recovers the interaction.**

**Regression, standard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 5.7 ± 0.1 | 5 |
| featranker_share_tree | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 8.5 ± 0.7 | 5 |

**Regression, hard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 5.9 ± 0.3 | 5 |
| featranker_share_tree | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 8.4 ± 0.9 | 5 |

**Classification, standard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 0.56 ± 0.09 | 0.00 ± 0.00 | 0.60 ± 0.55 | 3.3 ± 0.2 | 5 |
| featranker_share_tree | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 8.0 ± 0.1 | 5 |

**Classification, hard**

| Selector | Recall | Noise selected | Copy kept | Time (s) | Seeds |
|---|---|---|---|---|---|
| featranker_share_fast | 0.60 ± 0.00 | 0.00 ± 0.00 | 0.80 ± 0.45 | 3.2 ± 0.0 | 5 |
| featranker_share_tree | 0.92 ± 0.11 | 0.00 ± 0.00 | 1.00 ± 0.00 | 8.0 ± 0.6 | 5 |

**Classification, standard: share of seeds in which each role was selected**

| Selector | a (linear) | b (threshold) | c (interaction) | d (interaction) | e (linear) | f (copy of e) |
|---|---|---|---|---|---|---|
| featranker_share_fast | 1.0 | 1.0 | 0.0 | 0.0 | 0.8 | 0.6 |
| featranker_share_tree | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**Classification, hard: share of seeds in which each role was selected**

| Selector | a (linear) | b (threshold) | c (interaction) | d (interaction) | e (linear) | f (copy of e) |
|---|---|---|---|---|---|---|
| featranker_share_fast | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.8 |
| featranker_share_tree | 1.0 | 1.0 | 1.0 | 0.8 | 0.8 | 1.0 |

In classification, `group="tree"` raised recall from 0.56 to 1.00 (standard) and from 0.60 to 0.92 (hard) with no noise selected, at about 2.5 times the time (8.0 s against 3.3 s). Regression stayed at 1.00. The copy `f` was also kept in more seeds (1.00 against 0.60 and 0.80), which is consistent with the linear families being a part of the `e`/`f` problem but does not prove it.

**Per-family shares for `c` and `d`.** Mean over the 5 seeds of each family's above-shadow share (a share above 0.5 means the family usually ranks the feature above its shadow). The last row is the number the 0.5 rule compares against.

**Classification, standard, featranker_share_fast: above-shadow share by family, mean over seeds**

| Family | c | d |
|---|---|---|
| bagging | 0.96 | 0.72 |
| boosting | 1.00 | 1.00 |
| discriminant | 0.04 | 0.00 |
| linear_margin | 0.08 | 0.00 |
| logistic | 0.08 | 0.00 |
| naive_bayes | 0.08 | 0.12 |
| single_tree | 0.64 | 0.40 |
| **share used by the rule (average over families)** | 0.41 | 0.32 |

**Classification, standard, featranker_share_tree: above-shadow share by family, mean over seeds**

| Family | c | d |
|---|---|---|
| bagging | 0.88 | 0.75 |
| boosting | 0.79 | 0.80 |
| single_tree | 0.64 | 0.40 |
| **share used by the rule (average over families)** | 0.77 | 0.65 |

**Classification, hard, featranker_share_fast: above-shadow share by family, mean over seeds**

| Family | c | d |
|---|---|---|
| bagging | 0.80 | 0.84 |
| boosting | 1.00 | 1.00 |
| discriminant | 0.04 | 0.00 |
| linear_margin | 0.04 | 0.00 |
| logistic | 0.12 | 0.04 |
| naive_bayes | 0.08 | 0.20 |
| single_tree | 0.12 | 0.20 |
| **share used by the rule (average over families)** | 0.31 | 0.33 |

**Regression, standard, featranker_share_fast: above-shadow share by family, mean over seeds**

| Family | c | d |
|---|---|---|
| bagging | 0.96 | 0.92 |
| boosting | 1.00 | 1.00 |
| kernel | 1.00 | 1.00 |
| linear | 0.12 | 0.00 |
| single_tree | 0.32 | 0.44 |
| **share used by the rule (average over families)** | 0.68 | 0.67 |

**Reading.** In classification `"fast"`, boosting (1.00) and bagging (0.72 to 0.96) see `c` and `d` and the single tree partly does (0.40 to 0.64 in standard noise), but the four additive families, logistic, linear margin, discriminant and naive Bayes, sit at 0.00 to 0.20. Averaged over 7 families the share is 0.41 and 0.32 (standard) and 0.31 and 0.33 (hard), below 0.5. In regression `"fast"` the RBF kernel family scores both at 1.00, so with boosting and bagging three of five families are strong and the shares are 0.68 and 0.67, above 0.5. Tree families only (`group="tree"`) have no additive voters and the shares rise to 0.65 to 0.77 (standard) and 0.55 to 0.57 (hard). The `"all"` group was not instrumented; that quadratic discriminant analysis lifts one family there (which would cap the share near 0.5) is a reviewer's reading, not measured here.

**What this means.** The miss is a property of how FeatRanker combines families: the shadow share is an average over families, so an effect that only some families can model, such as a pure interaction, needs a majority of the families to represent it. Restricting to tree families removes that blind spot on this data, but it was chosen after seeing the results, drops whatever the additive families would contribute, and was run on the same five seeds, so treat it as a diagnostic and not as a tuned recommendation.

## Reproduce

```bash
cd Featranker
python -m venv /tmp/bench && /tmp/bench/bin/pip install -e . boruta feature-engine pytest
PYTHONPATH=. /tmp/bench/bin/python benchmarks/run.py --seeds 0 --n-rows 300   # quick smoke run
PYTHONPATH=. /tmp/bench/bin/python benchmarks/run.py                           # full run, about 15 minutes
PYTHONPATH=. /tmp/bench/bin/python -m pytest benchmarks                         # tests (real-wrapper checks skip if boruta/feature-engine are missing)
/tmp/bench/bin/python benchmarks/run.py --render benchmarks/results/20261019.json   # print the tables of a saved run
# post-hoc diagnostic (see the last section):
PYTHONPATH=. /tmp/bench/bin/python benchmarks/run.py --selectors featranker_share_fast,featranker_share_tree --out benchmarks/results/20261020-posthoc.json
```

If `boruta` or `feature-engine` is not installed, that selector is recorded as missing and the run continues; a selector that raises is recorded as an error.
