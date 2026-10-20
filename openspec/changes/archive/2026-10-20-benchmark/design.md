## Context

`spikes/investigator/` set the pattern for repo-local, unshipped experiment code with frozen settings and result files. FeatRanker 1.4 provides `ConsensusSelector(min_shadow_share=..., max_shadow_q=..., group=...)`. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:** reproducible, fair-by-default comparison; results and environment recorded; honest write-up.

**Non-Goals:** a leaderboard or tuned results.

## Decisions

### D1. Data generator
Own small generator in `benchmarks/run.py` (not imported from `spikes/`): `n_rows=1000`, 10 base features from standard normals, roles permuted per seed. Regression: `y = 2a + 3·[b > 0] + 2·c·d + 1.5·e + noise`, `f = e + 0.1·noise` (copy), the rest noise. Classification: the same latent score through a logistic link, sampled labels. Seeds 0–4 (10 seeds × the full registry under repeated CV would take hours on a laptop; 5 keeps the whole run near 30 minutes). A "hard" variant doubles the noise standard deviation. If a configuration still exceeds the budget, it is dropped from the table and the reason recorded, never silently.

### D2. Truth and scoring
Signal = {a, b, c, d, e}; neutral = {f}; noise = the rest. Per run: `recall = |selected ∩ signal| / 5`, `noise_selected = |selected ∩ noise|`, wall time of fit/selection only. The table reports mean and standard deviation over seeds.

### D3. Selectors and fairness
Every selector sees the same rows: FeatRanker selectors fit on the whole sample (they resample internally, `cv=RepeatedKFold(5, 2)` for the q rule and default `cv=5` otherwise); Boruta, permutation importance (on a 70/30 split, importance on the 30 %), and SelectByShuffling (its own CV) use their documented defaults with `random_state` fixed. Random forests use 100 trees everywhere. Any selector that fails to import is skipped and recorded as missing, so the script runs with only FeatRanker installed.

### D4. Environment
Run with a throwaway virtual environment in the session scratchpad: `pip install -e . boruta feature-engine`. Package versions are written into the result file and the docs table.

### D5. Packaging
`benchmarks/` is excluded from the sdist and wheel; a check in the tasks builds both and confirms no `benchmarks/` files (as done for `spikes/`).

### D6. Post-hoc diagnostic (amendment, 2026-10-20, after seeing the results)
The preregistered table showed FeatRanker missing the c·d interaction in classification. Review traced it to the family-averaged shadow share: 4 of the 7 classification "fast" families are additive and cannot rank an interaction above the shadows. To test that explanation, without changing the preregistered results:
- One extra configuration, `featranker_share_tree` (`group="tree"`, otherwise identical), run on the same seeds, tasks and variants, and saved to a separate file `benchmarks/results/<date>-posthoc.json`.
- Instrumentation for the FeatRanker share rows in that file: per-role and per-family `above_shadow_share` from `report_`.
- Reported in `docs/benchmark.md` under "Added after seeing the results", with this date and reason, outside the headline comparison. No change to `min_shadow_share`, scorers, or the preregistered rows.

Code changes made with the diagnostic: an untimed warm-up call per selector and task before timing (future runs; the preregistered timings predate it), `roles` stored in each run row, a per-family table renderer, and a `PREREGISTERED` tuple holding the original seven selectors as the default. Also written up: the D1 logit is `latent - 1.5 + sd·ε` (the docs describe the centring and noise term).

## Risks / Trade-offs

- [Synthetic data favours methods that match its structure] → stated in the write-up; roles cover linear, threshold, interaction, and correlation so no single model family has an easy win.
- [Boruta may not run on current NumPy] → if it fails, record the error in the results and the docs rather than patching it.
- [Runtime depends on the machine] → report relative times and the machine description.
