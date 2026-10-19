# Spike: is an LLM investigator worth building?

Go/no-go spike for the FeatRanker 1.1/1.2 roadmap. Design, benchmark, and decision rule: [`openspec/changes/spike-investigator-value/design.md`](../../openspec/changes/spike-investigator-value/design.md). This directory is not part of the `featranker` package and is not run by CI.

## Setup (maintainer)

Python 3.10 or newer (3.12 tested). In your own terminal, never through the Claude session:

```bash
source ~/venvs/featranker-llm/bin/activate          # or any venv with Python >= 3.10
pip install -e ~/Desktop/CodeHouse/Featranker shap openai pytest
cd ~/Desktop/CodeHouse/Featranker
python -m pytest spikes/investigator -q              # 29 tests, fake LLM, no network
read -s "DEEPSEEK_API_KEY?DeepSeek key: " && export DEEPSEEK_API_KEY && echo
python spikes/investigator/run.py --model <deepseek-model-id>
```

Each run makes up to 30 LLM calls (5 seeds × 6 calls at most) and writes `spikes/investigator/results/<model>-<timestamp>.json`. The key is read from the environment and never written. For another OpenAI-compatible provider: `--base-url <url> --key-env <VAR>`.

## Frozen setup

Fixed before any real LLM run. A result file counts only if its `spike_sha256` equals this hash; any later change to `spike.py` is listed under Deviations and all arms are re-run.

| Item | Value |
|---|---|
| `spike.py` SHA-256 | `822632a72b3885361ccd70083e2d526509904cddc59c916fc2a6d487b53f5ef2` |
| Seeds | 0, 1, 2, 3, 4 |
| Arms | narrator, evidence, investigator (≤ 3 follow-ups) |
| Temperature | 0 |
| Decision rule | D6: recall gap ≥ 0.2, investigator wins ≥ 4 of 5 seeds, mean false findings ≤ 0.5, confirmed-false hypotheses ≤ 1 in total; borderline = within 0.05 of a threshold |
| Primary model | the maintainer's DeepSeek model (id recorded in the result file) |
| Fallback model | to be chosen if the outcome is NO-GO or borderline |

Dry run (scripted fake LLM, all 5 seeds): the pipeline completes, results serialize, and the decision rule runs. Planted overfit check: the decision tree's train-minus-eval R² gap is 0.66–0.92, ridge's is −0.08 to 0.04.

## Results

### Run 1: `deepseek-flash` (`results/deepseek-flash-20261019-003924.json`)

`spike_sha256` matches the frozen hash. `spike_dirty: true` only because `spikes/` is not yet committed; `spike.py` is unchanged.

Recall (out of 5 planted mechanisms) and false findings per seed:

| Seed | Narrator | Evidence | Investigator |
|---|---|---|---|
| 0 | 0.2 / 1 | 0.8 / 1 | **1.0 / 0** |
| 1 | 0.6 / 3 | 0.8 / 1 | 0.8 / 1 |
| 2 | 0.4 / 1 | 0.6 / 1 | **1.0 / 0** |
| 3 | 0.2 / 1 | 0.8 / 1 | 0.8 / 1 |
| 4 | 0.2 / 1 | 0.6 / 1 | 0.8 / 1 |
| **Mean** | **0.32 / 1.4** | **0.72 / 1.0** | **0.88 / 0.6** |

Gaps: evidence − narrator **+0.40**, investigator − narrator **+0.56**, investigator − evidence **+0.16** (investigator ahead in 3 of 5 seeds, tied in 2). Confirmed-false hypotheses: **0**. Every run returned a valid final reply; the investigator used all 3 follow-ups every time (4 calls).

D6 outcome: **no_go, borderline** (investigator − evidence is 0.16, within 0.05 of 0.2). Two gates failed:
- loop: gap 0.16 < 0.2, 3 of 5 wins < 4, mean false findings 0.6 > 0.5;
- evidence only: mean false findings 1.0 > 0.5, although the recall gap (+0.40) passed.

What the false findings were: 13 of the 15 false findings across all arms are one mistake, **the threshold feature labelled `linear`** (in seed 4 the narrator listed it in one `linear` finding together with two true signal features). The narrator's other two were a noise feature called linear and a wrong correlated pair (seed 1). Evidence and investigator made no other kind of false finding.

What the investigator did: every trail is 3 preregistered `shap_dependence` / `shap_interaction` tests, all with code verdicts. In seeds 0 and 2 it tested the threshold feature for linearity (`linearity_r2 > 0.8`), got **REFUTED** (0.70, 0.69), and corrected its final finding to `nonlinear`, which gave full recall and no false findings. In seeds 1, 3, and 4 it spent the 3 experiments confirming the interaction pair and never tested the threshold feature, so it kept the evidence arm's mistake. Its predictions were lenient (for example `share > 0.02` where the true interaction share is about 0.09) but never confirmed a false claim.

Per D6, a borderline NO-GO requires one stronger model before deciding.

### Run 2: `deepseek-v4-pro` (`results/deepseek-v4-pro-20261019-012505.json`)

`spike_sha256` matches the frozen hash; `spike_dirty: true` for the same reason as run 1.

| Seed | Narrator | Evidence | Investigator |
|---|---|---|---|
| 0 | 0.0 / 1 | 0.6 / 1 | **1.0 / 0** |
| 1 | 0.6 / 2 | 0.4 / 1 | **0.8 / 1** |
| 2 | 0.0 / 1 | 0.8 / 0 | 0.8 / 0 |
| 3 | 0.4 / 1 | 0.8 / 0 | 0.8 / 0 |
| 4 | 0.8 / 0 | 0.4 / 1 | **1.0 / 0** |
| **Mean** | **0.36 / 1.0** | **0.60 / 0.6** | **0.88 / 0.2** |

Gaps: evidence − narrator **+0.24**, investigator − narrator **+0.52**, investigator − evidence **+0.28** (investigator ahead in 3 of 5 seeds, tied in 2, never behind). Confirmed-false hypotheses: **0**. Every run returned a valid final reply; the investigator used all 3 follow-ups every time.

D6 outcome: **no_go, borderline** (evidence − narrator is 0.24, within 0.05 of 0.2). Two gates failed:
- loop: only the seed-consistency condition, 3 of 5 wins < 4 (gap, false findings, and confirmed-false all passed);
- evidence only: mean false findings 0.6 > 0.5, although the recall gap (+0.24) passed.

False findings: 6 of the 9 are again the threshold feature labelled `linear` (narrator 2, evidence 3, investigator 1); the other 3 are wrong correlated pairs from the narrator.

What the investigator did: it tested the threshold feature in seeds 0, 2, and 4 and labelled it `nonlinear` each time (seed 0 after a REFUTED linearity prediction, 0.70). It did not test it in seeds 1 and 3: in seed 1 it kept the `linear` mistake; in seed 3 it dropped the correct `nonlinear` finding the evidence arm had. 2 of its 15 predictions were refuted; one of them refuted a true claim (seed 2: `linearity_r2 >= 0.95` for the linear feature on `hist_gb`, observed 0.943, because tree SHAP is a step function).

### Both runs

| | Narrator | Evidence | Investigator |
|---|---|---|---|
| Mean recall, run 1 / run 2 | 0.32 / 0.36 | 0.72 / 0.60 | 0.88 / 0.88 |
| Mean false findings, run 1 / run 2 | 1.4 / 1.0 | 1.0 / 0.6 | 0.6 / 0.2 |

Across 10 model–seed comparisons using the same five dataset seeds, the investigator was ahead of evidence in 6, tied in 4, and behind in none; none of its 30 preregistered predictions confirmed a false claim. These observations are paired across models; D6 is applied per model.

### Scoring sensitivity (exploratory, not part of D6)

D2 scores a finding by its exact feature set, so a grouped finding such as `linear: [f1, f2]` matches neither `linear:f1` nor anything else. The final-reply format shows one feature per finding but does not require it. The stronger model put several features into one `linear` finding in 6 of its 15 final replies (narrator 3, evidence 2, investigator 1). Rescoring with grouped `linear`/`nonlinear`/`noise` findings split into one finding per feature:
- run 1: decision unchanged (no_go, borderline; narrator recall 0.32 → 0.36);
- run 2: **go_1.1_1.2, borderline** (recall 0.48 / 0.68 / 0.92; the investigator's grouped finding in seed 2 becomes a win, 4 of 5).

The rescore was chosen after seeing the data, so it cannot replace the preregistered outcome. It shows that the run 2 outcome depends on one reply in one seed.

## v2: confirmatory run (design D8)

v1 above stays NO-GO. v2 tests whether grouped findings drove that result. One run; its D6 outcome is final, borderline or not.

| Item | Value |
|---|---|
| `spike.py` SHA-256 | `b90a37cd4c12e7a1a6485c5056475fd3129ccf9903caf811d48e47c1d0cc54e4` (v1 code: commit `8e7c2bf`) |
| Change from v1 | finding arity: one feature for `linear`/`nonlinear`/`noise`, two for `interaction`/`correlated`, none for `overfit`; stated in the final format; any other count is a false finding |
| Seeds | 5, 6, 7, 8, 9 (`run.py` default) |
| Model | `deepseek-flash`, temperature 0 |
| Decision rule | D6 unchanged; no follow-up run |

Run: `python spikes/investigator/run.py --model deepseek-flash`

Dry check on seeds 5–9: the decision tree's train-minus-eval R² gap is 0.57–0.88, ridge's −0.08 to 0.08, and the planted correlated pair appears in every seed.

### v2 result: `deepseek-flash` (`results/deepseek-flash-20261019-031917.json`)

`spike_sha256` matches the v2 hash; `commit` is `8e7c2bf` (v1); `spike_dirty: true` because the v2 changes were not committed before the run.

| Seed | Narrator | Evidence | Investigator |
|---|---|---|---|
| 5 | 0.6 / 0 | **1.0 / 0** | 0.6 / 1 |
| 6 | 0.4 / 1 | **1.0 / 0** | 0.6 / 1 |
| 7 | 0.4 / 2 | 1.0 / 0 | 1.0 / 0 |
| 8 | 0.4 / 2 | 1.0 / 1 | 1.0 / 0 |
| 9 | 0.6 / 0 | 0.8 / 2 | **1.0 / 0** |
| **Mean** | **0.48 / 1.0** | **0.96 / 0.6** | **0.84 / 0.4** |

Gaps: evidence − narrator **+0.48**, investigator − narrator **+0.36**, investigator − evidence **−0.12** (investigator ahead in 1 seed, tied in 2, behind in 2). Confirmed-false hypotheses: **0**. Every reply was valid and no finding broke the arity rule.

D6 outcome: **no_go, not borderline**. The loop fails on recall gap and seed consistency; evidence only fails on mean false findings (0.6 > 0.5) although its recall gap (+0.48) passed.

False findings: 6 of the 10 are again the threshold feature labelled `linear` (narrator 3, evidence 1, investigator 2); 3 are a noise feature labelled `linear` (narrator 1, evidence 2); 1 is the narrator calling the linear source of the correlated pair `nonlinear`.

What the investigator did: in seed 5 it tested the threshold feature with an interaction member; in seed 6 it tested the threshold feature with the linear feature. Both predictions were REFUTED. Both final replies omitted the true interaction and labelled the threshold feature `linear`, whereas the evidence arm identified both mechanisms correctly. In seed 7 a too-strict prediction (`linearity_r2 < 0.5`, observed 0.68) refuted a true `nonlinear` claim; the final reply kept the correct finding anyway.

Across runs: the evidence arm on the same model scored 0.72 on seeds 0–4 and 0.96 on seeds 5–9, but both the seeds and arity instructions changed, so this comparison cannot isolate seed sensitivity. Each run contains only five seeds.


## Outcome

**Final: NO-GO LLM** (D6 via D8: v1 NO-GO on both models, v2 NO-GO not borderline). Per D6: 1.1 `evidence-for-why` ships without LLM use, `explain()`/`to_prompt()` are deprecated in 1.1, and the 1.2 investigator is not built.

On this benchmark the evidence battery is what carried the gains: the arm that read it beat the report-only narrator proxy by +0.40, +0.24, and +0.48 in mean recall across the three runs. The evidence arm never met the false-findings limit (1.0, 0.6, 0.6), and the investigator's preregistered experiments did not add consistent value over reading the evidence. In v2 seeds 5 and 6, the investigator's final replies missed mechanisms that the evidence arm identified correctly; the recorded trails do not establish that it first held and then abandoned those findings. These results do not measure production `explain()` directly, and the NO-GO is D6's stipulated policy consequence, not a demonstration that LLM interpretation has no value.

## Outcome (v1)

Under the preregistered D6 rule, **NO-GO on both models** (both borderline). D6's "Next" for NO-GO: 1.1 `evidence-for-why` ships without LLM use, and `explain()`/`to_prompt()` are deprecated in 1.1.

On this benchmark, the report-only narrator proxy had the lowest mean recall and highest mean false-findings count, while the investigator had the opposite ordering on both models. Flash failed the loop's recall-gap, seed-consistency, and false-findings conditions; Pro failed seed consistency only. Trails suggest missed threshold checks contributed to some errors, but do not establish why the gates failed or measure production `explain()` directly. The NO-GO is D6's stipulated policy consequence, not a demonstration that LLM interpretation lacks value. The roadmap decision is recorded in `openspec/config.yaml` (task 7.1) once the maintainer has made it.

## Deviations

- Freeze by SHA-256 of `spike.py`, not by commit as design D6 says: `spikes/` was not committed before the runs. Both result files carry the frozen hash; `spike_dirty: true` only reflects the uncommitted directory.
- The scoring-sensitivity rescore above is post hoc and is reported only for transparency.
