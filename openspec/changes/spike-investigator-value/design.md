## Context

See proposal.md for why. FeatRanker 1.0 has `to_prompt()`/`explain()` (provider-neutral `llm(prompt) -> str`, JSON replies, path verification). `pyproject.toml` ships only `packages = ["featranker"]` and CI runs only `tests/`, so a top-level `spikes/` directory is neither packaged nor run by CI. Real LLM calls need the maintainer's API key, which never enters this session: the maintainer runs the benchmark and the results land as files in the repo.

## Goals / Non-Goals

**Goals:**
- Measure, with code-only scoring against known ground truth, whether an LLM finds more true mechanisms with (a) more evidence and (b) the ability to run preregistered follow-up experiments.
- Fix the decision rule before any real LLM run, so the spike cannot be tuned toward a wanted answer.
- Leave a replayable trail for every investigator run.

**Non-Goals:**
- Reusable or public APIs; spike code may call FeatRanker private helpers.
- Prompt optimisation beyond one honest draft per arm.

## Decisions

### D1. Layout: three files, outside the package

```
spikes/investigator/
  spike.py        benchmark, evidence battery, actions, arms, loop, scoring (no network)
  run.py          CLI: real LLM via an OpenAI-compatible endpoint; writes results/*.json
  test_spike.py   scripted fake LLM; run with: python -m pytest spikes/investigator
  results/        result files written by run.py
  FINDINGS.md     measured results and the outcome
```

One logic module rather than one per concern: the spike is frozen after use. `run.py` is the only file that touches the network; it reads the key from `DEEPSEEK_API_KEY` (base URL and model id are flags).

### D2. Benchmark: synthetic regression with planted mechanisms

Per seed: 1000 rows, 10 standard-normal features named `f0`–`f9`, a 700/300 train/eval split, `y = signal + N(0, 1)`. Feature roles are randomly permuted per seed, so position never gives the answer and names carry no domain meaning.

| Mechanism (ground truth) | Planted as | Scored finding |
|---|---|---|
| linear | `+2·a` | `{features: [a], mechanism: "linear"}` |
| nonlinear (threshold) | `+3·[b > 0]` | `{features: [b], mechanism: "nonlinear"}` |
| interaction | `+2·c·d` (no main effects) | `{features: [c, d], mechanism: "interaction"}` (order-free) |
| correlated duplicate | `+1.5·e`, `f = e + N(0, 0.1)` with no own effect | `{features: [e, f], mechanism: "correlated"}` |
| overfit model | unconstrained `decision_tree` | `{model: "decision_tree", mechanism: "overfit"}` |
| noise | four features with no effect | claiming any mechanism on them is a false finding |

Recall = matched mechanisms / 5. A **false finding** is any structured finding that contradicts ground truth (wrong mechanism on a signal feature, any mechanism except `noise` on a noise feature, `overfit` on a model whose train-minus-eval R² gap is below 0.1). Findings naming the right mechanism are matched once; duplicates are ignored.

Alternative considered: real datasets (diabetes, breast cancer). Rejected for the spike: no ground truth, so scoring would need judges, which overscore (Agentic XAI study).

### D3. Models: four, passed as custom `models=`

`ridge`, `random_forest` (100 trees), `hist_gb`, `decision_tree` (unconstrained), all seeded. Enough to give a linear-vs-tree contrast and one overfit candidate while keeping SHAP fast.

### D4. Three arms, one output format

All arms share the same rules and the same final reply: `{"findings": [{"features": [...], "model": optional, "mechanism": one of linear|nonlinear|interaction|correlated|overfit|noise, "reason": "..."}]}`. Same format means the narrator is not penalised for a different output shape.

| Arm | LLM sees | LLM calls |
|---|---|---|
| narrator | the FeatRanker report data view (what pasting into a chat gives) | 1 |
| evidence | narrator input + evidence battery | 1 |
| investigator | evidence input + action menu; may run up to 3 follow-ups | ≤ 4 |

**Evidence battery** (planned 1.1 contents, cheap global summaries): per model train and eval R²; per model and feature mean |SHAP| and direction (Spearman correlation between feature value and SHAP value); feature pairs with |Pearson r| ≥ 0.5 on eval data. It deliberately holds no per-feature drill-downs, so the follow-ups have something to add.

**Action menu** (one action per turn):

| Action | Returns |
|---|---|
| `shap_dependence(model, feature)` | mean SHAP per feature decile; R² of a straight-line fit of SHAP on the feature (`linearity_r2`) |
| `shap_interaction(model, f1, f2)` (tree models) | mean \|interaction value\| and its share of total attribution, on 200 eval rows |
| `refit(model, params)` with params limited to `max_depth`, `min_samples_leaf`, `alpha` | train and eval R² |
| `merge_rank(features)` | FeatRanker re-rank with those features as one group: merged rank, member ranks before merge |

### D5. Preregistration and code verdicts

Each investigator turn is either a final findings reply or `{"hypothesis": str, "claim": <one finding, D4 format>, "prediction": {"path": str, "op": ">"|">="|"<"|"<=", "value": number}, "action": {...}}`. The structured `claim` is what lets code score a hypothesis against ground truth; the free-text `hypothesis` is for the trail only. Code runs the action, resolves `path` in the result with the 1.0 path grammar, and records `confirmed`, `refuted`, or `invalid` (missing path, bad action, or a non-numeric value). The LLM sees each verdict before its next turn. After 3 follow-ups, or on an invalid reply, the next reply must be final.

Trail entry: `{turn, hypothesis, prediction, action, result, verdict}`. Actions are deterministic given the seed, so `replay(trail)` recomputes every result and verdict without an LLM; a test checks they match exactly.

Also recorded: **confirmed-false hypotheses** (verdict `confirmed` on a step whose `claim` is a false finding under D2). They count against the investigator in D6. A step with a missing or malformed `claim` is invalid.

### D6. Preregistered decision rule

5 seeds × 3 arms, temperature 0, model id recorded. With R̄ = mean recall over seeds, F̄ = mean false findings per run:

| Outcome | Condition (all must hold) | Next |
|---|---|---|
| **GO: 1.1 + 1.2** | R̄_inv − R̄_evi ≥ 0.2; investigator beats evidence in ≥ 4 of 5 seeds; R̄_inv − R̄_nar ≥ 0.2; F̄_inv ≤ 0.5; confirmed-false hypotheses ≤ 1 in total | `evidence-for-why`, then `llm-investigator` |
| **GO: 1.1 only** | not the row above; R̄_evi − R̄_nar ≥ 0.2; F̄_evi ≤ 0.5 | `evidence-for-why` with `explain()` reading the evidence; no loop |
| **NO-GO LLM** | neither row above | 1.1 evidence ships without LLM use; `explain()`/`to_prompt()` deprecated in 1.1 |

One primary model (the maintainer's DeepSeek model). If the result is NO-GO or borderline (within 0.05 of a threshold), one stronger model is also run before deciding, since weaker models refine hypotheses poorly (InterpAgent). Both are reported.

Thresholds are set before any real run. Prompts and code are frozen at a recorded commit before the first real run; any later change is reported in FINDINGS.md, and all arms are re-run.

### D7. Dependencies

Spike-only: `shap` (TreeExplainer for tree models, LinearExplainer for `ridge`) and `openai` (OpenAI-compatible client for DeepSeek). Neither goes into `pyproject.toml`. Justification: the spike must measure the exact explainers 1.1 would use.

### D8. Confirmatory v2 (after v1 NO-GO)

v1 (seeds 0–4, both models) stays recorded as NO-GO under D6. v2 is one new run that tests whether the D2 scoring ambiguity drove the result: a grouped finding such as `linear: [f1, f2]` earned no credit, and the final format did not forbid it.

Frozen before the run, by a new SHA-256 of `spike.py`:
- Finding arity: `linear`, `nonlinear`, `noise` name exactly one feature; `interaction`, `correlated` exactly two; `overfit` none. The final format states this. A finding (or claim) with any other count is a false finding.
- Seeds 5–9 (`run.py` default). Model: the maintainer's `deepseek-flash`. Temperature 0.
- Everything else unchanged: benchmark, models, actions, prompts apart from the arity rule, scoring apart from arity, D6 thresholds.
- Stopping rule: one run, all three arms; the D6 outcome is final. A borderline result triggers no further model or run. No tuning or repeat attempts after v2.

## Risks / Trade-offs

- [Small benchmark: 5 mechanisms × 5 seeds is noisy] → the rule needs consistency across seeds as well as mean gaps; borderline results trigger a second model.
- [Synthetic data is cleaner than real data] → the spike decides only whether to build; 1.2 must still be validated on a real dataset.
- [Self-tuning toward the benchmark (our own p-hacking)] → rule and prompts frozen before real runs; changes disclosed and all arms re-run.
- [The investigator chooses lax prediction thresholds to get "confirmed"] → confirmed-false hypotheses are counted and gated.
- [shap or numba fails to install on new Python] → run in a Python 3.12 environment.
- [Malformed LLM replies] → parsed strictly; an invalid final reply scores zero findings for that run and is reported, not retried.
