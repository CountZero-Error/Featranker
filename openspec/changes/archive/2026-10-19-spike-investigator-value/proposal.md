## Why

The 1.0 LLM feature narrates a finished report, which a user can already do by pasting the report into any chat. The planned replacement, a hybrid investigator in which the LLM designs preregistered experiments on the fitted models and code decides the verdicts, has precedent outside tabular ranking (MAIA, InterpAgent, Agentic XAI) but no proof that it finds more than a narrator does. Research also shows such loops degrade after a few rounds and are prone to confirmation bias. Before building 1.1 evidence and 1.2 investigator, a go/no-go spike measures whether the investigator is worth building.

## What Changes

- A throwaway prototype under `spikes/investigator/`, outside the `featranker` package and never shipped:
  - a synthetic regression benchmark with planted, known mechanisms (linear, threshold, interaction, correlated duplicate, overfit-prone model, pure noise);
  - three arms with one shared structured-findings output format: **narrator** (report only, like pasting into a chat), **evidence** (report plus a fixed evidence battery, one LLM call), **investigator** (evidence plus up to 3 preregistered follow-up experiments);
  - code-only scoring of each arm's findings against the planted ground truth (no LLM judge);
  - a replayable trail of every investigator step;
  - tests driven by a scripted fake LLM.
- A preregistered decision rule, fixed before any real LLM run, mapping results to one of three outcomes: build 1.1 + 1.2, build 1.1 + an evidence-aware `explain()` only, or deprecate the LLM feature.
- A findings report (`spikes/investigator/FINDINGS.md`) with the measured results and the outcome, for Signal review.
- Real LLM runs are done by the maintainer with their own API key; results are written to files in the repo for analysis.

## Non-goals

- No change to the `featranker` package, its public API, the report schema, the CLI, or dependencies. **The public report schema does not change.**
- No production-quality investigator, action set, or prompt; the spike answers one question and is then frozen.
- Classification tasks, real-world datasets, and models beyond a small fixed set (one linear, three tree-based) are out of scope for the spike.
- No LLM-as-judge scoring and no human user study.
- Deep-learning or image/text models.

## Capabilities

### New Capabilities

None. The spike is tooling outside the package; no spec-level behavior changes (`skip_specs: true`).

### Modified Capabilities

None.

## Impact

- New directory `spikes/investigator/` (code, tests, results, findings). Not in `packages`, not in CI `testpaths`.
- Spike-only dependencies (`shap`, an OpenAI-compatible client) installed in the maintainer's environment, not added to `pyproject.toml`.
- Outcome decides the scope of the next changes: `evidence-for-why` (1.1) and `llm-investigator` (1.2), or deprecation of `explain()`/`to_prompt()` under the stability policy.
