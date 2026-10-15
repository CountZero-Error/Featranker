## Context

- Reports are plain dicts: single-split from `rank_features`, resampled from `rank_resampled`.
- `summarize()` already turns the reading guide into findings, for single-split reports only.
- The project rules forbid network calls or API keys in the core package.
- Requirements are in specs/. Motivation is in proposal.md.

## Goals / Non-Goals

**Goals:**
- One small module, `featranker/interpret.py`, with no new dependency, fully testable offline with fake callables.
- Machine-checkable output: the LLM must cite field paths, and FeatRanker checks them.

**Non-Goals:**
- Judging wording quality or detecting causal language (proposal non-goal).

## Decisions

**1. The LLM is a user-supplied `Callable[[str], str]`.**

It is the smallest possible contract. It works with any provider, a local model, or a test fake, and keeps keys and SDKs out of FeatRanker.

Alternatives considered:
- a `featranker[llm]` extra with built-in clients. Rejected: SDK churn and key handling in the core;
- a `litellm` dependency. Rejected: a heavy transitive dependency for one call.

**2. A compact data view, shared by the prompt and the verifier.**

`_data_view(report)` builds the JSON the model sees. Evidence paths are resolved against that same object, so "what the model could cite" and "what we check" can't drift apart. It:
- drops `values` arrays and `params`, to save tokens and noise;
- keeps scores, the consensus, family ranks, shadow fields, and failures, cut down to `model`/`stage`/`error_type`;
- for resampled reports, keeps the stability rows and split metadata, but not each split's full report.

Alternative considered: verifying against the full report. Rejected because paths would then refer to data the model never saw.

**3. Anonymization is positional, not string replacement.**

Names are mapped where they occur structurally:
- `feature_groups` keys and members;
- importance and shadow-importance keys;
- consensus and stability `feature_group`;
- `summarize` evidence group lists.

Then `summarize` messages are regenerated from the anonymized report, so no free text leaks names.

Alternative considered: a global find-and-replace. Rejected because a column named `fit` or `ridge` would corrupt stage or model names.

**4. Reply format and tolerance.**

- The reply is JSON with `summary` and `claims[{text, evidence[{path, value}]}]`, and the prompt asks for JSON only.
- One fenced code block is tolerated.
- Numeric matching uses `rel_tol=abs_tol=0.01`, so rounding to two decimals passes, while real differences (such as rank 3 vs 3.25) fail.

**5. Determinism.**

The prompt serializes the data view with `json.dumps(sort_keys=True, indent=1)`, and the rules are fixed text, so identical reports give identical prompts. This is useful for prompt caching on the user's side.

**6. Docs recipe for Claude.**

The example uses the official `anthropic` Python SDK with the current default model, following that SDK's documented patterns, in the user's own code. FeatRanker never imports it. A local-model example shows the same callable shape.

## Risks / Trade-offs

- [Verified numbers do not mean a correct interpretation; a claim can cite true numbers and still overreach] → Docs say so; the prompt rules guide wording.
- [Prompts leak feature names to a third party] → `anonymize=True`, plus docs privacy guidance. Data rows are never included.
- [A large resampled report could produce a long prompt] → The compact view drops per-split reports and value arrays. The docs note that size grows with the number of feature groups.

## Migration Plan

- Additive. Release as 0.9.0.
