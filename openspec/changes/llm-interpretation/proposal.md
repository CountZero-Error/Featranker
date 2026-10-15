## Why

FeatRanker's reports are auditable but dense. Many users want a plain-language reading, and they will paste reports into an LLM anyway. Unguided, an LLM tends to reintroduce exactly the misreadings the docs warn against:
- causal claims;
- treating near-zero importance as "unused";
- treating many votes as independent confirmations;
- overreading small rank gaps;
- treating `above_shadow_share` as a p-value.

0.9 makes the guided path the easy one. It builds a prompt that carries FeatRanker's own interpretation rules and the deterministic `summarize()` findings, and it checks every number an LLM reply cites against the report. The core stays offline, keyless, and free of any provider SDK.

## What Changes

- **`featranker.to_prompt(report, *, anonymize=False) -> str`.** A deterministic prompt made of:
  - the interpretation rules;
  - the `summarize()` findings (for a resampled report: the stability table instead, since `summarize` reads single-split reports);
  - a compact JSON view of the report, with per-repeat value arrays and fitted params dropped;
  - a required JSON reply format in which every claim cites the field paths and values it relies on.

  It works for single-split and resampled reports. With `anonymize=True`, feature-group and column names are replaced by `g1…`/`c1…` throughout, so they never appear in the prompt.
- **`featranker.explain(report, llm, *, anonymize=False) -> dict`.** It:
  - builds the prompt;
  - calls `llm(prompt)`, any callable from `str` to `str` the user supplies;
  - parses the JSON reply, tolerating a fenced code block;
  - checks every cited `(path, value)` against the compact view;
  - returns the summary and claims with per-evidence and per-claim `verified` flags, plus `all_verified`.

  With `anonymize=True`, names are mapped back in the returned text. Malformed replies raise `ValueError` with the reply excerpt.
- **Docs:** a new "LLM interpretation" section with a provider-neutral example, a Claude recipe using the official `anthropic` SDK, a local-model note, and privacy guidance. The recipe is not run by the test suite; it needs the user's own key.

**Report schema:** unchanged. Both functions read reports and never modify them.

## Non-goals

- Any provider SDK dependency, API-key handling, or network call in FeatRanker itself.
- Detecting causal or otherwise misleading wording in replies. Only cited numbers are machine-checked; wording is guided by the prompt rules.
- Multi-turn chat, streaming, retries, or cost control. Those belong in the user's callable.
- A CLI.

## Capabilities

### New Capabilities

- `llm-interpretation`: covers
  - prompt construction (rules, findings, compact data, reply format, determinism, anonymization);
  - the `explain` call contract;
  - reply parsing;
  - evidence verification;
  - de-anonymization;
  - error handling.

### Modified Capabilities

(none)

## Impact

- **Code:** a new `featranker/interpret.py`, and exports in `featranker/__init__.py`.
- **Dependencies:** none. `json` and `re` from the standard library.
- **Docs:**
  - `docs/usage.md`: a new section;
  - `README.md` and `docs/README.md`: table entries.
- **Tests:** a new `tests/test_interpret.py` using fake `llm` callables.
- **Version:** 0.9.0.
