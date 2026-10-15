## 1. Data view and prompt

- [x] 1.1 Write tests:
  - the prompt contains the rules (causal rule text), each `summarize` finding code, and every consensus group;
  - it is deterministic;
  - it has no `values` and no `params`;
  - the report is unchanged;
  - a resampled report works and includes stability and `n_splits`;
  - a non-dict raises `TypeError`.

  Confirm they fail.
- [x] 1.2 Implement `featranker/interpret.py`: `_data_view` and `to_prompt`, with the fixed rules and reply-format text. Export `to_prompt`.

## 2. Anonymization

- [x] 2.1 Write tests:
  - with `anonymize=True`, no original group or column name appears anywhere in the prompt;
  - `g1…`/`c1…` are present in a stable order;
  - this holds for a column name that collides with a model or stage name (such as `fit`).

  Confirm they fail.
- [x] 2.2 Implement positional anonymization of the report, with summarize regenerated on the anonymized report, and keep the mapping.

## 3. explain

- [x] 3.1 Write tests with fake callables:
  - exactly one call, with the `to_prompt` string;
  - a non-callable or non-str return raises `TypeError`;
  - fenced JSON is parsed;
  - prose and wrong shapes raise `ValueError` with an excerpt;
  - a correct citation (rounded) is verified;
  - a wrong value, a missing path, and an uncited claim are not verified;
  - `all_verified`;
  - de-anonymization restores names in the text (whole tokens only), with evidence returned as cited.

  Confirm they fail.
- [x] 3.2 Implement `explain`: path resolver, tolerance check, parsing, and de-anonymization. Export `explain`.

## 4. Docs

- [x] 4.1 `docs/usage.md`: add an "LLM interpretation" section covering:
  - what the prompt contains;
  - a provider-neutral `explain` example with a fake or local callable and its real output;
  - a Claude recipe using the official SDK;
  - the result shape;
  - the warning that verified numbers do not mean a correct interpretation;
  - privacy and `anonymize`.

  Mention it in the API intro.
- [x] 4.2 `README.md` and `docs/README.md`: table entries.
- [x] 4.3 Verify the doc claims and the example byte-for-byte, render through the GitHub API, and check anchors and escapes.

## 5. Release prep

- [x] 5.1 Bump to 0.9.0. Add a CHANGELOG entry.
- [x] 5.2 Run the full suite outside the sandbox, on sklearn 1.5.1 and the latest.
