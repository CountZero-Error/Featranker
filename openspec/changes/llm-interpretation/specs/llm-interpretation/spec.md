## Purpose

Let users get a plain-language reading of a FeatRanker report from any LLM they choose. FeatRanker supplies a prompt that carries its interpretation rules and checks every number the reply cites, without taking any API key or making any network call itself.

## ADDED Requirements

### Requirement: Prompt construction
`to_prompt(report, *, anonymize=False)` SHALL return a string containing, in order:
1. interpretation rules;
2. precomputed findings;
3. a JSON data view;
4. reply-format instructions.

The rules SHALL state:
- rankings are associations on this evaluation data, not causal effects;
- equal `median_rank` ties the primary score only; single-split consensus uses `mean_rank` next and stability uses `top_k_share`, and small gaps alone do not establish statistical significance;
- near-zero importance or `mean <= std` is not proof a feature is unused, and correlated features share credit;
- `n_models` measures coverage, and `n_families` counts more independent views;
- `above_shadow_share` and stability shares are descriptive, not p-values;
- every number stated must be cited from the data view.

For a single-split report, the findings SHALL be `summarize(report)`. For a resampled report, the stability table in the data view stands in for them.

The data view SHALL omit per-repeat `values` arrays and fitted `params`. Failure `message` text SHALL NOT appear anywhere in the prompt, including in the findings. A failure `error_type` that is not a Python built-in exception name (or `NotFittedError`) SHALL appear as `CustomError`. The same report and arguments SHALL always give the same string. `to_prompt` SHALL NOT modify the report and SHALL raise `TypeError` for a non-dict.

#### Scenario: Rules and data present
- **WHEN** `to_prompt(report)` is called on a single-split report
- **THEN** the string contains the causal-language rule, each `summarize` finding code, and every consensus `feature_group`

#### Scenario: No failure messages
- **WHEN** a failure message contains feature names and data values
- **THEN** neither appears in the prompt, with or without anonymization, and the report is unchanged

#### Scenario: Compact and deterministic
- **WHEN** `to_prompt(report)` is called twice
- **THEN** both strings are equal, contain no `"values"` arrays and no `"params"`, and the report is unchanged

#### Scenario: Resampled report
- **WHEN** `to_prompt` is given a report from `rank_resampled`
- **THEN** the data view contains the `stability` rows and `n_splits`, and no error is raised

### Requirement: Anonymization
With `anonymize=True`, these caller-chosen identifiers SHALL be aliased in stable order wherever the report stores them as identifiers:
- feature groups, as `g1, g2, …`;
- columns, as `c1, c2, …`;
- model names not in the built-in registry, as `m1, …`;
- family names not in the built-in registry, as `f1, …`;
- a scorer name that is not a scikit-learn scorer, as `custom_scorer`.

The prompt SHALL instruct the model to wrap aliases present in the data as `[[g1]]`-style markers, keep unaliased built-in names unchanged, and not invent aliases. Fixed vocabulary, such as failure stages, is not an identifier and is not renamed.

#### Scenario: Names hidden
- **WHEN** a report with groups `bmi` and `serum` is passed with `anonymize=True`
- **THEN** the prompt contains neither `bmi` nor `serum`, and contains `g1` and `g2`

#### Scenario: Custom model and scorer names hidden
- **WHEN** a custom model `patient_secret` and a callable scorer are used with `anonymize=True`
- **THEN** `patient_secret` and the scorer's name do not appear, and built-in names such as `ridge_regression` and `r2` do

### Requirement: Explain contract
`explain(report, llm, *, anonymize=False)` SHALL:
- call `llm` exactly once, with the `to_prompt` string;
- require `llm` to be callable, else raise `TypeError`;
- require the result to be a `str`, else raise `TypeError`.

FeatRanker SHALL NOT import any LLM SDK or perform network access itself.

#### Scenario: One call with the prompt
- **WHEN** `explain(report, fake)` is called
- **THEN** `fake` receives exactly one argument, equal to `to_prompt(report)`

### Requirement: Reply parsing
The reply SHALL be parsed as JSON of the form `{"summary": str, "claims": [{"text": str, "evidence": [{"path": str, "value": number|string|bool|null}]}]}`. Evidence values that are lists, objects, or non-finite numbers SHALL be rejected. A single fenced code block, with or without a `json` tag, SHALL be unwrapped first. Invalid JSON or a wrong shape SHALL raise `ValueError` that includes the first 200 characters of the reply.

#### Scenario: Fenced JSON
- **WHEN** the reply is a valid reply object inside a ```json fence
- **THEN** it is parsed normally

#### Scenario: Malformed reply
- **WHEN** the reply is plain prose
- **THEN** `ValueError` is raised containing the start of the reply

### Requirement: Evidence verification
Each evidence `path` SHALL be resolved against a snapshot of the data view taken when the prompt was built. Paths use dot keys and `[i]` list indices, for example `consensus[0].median_rank`, or an RFC 6901 JSON Pointer starting with `/` for keys containing `.`, `[`, or `/`. A pointer component with a `~` not followed by `0` or `1` SHALL NOT resolve.

An evidence item is `verified` when the path resolves and the value matches:
- numbers (not booleans) within `rel_tol=0.01` or `abs_tol=0.01`; a number too large to compare is not verified;
- other values with the same type and an equal value, so `true` never matches `1`.

A claim is `verified` when it has at least one evidence item and all of them are verified. The result SHALL include `all_verified`, which is true when every claim is verified.

#### Scenario: Correct citation
- **WHEN** a claim cites `consensus[0].median_rank` with the report's value rounded to two decimals
- **THEN** that evidence and claim are verified

#### Scenario: Wrong number or path
- **WHEN** a claim cites a wrong value, or a path that does not exist
- **THEN** that evidence is not verified, the claim is not verified, and `all_verified` is false

#### Scenario: No cross-type match
- **WHEN** `in_consensus` (true) is cited as `1`, or `n_models` is cited as `true`
- **THEN** neither is verified

#### Scenario: JSON Pointer
- **WHEN** a model `model.v1` and a feature `lab[0]` are cited via `/models/model.v1/importance/lab[0]/mean`
- **THEN** the evidence is verified

#### Scenario: Snapshot
- **WHEN** the `llm` callable changes the report and then cites the changed value
- **THEN** the evidence is not verified

#### Scenario: Uncited claim
- **WHEN** a claim has an empty evidence list
- **THEN** it is not verified

### Requirement: De-anonymization
With `anonymize=True`, only `[[g…]]`, `[[c…]]`, `[[m…]]`, and `[[f…]]` markers in the returned `summary` and claim texts SHALL be mapped back to the original names; unmarked text and unknown markers SHALL be left unchanged. Every alias SHALL restore, including both the group and column aliases of a singleton group. Evidence paths and values SHALL be returned as cited.

#### Scenario: Names restored
- **WHEN** the reply text says "[[g1]] ranks first; plain g1 stays" and `g1` stands for `bmi`
- **THEN** the returned claim text says "bmi ranks first; plain g1 stays"
