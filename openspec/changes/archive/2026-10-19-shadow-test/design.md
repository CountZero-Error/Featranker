## Context

`_build_stability()` aggregates per-split consensus places into stability rows and averages `above_shadow_share` over ranked splits when shadows ran. `ConsensusSelector.fit()` filters `report["stability"]` by `min_shadow_share` and `k`. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:** a significance statement that costs no extra fits, with its limits visible in the report.

**Non-Goals:** exact error control under dependent splits (see Risks).

## Decisions

### D1. One trial per split, family consensus as the hit
A hit is `above_shadow_share > 0.5` in that split's consensus: a strict majority of the family-weighted vote put the group above that model's strongest shadow. Counting per model instead would count correlated models on the same split as separate trials. Missing consensus rows are non-hits and the trial count is `n_splits`, matching `top_k_share`.

### D2. Null probability 0.5
Under no effect, a group beats the *maximum* of all shadow groups with probability at most 0.5 (about 1/2 with one real group, much lower with many), so testing against 0.5 is conservative. This is Boruta's choice; the docs say so.

### D3. Computation
`p = scipy.stats.binom.sf(hits - 1, n, 0.5)` (available in every scipy that scikit-learn 1.5 supports; `binomtest` needs a newer scipy). Benjamini–Hochberg is implemented in a few lines of NumPy (sort, `p * m / rank`, reverse cumulative minimum, cap at 1) rather than `scipy.stats.false_discovery_control`, which needs scipy ≥ 1.11.

### D4. Report placement
Three keys per stability row, added only when shadows ran, next to `above_shadow_share`; a `shadow_test` block at the top level with `min_p_value = 0.5 ** n_splits`. No warning for few splits: the default `cv=5` would warn on every shadow run; `min_p_value` and the docs carry the message instead.

### D5. Selector
`max_shadow_q` joins `min_shadow_share` as a filter applied before `k`; either filter turns shadows on. Validation `(0, 1]`.

## Risks / Trade-offs

- [Splits overlap (k-fold training sets share rows), so trials are not independent; the p-value is approximate] → documented as descriptive-plus, same caveat as Boruta's iterations.
- [Low power with few splits: at 5 splits only a 5/5 group reaches p = 0.031 (4/5 gives 0.19), and BH raises that above 0.05 unless most groups are 5/5] → `min_p_value` in the report; docs recommend `RepeatedKFold` / 10+ splits; the selector's empty-selection warning names `n_splits` and `min_p_value` when `max_shadow_q` is set.
- [Users read q < 0.05 as proof of a causal effect] → docs: it says the group beats shuffled noise across splits, nothing more.
