## 1. auto_groups

- [x] 1.1 Test: near-copy pair grouped as `{"e+f": ["e", "f"]}`; independent columns give `{}`; complete-linkage scenario keeps `a` and `c` apart; constant column never grouped; names from DataFrame, `feature_names`, and `x0..`; ordering by first member; threshold validation (0, negative, > 1, non-number); generated-name collision raises; X unchanged; result usable as `feature_groups` in `rank_features`
- [x] 1.2 Implement `auto_groups` in `featranker/evidence.py` and export it (D1–D3); update the public-names test for seven names

## 2. Selector and CLI

- [x] 2.1 Test: `ConsensusSelector(feature_groups="auto")` groups a near-copy pair in `report_["feature_groups"]`; `get_params()` keeps `"auto"` and `clone` works; another string raises ValueError in `fit()`; inside a Pipeline + cross_val_score the groups come from the fold (check via a fitted step's report); CLI `--feature-groups auto` groups the pair; a YAML path still works
- [x] 2.2 Implement the selector and CLI handling (D4, D5); version 1.3.0

## 3. Docs and checks

- [x] 3.1 Docs: docs/usage.md (correlated features: `auto_groups`, threshold meaning, selector `"auto"`, CLI `auto`, link to 1.1 correlation evidence), docs/api.md (new public name), README mention, CHANGELOG.md 1.3.0 entry (local); docs examples run and output pasted
- [x] 3.2 Full suite passes; mypy clean; `openspec validate auto-groups --strict` passes
