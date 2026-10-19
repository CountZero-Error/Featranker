## 1. Preset

- [x] 1.1 Test: `group="fast"` is accepted; an unknown group and `fast` with `models=` raise ValueError; the fitted model sets equal the spec's lists for clf and reg; fast families equal all families per task, each once; no initialization failures or warnings with `group="fast"` even when xgboost/catboost imports fail (monkeypatch)
- [x] 1.2 Tag representatives with `fast: true` in `importance_config.yaml` and implement selection and validation (D1, D3)

## 2. Pass-through

- [x] 2.1 Test: `rank_resampled()` on a fast ranker uses the fast set in every split; `ConsensusSelector(group="fast")` likewise; CLI `--group fast` report lists the fast set; typing consumer accepts `group="fast"`
- [x] 2.2 Implement the `Literal` types, `ConsensusSelector` and CLI choice; version 1.2.0

## 3. Docs and checks

- [x] 3.1 Docs: docs/models.md (a Fast column or marker, and the `group="fast"` option), docs/usage.md (a short speed section: model counts, same families, svr cost note), README mention, CHANGELOG.md 1.2.0 entry (local)
- [x] 3.2 Full suite passes; mypy clean; `openspec validate fast-preset --strict` passes
