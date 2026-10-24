## 1. Cache

- [x] 1.1 Test: predict-call count (regression two scorers; classification accuracy + roc_auc mixes predict and predict_proba/decision_function); report equality cached vs fallback; secondary failure entry identical; primary failure message identical; weights; shadows; resampled
- [x] 1.2 Implement D1–D5

## 2. Docs

- [x] 2.1 docs/usage.md multiple-scorers cost note; CHANGELOG 1.19.0 (local); roadmap 1.19 in openspec/config.yaml; version 1.19.0
- [x] 2.2 Full suite passes; mypy clean; `openspec validate scorer-cache --strict` passes
