## 1. Setup

- [x] 1.1 Add the `shap` extra and add `shap` to the `test` extra in `pyproject.toml`; test that the extras match the public-api spec (update `tests/test_public_api.py` or the existing extras test)
- [x] 1.2 Version 1.1.0

## 2. Flag and train score

- [x] 2.1 Test: `evidence` defaults to False, rejects non-bool with TypeError, default reports have no new keys, and a ranker unpickled without the attribute behaves as `evidence=False`
- [x] 2.2 Implement the flag, keep training data in `fit()` when on, `__setstate__` default
- [x] 2.3 Test: `train_score` and `overfit_gap` on every model report (unpruned tree overfits; works with shadows); a train-scoring error becomes a stage `"evidence"` failure without dropping the model
- [x] 2.4 Implement train score in the per-model ranking job (D2)

## 3. Correlated groups

- [x] 3.1 Test: planted copy pair found with rho > 0.9; independent features give no pairs; multi-column groups use the strongest member pair with its sign; constant column skipped; shadows excluded; sort order
- [x] 3.2 Implement correlation in `featranker/evidence.py` and add `evidence.correlation` to the report (D3)

## 4. SHAP summary

- [x] 4.1 Test (importorskip shap): ridge pipeline → `explainer: "linear"`, random forest → `"tree"`; every group present; planted positive effect has direction > 0.9; multi-column group direction is null; binary and multiclass classification shapes; KNN listed in `unsupported`; row cap of 1000, evenly spaced; shadow columns not reported; results identical across `random_state` values
- [x] 4.2 Test: with `shap` import blocked (monkeypatch), a UserWarning is raised, `evidence.shap == {"available": False}`, and the rest of the evidence is present
- [x] 4.3 Implement the SHAP helper and its wiring (D4, D5)
- [x] 4.4 Test: a full evidence report passes `json.dumps(report, allow_nan=False)`

## 5. Resampled and CLI

- [x] 5.1 Test: `rank_resampled()` with `evidence=True` puts `evidence` in every split report and not at top level; CLI `--evidence --output` writes `evidence`
- [x] 5.2 Implement the pass-through and the `--evidence` flag

## 6. Summary findings

- [x] 6.1 Test: `overfit` finding (share 0.8 listed, healthy model not, null train score skipped, no finding without `train_score` or with `baseline_score` None); `correlated_groups` finding; ordering after `below_shadow`; stage `"evidence"` failures give an info `failures` finding
- [x] 6.2 Implement the two rules and the new failure stage (D6)

## 7. Deprecation

- [x] 7.1 Test: `to_prompt()` and `explain()` each emit exactly one FutureWarning mentioning 2.0; outputs are unchanged from 1.0; existing interpret tests still pass (filter the warning there)
- [x] 7.2 Implement the warnings and the private prompt builder (D7)

## 8. Docs and checks

- [x] 8.1 Docs: README and docs/usage.md (evidence section with a checked example, shap extra, pickling note), docs/report.md (new keys, units note for SHAP), docs/api.md (deprecations), deprecation notes on the LLM section; CHANGELOG.md 1.1.0 entry (local file)
- [x] 8.2 Full test suite passes with and without `shap` installed; `mypy featranker tests/typing/consumer.py` clean; `openspec validate evidence-for-why` passes
