# Leakage-Safe Feature Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide explicit train/evaluation separation, grouped permutation importance, rank consensus, legacy migration, core-only installation, tests, and documentation for Featranker 0.2.0.

**Architecture:** Keep implementation in `featranker/importance.py`. Add small private validation, grouping, failure, and consensus helpers around existing YAML model loading. Normal `fit()` stores only fitted models and feature metadata; prep-file compatibility alone stores training data for deprecated `rankFeatures()`.

**Tech Stack:** Python 3.10+, NumPy, scikit-learn, PyYAML, tqdm, pytest; pandas only for DataFrame compatibility tests; setuptools/build for distributions.

## Global Constraints

- Never rank on training data through `rank_features()`; evaluation arrays are mandatory.
- Do not implement cross-validation, splitting, preprocessing, selection, SHAP, UI, or clinical logic.
- Core operation and tests must not require XGBoost, LightGBM, or CatBoost.
- Importance is `baseline_score - permuted_score` using one scorer.
- Multi-column groups use one row permutation per repeat.
- Failed models are excluded, reported, and never converted to zero.
- Normal `fit()` does not retain raw training data.
- Version becomes `0.2.0`; do not push or publish.

---

### Task 1: Explicit Fit API and Input Validation

**Files:**
- Create: `tests/test_importance.py`
- Modify: `featranker/importance.py`

**Interfaces:**
- Produces: `FeatureRanker.fit(X, y, feature_names=None) -> FeatureRanker`
- Produces: fitted attributes `feature_names_`, `n_features_in_`, `trained_models`, `fit_failures_`, `label_encoder`
- Consumes: configured nested model mapping returned by `init_models()`

- [ ] **Step 1: Write failing validation and fit tests**

Create lightweight `MeanRegressor`, `FailFitRegressor`, and `ranker_with_models(monkeypatch, models)` helpers. Add tests equivalent to:

```python
def test_numpy_fit_requires_feature_names(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    with pytest.raises(ValueError, match="feature_names"):
        ranker.fit(np.ones((4, 2)), np.arange(4))

def test_fit_validates_rows_target_and_names(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    with pytest.raises(ValueError, match="row"):
        ranker.fit(np.ones((4, 2)), np.arange(3), ["a", "b"])
    with pytest.raises(ValueError, match="one-dimensional"):
        ranker.fit(np.ones((4, 2)), np.ones((4, 1)), ["a", "b"])
    with pytest.raises(ValueError, match="2 feature names"):
        ranker.fit(np.ones((4, 2)), np.arange(4), ["a"])

def test_dataframe_names_are_retained_and_checked(monkeypatch):
    pd = pytest.importorskip("pandas")
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    ranker.fit(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), [1, 2])
    assert ranker.feature_names_ == ["a", "b"]
    with pytest.raises(ValueError, match="conflict"):
        ranker_with_models(monkeypatch, {"ok": MeanRegressor()}).fit(
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}), [1, 2], ["b", "a"]
        )

def test_fit_raises_when_every_model_fails(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"bad": FailFitRegressor()})
    with pytest.raises(RuntimeError, match="No models fit successfully"):
        ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])
    assert not ranker.is_fitted_

def test_normal_fit_does_not_retain_training_data(monkeypatch):
    ranker = ranker_with_models(monkeypatch, {"ok": MeanRegressor()})
    ranker.fit(np.ones((4, 1)), np.arange(4), ["a"])
    assert not hasattr(ranker, "_legacy_X")
    assert not hasattr(ranker, "_legacy_y")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `rtk pytest -q tests/test_importance.py -k 'fit or dataframe or numpy'`

Expected: failures because safe constructor/`fit()` and fitted-state validation do not exist.

- [ ] **Step 3: Implement minimal fit lifecycle**

Add `_validate_X(X, feature_names=None, expected_names=None) -> tuple[np.ndarray, list[str]]` to validate two-dimensional numeric input and names; `_validate_y(y, n_rows: int) -> np.ndarray` to validate a one-dimensional matching target; and `_failure(model: str, stage: str, error: Exception) -> dict[str, str]` to serialize model failures.

Change constructor to initialize models and state only. Implement `fit()` to validate, label-encode object/string targets, fit selected models, record failures, set fitted metadata only after one success, and raise when none succeed. Do not assign training arrays to the instance.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `rtk pytest -q tests/test_importance.py -k 'fit or dataframe or numpy'`

Expected: selected tests pass.

---

### Task 2: Held-Out Scoring and Grouped Permutation

**Files:**
- Modify: `tests/test_importance.py`
- Modify: `featranker/importance.py`

**Interfaces:**
- Consumes: fitted state from Task 1
- Produces: `rank_features(X_eval, y_eval, *, scoring, feature_groups=None, n_repeats=10, random_state=None) -> dict`
- Produces: `_resolve_feature_groups(feature_names, feature_groups) -> dict[str, list[int]]`

- [ ] **Step 1: Write failing held-out, scorer, and permutation tests**

Add tests equivalent to:

```python
def test_rank_features_scores_evaluation_not_training(monkeypatch):
    seen = []
    def scorer(model, X, y):
        seen.append((X.copy(), np.asarray(y).copy()))
        return model.score(X, y)
    ranker = fitted_ranker(monkeypatch, X_train=np.zeros((6, 2)))
    X_eval = np.arange(8, dtype=float).reshape(4, 2)
    y_eval = np.arange(4, dtype=float)
    ranker.rank_features(X_eval, y_eval, scoring=scorer, n_repeats=2, random_state=7)
    assert all(x.shape == X_eval.shape for x, _ in seen)
    assert np.array_equal(seen[0][0], X_eval)
    assert np.array_equal(seen[0][1], y_eval)

def test_named_scorer_sets_baseline_score(monkeypatch):
    ranker = fitted_ranker(monkeypatch)
    report = ranker.rank_features(X_EVAL, Y_EVAL, scoring="neg_mean_absolute_error", random_state=1)
    expected = get_scorer("neg_mean_absolute_error")(next(iter(ranker.trained_models.values())), X_EVAL, Y_EVAL)
    assert report["models"]["ok"]["evaluation_score"] == expected

def test_group_columns_share_each_row_permutation(monkeypatch):
    observed = []
    def scorer(model, X, y):
        observed.append(X.copy())
        assert np.array_equal(X[:, 1] - X[:, 0], np.full(len(X), 100.0))
        return 0.0
    X = np.column_stack([np.arange(8.0), np.arange(8.0) + 100, np.arange(8.0)])
    ranker = fitted_ranker(monkeypatch, X_train=X, names=["g1", "g2", "other"])
    ranker.rank_features(X, np.arange(8.0), scoring=scorer,
                         feature_groups={"gene": ["g1", "g2"]}, n_repeats=3, random_state=2)
    assert len(observed) > 1

def test_ungrouped_features_become_singletons(monkeypatch):
    report = fitted_ranker(monkeypatch).rank_features(
        X_EVAL, Y_EVAL, scoring="r2", feature_groups={"pair": ["a", "b"]}, random_state=2
    )
    assert report["feature_groups"] == {"pair": ["a", "b"], "c": ["c"]}
```

Parametrize clear failures for an empty group, unknown member, duplicate member within one group, overlap across groups, and group-name/singleton collision.

- [ ] **Step 2: Run tests and verify RED**

Run: `rtk pytest -q tests/test_importance.py -k 'rank_features or group or scorer or singleton'`

Expected: failures because `rank_features()` and group resolution do not exist.

- [ ] **Step 3: Implement scorer and grouped permutation**

Resolve string scorers with `get_scorer`; require callable otherwise. Validate fitted state, evaluation feature count/names, target, positive `n_repeats`, and groups. For each model, call scorer once on baseline and once per group/repeat. Use:

```python
row_order = rng.permutation(len(X_eval))
X_permuted = X_eval.copy()
X_permuted[:, column_indices] = X_eval[row_order][:, column_indices]
importance = baseline_score - scorer(model, X_permuted, y_eval)
```

Seed a fresh generator identically per model so successful models receive identical permutations. Store values, mean, population std, evaluation score, and `evaluation_mode="held_out"`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `rtk pytest -q tests/test_importance.py -k 'rank_features or group or scorer or singleton'`

Expected: selected tests pass.

---

### Task 3: Rank Consensus, Reproducibility, and Failures

**Files:**
- Modify: `tests/test_importance.py`
- Modify: `featranker/importance.py`

**Interfaces:**
- Produces: `_assign_ranks(importance: dict[str, float]) -> dict[str, int]`
- Produces: `_build_consensus(model_reports: dict) -> list[dict]`
- Extends report with structured `failures`

- [ ] **Step 1: Write failing rank/failure tests**

```python
def test_ties_share_minimum_rank_and_consensus_uses_ranks():
    assert _assign_ranks({"b": 2.0, "a": 2.0, "c": 1.0}) == {"a": 1, "b": 1, "c": 3}
    reports = {
        "m1": {"importance": {"a": {"mean": 100.0, "rank": 1}, "b": {"mean": 99.0, "rank": 2}}},
        "m2": {"importance": {"a": {"mean": 0.0, "rank": 2}, "b": {"mean": 1.0, "rank": 1}}},
    }
    consensus = {row["feature_group"]: row for row in _build_consensus(reports)}
    assert consensus["a"]["mean_rank"] == 1.5
    assert consensus["b"]["mean_rank"] == 1.5
    assert "mean_importance" not in consensus["a"]

def test_fixed_seed_is_reproducible(monkeypatch):
    ranker = fitted_ranker(monkeypatch)
    first = ranker.rank_features(X_EVAL, Y_EVAL, scoring="r2", n_repeats=4, random_state=42)
    second = ranker.rank_features(X_EVAL, Y_EVAL, scoring="r2", n_repeats=4, random_state=42)
    assert first == second

def test_partial_ranking_failure_is_reported(monkeypatch):
    ranker = fitted_ranker(monkeypatch, models={"ok": MeanRegressor(), "bad": FailingScoreRegressor()})
    report = ranker.rank_features(X_EVAL, Y_EVAL, scoring=lambda model, X, y: model.score(X, y), random_state=1)
    assert set(report["models"]) == {"ok"}
    assert any(f["model"] == "bad" and f["stage"] == "ranking" for f in report["failures"])

def test_all_ranking_failures_raise(monkeypatch):
    ranker = fitted_ranker(monkeypatch, models={"bad": FailingScoreRegressor()})
    with pytest.raises(RuntimeError, match="No models ranked successfully"):
        ranker.rank_features(X_EVAL, Y_EVAL, scoring=lambda model, X, y: model.score(X, y))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `rtk pytest -q tests/test_importance.py -k 'rank or reproducible or failure or consensus'`

Expected: missing helpers/report behavior fails.

- [ ] **Step 3: Implement deterministic rank consensus and failure semantics**

Assign descending minimum ranks with exact ties. Aggregate only successful model ranks using NumPy median/mean/std with `ddof=0`; report `n_models`. Sort consensus by median rank, mean rank, then group name. Append initialization, fit, and ranking failure records. Raise only when zero fit or zero rank models succeed.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `rtk pytest -q tests/test_importance.py -k 'rank or reproducible or failure or consensus'`

Expected: selected tests pass.

---

### Task 4: Legacy Preparation Workflow

**Files:**
- Modify: `tests/test_importance.py`
- Modify: `featranker/importance.py`

**Interfaces:**
- Preserves: `build_ranker(task, group="all", prep_file=None, prep_class="prepFeature", **kwargs)`, `run_ML()`, `rankFeatures()`
- Produces: legacy report `evaluation_mode="in_sample"`

- [ ] **Step 1: Write failing legacy smoke test**

Create a temporary prep file returning two numeric features and `label`. Patch model initialization to one lightweight estimator. Assert `build_ranker(task="reg", prep_file=str(prep_path), prep_class="Prep").rankFeatures()` emits `DeprecationWarning`, returns `evaluation_mode == "in_sample"`, and retained legacy arrays exist. Assert ordinary `FeatureRanker.fit()` from Task 1 has no retained arrays.

- [ ] **Step 2: Run test and verify RED**

Run: `rtk pytest -q tests/test_importance.py -k legacy`

Expected: current constructor-driven workflow and report schema fail the new assertions.

- [ ] **Step 3: Route prep workflow through `fit()` and shared ranking**

Add `_load_prep_data(prep_file, prep_class, kwargs)`. `build_ranker()` uses default `featureCalc.py`, calls `fit()`, then stores legacy arrays. `rankFeatures()` warns and calls one private ranking implementation with `evaluation_mode="in_sample"` and task default scorer (`accuracy` or `r2`). `run_ML()` warns and refits stored prep data. Constructor with explicit `prep_file` may use this compatibility path; bare constructor stays unfitted.

- [ ] **Step 4: Run test and verify GREEN**

Run: `rtk pytest -q tests/test_importance.py -k legacy`

Expected: legacy tests pass with warning captured.

---

### Task 5: Optional Model Dependencies and Core Packaging

**Files:**
- Modify: `tests/test_importance.py`
- Modify: `featranker/importance.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Core dependencies: NumPy, scikit-learn, PyYAML, tqdm
- Extras: `xgboost`, `catboost`, `all`, `test`

- [ ] **Step 1: Write failing optional-dependency test**

Patch `importlib.import_module` to raise `ModuleNotFoundError` for `xgboost` and `catboost` while delegating sklearn imports. Initialize a tree ranker under `pytest.warns(UserWarning, match="optional dependency")`; assert core sklearn models remain and optional model failures have stage `initialization`. Parse `pyproject.toml` with `tomllib`; assert optional packages are absent from core dependencies and present in extras.

- [ ] **Step 2: Run test and verify RED**

Run: `rtk pytest -q tests/test_importance.py -k optional`

Expected: current loader aborts or packaging assertions fail.

- [ ] **Step 3: Implement optional import skipping and extras**

Catch missing imports only for configured optional top-level modules, warn with model/dependency name, record failure, continue. Keep other import/config errors explicit. Set version `0.2.0`; move XGBoost/CatBoost from core to extras; remove LightGBM from core without adding a model. Put pytest, pandas, and build in test/development extras and requirements as appropriate.

- [ ] **Step 4: Run test and verify GREEN**

Run: `rtk pytest -q tests/test_importance.py -k optional`

Expected: optional tests pass without importing those libraries.

---

### Task 6: Documentation, Changelog, and CI

**Files:**
- Modify: `README.md`
- Create: `CHANGELOG.md`
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Documents: new API/report, leakage boundaries, failures, optional extras, reproducibility, migration
- CI command: `pytest -q`

- [ ] **Step 1: Update README**

Replace in-sample quick start with held-out regression and grouped genotype examples. State ranking occurs inside each training portion; outer/final test data cannot guide selection. State rankings support selection/ablation evidence, not causal or clinical importance. Document exact report fields, `evaluation_mode`, failure rules, tie handling, seeds, optional install commands, and legacy warnings.

- [ ] **Step 2: Add release notes**

Create `CHANGELOG.md` with `0.2.0` added/changed/deprecated sections covering safe fit/rank split, grouped permutation, scorer/rank consensus, optional dependencies, and legacy in-sample marker.

- [ ] **Step 3: Add minimal core CI**

Create GitHub Actions workflow on push/pull_request using Python 3.10 and 3.12, `pip install -e '.[test]'`, and `pytest -q`. Do not install optional estimator extras.

- [ ] **Step 4: Validate documentation/package metadata**

Run: `rtk pytest -q`

Expected: complete test suite passes.

---

### Task 7: Build and End-to-End Verification

**Files:**
- No source changes expected
- Generated then removed/ignored: `dist/`, `build/`, `*.egg-info/`

**Interfaces:**
- Verifies local package artifacts and core-only workflow

- [ ] **Step 1: Run complete tests fresh**

Run: `rtk pytest -q`

Expected: zero failures.

- [ ] **Step 2: Build both distributions**

Run: `rtk python -m build`

Expected: one `.whl` and one `.tar.gz` under `dist/`, exit code 0.

- [ ] **Step 3: Run core-only end-to-end example**

Use a temporary Python process with `FeatureRanker.init_models` patched to one `DecisionTreeRegressor`, fit synthetic train data, rank separate evaluation data with `neg_mean_absolute_error` and one grouped feature, and assert `evaluation_mode == "held_out"` plus non-empty consensus.

- [ ] **Step 4: Review compatibility and accidental files**

Run `rtk git diff --check`, `rtk git status --short`, and inspect `rtk git diff --stat` plus focused full diff. Confirm only intended source/tests/docs/CI/metadata files changed and `.codegraph/` remains untouched/untracked.

- [ ] **Step 5: Report evidence**

Report exact test/build/example commands and outputs, changed files, new API, migration, local commits, and remaining statistical limitations. Do not push or publish.
