# Changelog

## Unreleased

Polish: `--output` writes a report with other hard links, or whose owner or group cannot be copied, in place; a new report follows the umask without reading it; on Windows a report open in another program raises a clear error. A non-iterable `estimators_` counts as one member, unhashable index labels warn instead of crashing, and the selector docs and constants match the code.

Review 18 fixes: the index check tells NumPy datetime64 and timedelta64 NaT labels of different units apart, as `.loc` does.

Review 17 fixes: the index check tells different missing labels apart (NaN, None, NaT, pd.NA), as `.loc` does, so swapping rows labelled with different missing values is refused instead of passing silently.

Review 16 fixes: an index whose tuple labels hold missing values in the same order as X (for example `(nan, 1)`) is no longer refused as reordered.

Review 15 fixes: each scorer call also gets its own column and row Index objects, and the index check treats missing labels as equal at any tuple depth.

Review 14 fixes: the dummy baseline is scored on the models' columns, shadow columns included (a scorer that reads a shadow column no longer disables the gate); `ConsensusSelector.fit()` raises `ValueError` when the dummy could not be scored in a split, instead of selecting without the gate (`rank_features()` and `rank_resampled()` still report `gate_off`); the index check treats missing labels (NaN, None, also inside a MultiIndex) as equal; each scorer call during baseline scoring gets its own DataFrame, so one scorer replacing a column cannot change what the next sees. Documentation: a DataFrame keeps its index through `set_output(transform="pandas")`; only a NumPy input gets a new `RangeIndex`.

Review 13 fixes: the dummy baseline is scored on the same input form and read-only views as the models (a DataFrame-aware scorer no longer disables the dummy gate, and a scorer cannot change `y` through the dummy); a pandas `y`, `y_eval`, `sample_weight`, `eval_sample_weight`, `groups` or `shuffle_groups` is checked against the index of a DataFrame `X` (same labels in another order raise `ValueError`, other labels warn and are used by position); an integer `cv` with `groups` warns that `GroupKFold` is needed. Documentation: zero-weight rows still donate values to permutations; `same_holdout` false means not identical, not new rows.

Documentation: a banner and a consensus chart (made by `docs/assets/make_consensus.py` from the quick-start example) in the README; a Simplified Chinese README (`README.zh-CN.md`) and manual (`docs/zh-CN/`), with language switches on every translated page; the manual index's broken navigation row is fixed.

## 2.0.0 - 2026-11-22

Versions 1.22, 1.23 and 1.24 were never released; their entries below are folded into 2.0.0.

Review fixes (rc-fixes): a meta-estimator whose fitted members contain a selector (bagging over a grid search that picks one) counts as unverified for shadow evidence; a refit refused for anything but the seed leaves the ranker unfitted; a new CLI report follows the umask, and volumes without ACL or xattr support still receive it; the Windows report branch type-checks on every platform.

Release review fixes:

- Grouped permutations preserve the types of Python labels. Mixed, unsortable labels raise an error. Object-array date features follow the same validation as date arrays.
- Integer cross-validation accepts the full documented 64-bit seed range. Existing 32-bit seeds retain their previous folds. Reassigned shadow seeds are validated before ranking.
- A refit discards evidence from the previous training set. Enabling evidence after a fit that did not retain training data requires another fit.
- Custom FunctionTransformer subclasses cannot supply verified shadow evidence based on an assumed identity transform. Ablation checks the fully qualified class of each voting model when the report records it.
- CLI overwrites preserve report permissions. The source distribution includes the changelog and excludes the repository test suite.

Documentation (manual-2): a new manual in docs/ — Getting started, Concepts, task guides (models and scoring, custom models, data, validation, noise and evidence, selection, performance), an API reference with an "Input rules" section, a CLI page and Troubleshooting; `docs/usage.md` is now a landing page; every public class and function has a NumPy-style docstring, so `help()` and IDEs describe every parameter.

Review fixes (codex-review-12): user models are trained from the snapshot `ablate()` refits, so an estimator that draws a seed from a shared RandomState on each clone (XGBoost) refits with the seed it was fitted with; the SHAP direction of a feature with missing holdout values uses the rows where it is present; the CLI writes `--output` through a temporary file and a replace (directly when no temporary file can be made beside it), so a failed save keeps the earlier report; the real-data `--render` survives a console that cannot encode some characters.

Review fixes (codex-review-11): `ablate()` accepts an unchanged ranker in a spawned process and compares estimators nested in containers (grid candidates), pandas labels, record fields, masked fill values and n-dimensional sparse entries; across scikit-learn versions params and param_hashes are compared together; SHAP lists regressors that predict several columns as unsupported; options assigned as NumPy integers keep reports strict JSON; changing `missing` on a ranker of built-in models needs a new FeatureRanker; the real-data script resolves `--out` once and refuses an output ending in `.lock`; the sdist leaves out tests/.

Review fixes (codex-review-10): reports record `param_hashes` (additive) and `ablate()` refuses a changed pipeline order, container or array parameter, ColumnTransformer selector, categorical columns or `model_input`; older pickles refit the fitted model; regressors with several SHAP outputs are unsupported; `rank_features()` refuses options changed after `fit()`; real-data runs lock their output file and keep a wheel's version.

Review fixes (codex-review-9): `ablate()` refits with the settings and one-hot density of the last fit; tree SHAP covers exact scikit-learn, xgboost, lightgbm and catboost classes only, and LightGBM random forests are unsupported (SHAP summed what the prediction averages); one-hot SHAP only for built-in models; integer options refuse NumPy durations, `n_jobs` must be None or a nonzero int (no bool), `max_categories` and `min_shadow_families` are below 2**63, and options are checked before the data are read; real-data runs save atomically and refuse duplicate seeds or another version on resume.

Review fixes (codex-review-8): SHAP evidence needs shap >= 0.49 (an older one is treated like a missing one); `ablate()` refits 1.x reports of built-in models with categorical columns on the dense one-hot coding 1.x used; a real-data `--resume` keeps the stored commit; seeds and counts are checked before any grouping or fitting, and counts must be below 2**63.

Review fixes (codex-review-7): SHAP lists histogram boosting with native categorical features and CatBoost with Lossguide or Depthwise trees as unsupported (their values were wrong), and gblinear explains NaN as 0; `random_state` and `shadow_seed` must be None or an integer from 0 to 2**64 - 1; `ablate()` warns about old reports before checking models; the CLI rejects only a literal `--output -`; the real-data script records failing splits and no default commit.

Review fixes (codex-review-6): linear SHAP covers exact scikit-learn and xgboost classes only (custom linear estimators, wrappers, multiclass gblinear and gblinear with a `missing` sentinel are unsupported), which replaces the output check of codex-review-4 and -5; `ablate()` compares parameter types as well as values, and pre-2.0 reports by the 1.x recording rule; the real-data RF arm never selects an all-missing column.

Review fixes (codex-review-5): the linear SHAP output check accepts only an exact match, directly or through a log or logit link (no longer any monotone function), on the explained and the background rows, with a finite rounding allowance; NumPy scalar parameters are recorded in `params`, so `ablate()` compares them; tests pass on scikit-learn 1.9.

Review fixes (codex-review-4): linear SHAP values are kept only when they reproduce the model's own output (otherwise the model is listed as unsupported), Pipeline subclasses are refused as the final step too, and one-vs-one SVC/NuSVC are unsupported; each held-out model report records `components` (required by the schema), which `ablate()` compares on any scikit-learn version; `ablate()` raises when a voter is missing from `trained_models`; only scikit-learn's own scorer objects are named by their score function.

Review fixes (codex-review-3): SHAP trusts only exact scaler, imputer and fold classes (subclasses are unsupported) and `mean_abs` is never inf; `ablate()` refits the configuration `fit()` froze (`fit_models_`); reports carry `schema_version: 2`, which alone drives the `ablate()` old-report warning; the ablation median keeps subnormal scores; SHAP lists Pipeline subclasses as unsupported (they may reorder columns).

Review fixes (codex-review-2): `ablate()` warns on reports made before 2.0 (no `run` block or featranker 1.x) and its median no longer overflows; SHAP folds a one-hot pipeline only when its ColumnTransformer sources are distinct and cover every input column once; a callable scorer without a usable `__name__` is named by its type; the real-data benchmark records `script_sha256` and `--resume` refuses a different script; docs: README recipe decides on validation data and uses the test set once, schema promise scoped to 2.x, sparse one-hot limits, real-data split wording.

Breaking: `to_prompt()`, `explain()` and the `featranker.interpret` module are removed (deprecated since 1.1); `featranker.__all__` is now `ConsensusSelector`, `FeatureRanker`, `Report`, `auto_groups`, `summarize`. The report evidence block replaces them. CI now also runs the full suite on Windows. Reports now ship a JSON Schema, `featranker/report.schema.json` (see docs/api.md#report-schema).

Fixes from a pre-push review: sag/saga logistic models get dense one-hot again (sparse made saga stop early); SHAP folds only one-hot pipelines FeatRanker built (other ColumnTransformers are unsupported, as documented); the report schema types `scoring` (string) and model `importance` (object); the realdata benchmark RF arm keeps all-missing columns (`keep_empty_features`; published results unaffected, DEVIATIONS item 10). `linear_regression`/`ransac_regression` scores with categorical columns can differ from pre-1.24 dense results.

## 1.24.0 - 2026-11-20

One-hot coded categorical columns stay sparse for sparse-capable built-in linear models (LDA, QDA, GaussianNB, PLSRegression, SGDClassifier and Perceptron stay dense; SGD-based models stay dense because their intercept update differs on sparse input). Scores match dense within 1e-3. Sparse one-hot also fixes a divergence of linear_regression/ransac_regression with categorical columns (dense least squares diverged on a full one-hot block plus intercept; present from 1.9 through 1.22; 1.23 changed no package code).

SHAP now covers built-in categorical linear pipelines: one-hot SHAP values are summed back to each source column (other `ColumnTransformer`s stay unsupported).

## 1.22.0 - 2026-11-20

Clear workflow: a recipe and entry docs, and four small fixes left from the 1.21.2 review. Report schema: additive (`models[*].class`, findings' `basis` and `scoring`).

### Added

- README "10-minute recipe" (rank, check support, choose a subset, validate on untouched test rows) with real output, and a "When to use FeatRanker" section, linked from `docs/README.md`.
- Every `summarize()` finding has `basis` (`"descriptive"` or `"experimental"`; findings on shadow p-/q-values are experimental) and `scoring` (the report's scorer name, `null` if none).
- Each model entry of a held-out report records `class`; `ablate()` raises `ValueError` naming the model when it differs from the ranker's model, whatever the scikit-learn version (reports without `class` are not checked).

### Fixed

- A ranker pickled before 1.21.2 with a NumPy integer `shadow_seed` loads with a plain `int`, so its reports stay strict JSON.
- The CLI rejects `--output -` (exit status 2, before ranking) instead of writing `-.json`.
- The ablation cross-version warning points at the caller's line for `FeatureRanker.ablate()` and `featranker.ablation.ablate()`.

### Tests

- `tests/test_clear_workflow.py` and `tests/test_clear_workflow_extra.py`.

## 1.21.2 - 2026-10-26

Fixes from an independent review of 1.21.1 (a7a68aa): eight defects, two silent. Report schema: additive (`shadow_test.experimental`); NumPy integer arguments are now recorded as Python ints. Rankings are unchanged except where a model's input was being corrupted (F1).

### Fixed

- Every model is fit and scored on read-only views (buffers that scikit-learn's `check_array` cannot flip back to writable, also in `model_input="dataframe"` and in the SHAP evidence) of the shared feature, target and weight arrays (FeatRanker copies nothing, but some scikit-learn linear models such as `Ridge` copy a read-only `X` once more while fitting: about +17% peak memory end to end on 1e5 x 200), so one model writing into its input (reproduced with `LinearRegression(copy_X=False)` listed before `Ridge`) no longer changes every later model's fit. An estimator that writes raises and is recorded as a `fit` failure for that model alone; the caller's arrays keep their flags (F1).
- `ConsensusSelector` with `min_shadow_share`, `max_shadow_q` or `min_shadow_families` raises a `ValueError` naming the rule when no model kept the shadow columns in any split, instead of a `KeyError` (F2).
- `ablate()` compares each voting model's parameters with the report's `models[m]["params"]` and raises `ValueError` naming the model on a difference; reports without recorded params skip the check, and a report made under another scikit-learn version is compared on the parameters both versions have and only warns when they differ (F3).
- `cv` with two identical (train, eval) splits raises `ValueError` naming both positions; a repeated split shrank q-values as if it were an independent trial (F4).
- NumPy integer `random_state`, `shadow_seed`, `n_repeats`, `top_k`, `cv` and the other integer arguments are recorded as Python ints, so reports stay strict JSON (`json.dumps(report, allow_nan=False)`). `n_repeats`, `top_k` and `shadow_seed` now accept NumPy integers (they were rejected); booleans are still rejected (F7).
- The CLI checks before ranking that `--output` has a file name, is not a directory (after the `.json` extension is applied) and that its folder exists, and reports write errors as a usage error (exit status 2), not a traceback (F8).

### Changed

- **Shadow evidence**: a model with a `FunctionTransformer` that has a function, or a transformer (has `transform`, no `predict`) not defined by scikit-learn, anywhere in it (any `Pipeline` step, or inside `TransformedTargetRegressor`'s regressor, `BaggingRegressor`, `GridSearchCV` and other meta-estimators; a `TransformedTargetRegressor`'s own target transformer sees y only and does not count), is unverified for shadows: a `shadows` failure ("could not be verified"), no shadow evidence, ranking unchanged. scikit-learn's own transformers and final predictors of any library are still assumed to use every column (F5-a).
- `ConsensusSelector(max_shadow_q=...)` emits a `UserWarning` that the rule is experimental, and resampled reports' `shadow_test` has `"experimental": true` (B-a).

### Tests

- `tests/test_review_fixes.py` covers each fix. Two tests changed: the resampled `shadow_test` block now includes `experimental`, and the empty-q warning test picks its warning by text because the experimental warning comes first.
- The calibration harness test stub carries `splits`, and `testpaths` is `["tests", "benchmarks"]`, so `python -m pytest` collects `benchmarks/` (F6). Two benchmark tests tolerate newer SciPy/NumPy (`ZeroDivisionError` for an empty cell, `np.True_`).

## 1.21.1 - 2026-10-25

Fixes from Signal's pre-push review of 1.17-1.21, which also found two failures on newer dependencies (scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1). Report schema: unchanged except the resampled `constant_within_groups` finding's message and evidence.

### Fixed

- Shadow check: once a `Pipeline` step's column mapping is unknown, any later step that is or contains a selector, `FeatureUnion` or `ColumnTransformer` makes the model unverified (a `shadows` failure); so does a meta-estimator (`TransformedTargetRegressor`, `BaggingRegressor`, ...) wrapping a selector, `FeatureUnion` or `ColumnTransformer`, wherever it sits. A `SelectKBest` after a passthrough `ColumnTransformer` was certified as keeping every shadow.
- Scorer sharing applies only to scorers whose class is one of scikit-learn's own (exact type). A subclass overriding `__call__` was served by the multi-metric path, which skips `__call__`, so cached scores could differ from uncached ones.
- `ablate()` compares the ranker's training rows with the report's in both directions (no `train_rows_sha256` means every row; no version before 1.21 could subsample training rows), so a full-data report, old or new, ablated after a `fit(max_samples=...)` refit now raises `ValueError`.
- A ranker pickled before fingerprint scheme 2 hashes shuffle labels with scheme 1 in `rank_features()`, like the rest of its run block.
- `_spearman_matrix` buffers take the dtype `rankdata` returns, so float32 input gives the same values as 1.20 on SciPy versions that keep float32 ranks (1.18).
- `summarize()` of a resampled report names the splits in which each `constant_within_groups` group was constant (evidence `splits`) and says its importance was zero by construction in those splits only.
- Docs, docstrings and the memory-limits spec: an unseen level joins the infrequent column when the column has at least `max_categories` levels (scikit-learn builds the bucket at equality too), not only with more.

### Tests

- A test used the scorer name `"max_error"`, removed in scikit-learn 1.9; it now uses `make_scorer(max_error, greater_is_better=False)`.
- The installed-wheel smoke test requires every ablation row to have scored models and a finite `paired_median_score`, and prints the ablation failures otherwise.

## 1.21.0 - 2026-10-24

Memory limits, from the memory study (`spikes/memory/FINDINGS.md`, with an "After 1.21" section). Report schema: additive (`run.n_train_rows` only with a training subsample, `run.max_categories` only when set, `max_train_samples` on resampled reports). Without the new options, reports equal 1.20's byte for byte.

### Added

- `FeatureRanker(max_categories=None)`: `None` or an int of at least 2 (else `ValueError`). Built-in linear models one-hot encode at most that many levels per categorical column; the rest, and levels unseen in training, share one infrequent column (`OneHotEncoder(max_categories=..., handle_unknown="infrequent_if_exist")`, set only when the option is). Tree and custom models are unaffected. 10 categorical columns of 1000 levels on 4000 rows: peak RSS 3.1 GB to 0.37 GB. `ConsensusSelector(max_categories=)` and CLI `--max-categories` pass it through; `ablate()` refits with the same cap.
- `fit(..., max_samples=None, random_state=None)`: fit the models on a random subsample of the training rows (an int or a float in (0, 1], validated like `rank_features()`'s). The draw uses its own stream (`default_rng([random_state, 3])`; `None` is fresh entropy), keeps row order, and draws whole groups under `shuffle="within_groups"`. All-zero sampled `sample_weight` raises `ValueError`. Shadows and the evidence copy come from the sample. `run["train_data_sha256"]` and the weight and label fingerprints still cover every row passed, so `ablate()` accepts the report, and it refits on the same sampled rows. `rank_resampled(max_train_samples=)` (split seed `random_state + i`), `ConsensusSelector(max_train_samples=)` and CLI `--max-train-samples` pass it through.
- Report: `run["n_train_rows"]` (with a training subsample), `run["max_categories"]` (when set; also in a resampled report's `run`), top-level `max_train_samples` on resampled reports.
- `spikes/memory/run_after.py`, `results_after_1_21.jsonl`: the before/after measurements.

### Changed

- The shadow block is filled into one preallocated array instead of gather plus `hstack`: the peak while building it is 2X of the matrix, not 3X (100k x 500: 2.5 to 2.1 GB). Values, random stream and dtype are unchanged.
- The Spearman matrix behind `evidence=True` and `auto_groups()` ranks, centres and squares in column blocks into one array: the step adds about 1X of the data to the peak, not about 4X (`auto_groups` on 100k x 500: 4.1 to 2.1 GB). Values are bit for bit those of 1.20.
- `fit()` encodes the target before drawing a training subsample, so a class the sample misses is still known to the evaluation encoder.

### Documentation

- `docs/usage.md` "Large data": what `max_samples` (evaluation and fit), `max_categories`, `n_repeats` and `feature_groups` each bound, with the measured numbers; `docs/report.md`, README.
- Roadmap line 1.21 and limitations L11 and L12 in `openspec/config.yaml`.

## 1.20.0 - 2026-10-24

Installed-wheel smoke test in CI and release. Library code and report schema: unchanged.

### Added

- `tests/smoke_installed.py` (`--extras` for xgboost, catboost, SHAP and plotting): checks an installed copy of the package, not the checkout. Version matches metadata, registry YAML and `py.typed` are packaged, a small fit/rank/summarize/ablate run and a `ConsensusSelector` pipeline work, and the `featranker` console script writes a report for two CSVs. It refuses to run against the source tree.
- `tests.yml` `wheel` job: builds the wheel, installs it in a fresh virtualenv outside the checkout and runs the smoke script, core-only and with `[all,shap,plot]`, on the oldest (`>=1.5,<1.6`) and newest scikit-learn.
- `release.yml` runs the smoke script (core, then extras) on the built wheel before the artifact is uploaded.

### Documentation

- `docs/api.md` "Releasing" describes the smoke step.

## 1.19.0 - 2026-10-24

Shared predictions across scorers. Report schema: unchanged; scores, importances, failures and reports equal 1.18's for the cases below.

### Changed

- With several scorers, the baseline and every permuted copy are predicted once per response method (`predict`, `predict_proba`, `decision_function`) for scorers that scikit-learn can cache and that agree on the positive label, through its multi-metric scorer; plain callables are called as before, and so are scorers with a different `pos_label` or, with `sample_weight`, whose function takes no weights. Each model's baseline is also scored per scorer, and sharing is used for that model only if the shared scores equal the direct ones exactly. A failing scorer is re-run directly, so its exception and failure entry are unchanged. If the private `sklearn.metrics._scorer._MultimetricScorer` is unavailable or raises, scoring falls back to per-scorer calls.

## 1.18.0 - 2026-10-24

Statistical contracts from an external review. Report schema: additive (`constant_within_groups` only with `shuffle="within_groups"`).

### Added

- `FeatureRanker(families={model_name: family})` and `ConsensusSelector(families=...)`: assign the consensus family of custom or built-in models, so near-identical custom models cast one vote. Invalid values raise `ValueError` at construction; a name that is not a configured model raises `ValueError` in `fit()`, `rank_resampled()` and `ConsensusSelector.fit()` (before any split); any mapping is accepted. Old pickles load with `None`.
- `rank_features()` with `shuffle="within_groups"` records `constant_within_groups` (feature groups constant within every shuffle group on the scored rows), emits a `UserWarning` when non-empty, and `summarize()` adds a `constant_within_groups` warning. Resampled reports carry the union of their split lists as top-level `constant_within_groups`.

### Documentation

- Within-group shuffling measures reliance on within-group variation only; patient-level predictors get zero importance by construction.
- With sample weights, shuffle donors come from all evaluation rows (zero weight included); suits cost weights, not survey or frequency weights.
- Ablation: a data set used to choose k gives a validation score; final performance needs another test set or outer cross-validation. The example now runs on its own, with its real output.
- Custom models are separate families unless `families=` maps them.

## 1.17.0 - 2026-10-23

Trust fixes from an external review. Report schema: additive (`run.fingerprint_scheme`; new `ablation` keys); fingerprint values change for integer and boolean data.

### Fixed

- Shadow-drop check: it now follows a `ColumnTransformer` into each branch (and a `Pipeline` into its first step), mapping the branch columns back to the model input, so an inner transformer that drops shadow columns gives a `shadows` failure. It also follows `FeatureUnion` parts and fitted feature selectors (`SelectKBest`, ...); a later `ColumnTransformer` in a `Pipeline` is checked only after steps that keep the columns one for one (scalers, a plain `SimpleImputer`) and is otherwise unverified, as is a meta-estimator wrapping a `ColumnTransformer`. A `FunctionTransformer` or custom transformer that drops columns is still assumed to use all of them. A model whose used columns cannot be worked out (scikit-learn private attributes missing) now also gets a `shadows` failure, with a message saying the shadows could not be verified, instead of a `UserWarning`; such a model has no shadow evidence.
- Fingerprints (scheme 2): integer and boolean columns are hashed exactly, tagged with the dtype kind (not the size: `int32` equals `int64`, booleans hash as 0/1; an integer column and a float column with equal values still differ), where scheme 1 converted them to float64 and could not tell integers above 2**53 apart (swapped labels `2**53` and `2**53+1` gave equal fingerprints). Float and text columns hash as before. Every `run` block records `fingerprint_scheme: 2`. Fingerprints of integer or boolean data (class labels, for example) from 1.12-1.16 no longer equal 1.17 ones, so reports cannot be compared by hash across that boundary. `ablate()` checks a report without `fingerprint_scheme` with scheme 1, and a ranker pickled before 1.17 keeps fingerprinting its `rank_features()` runs with scheme 1 (`fingerprint_scheme_`).
- `ablate()` records its inputs and its cohort: the block gains `fingerprint_scheme`, `data_sha256`, `weights_sha256`, `n_eval_rows`, `same_holdout`, and `paired_models`; each curve row gains `n_models_scored` and `paired_median_score` (the median over the models scored at every k, `null` when there are none). `median_score` is unchanged.
- `summarize()` picks the best `k` on `paired_median_score` when there are paired models (else on `median_score`, and says so) and adds an `ablation_cohort_changed` warning when the set of scored models differs between rows (evidence rows list `models`). A block from 1.16 is described as predating the paired curve.
- `rank_features(max_samples=..., sample_weight=...)` raises `ValueError` before scoring when the sampled rows all have zero weight (it failed with `ZeroDivisionError` in every scorer). In `rank_resampled()` that split fails.

### Changed

- `scikit-learn>=1.5` is required (the tested floor; the metadata-routing and imputation code needs a recent release).

### Documentation

- Usage guide: the shadow contract and named-input limitations, ablation paired median and provenance, comparing runs (`fingerprint_scheme`), `max_samples` with weights; the report reference describes the new keys.

## 1.16.0 - 2026-10-23

Ablation curve: check a ranking by refitting on the top-k groups. Reports without `ablate()` are unchanged.

### Added

- `FeatureRanker.ablate(X, y, X_eval, y_eval, report, *, ks=None, scoring=None, sample_weight=None, eval_sample_weight=None)` returns a new `Report`: a copy of the held-out `report` plus an `ablation` block. For each `k`, every voting model (`in_consensus: true`) is refit on the columns of the top-k consensus groups, with the ranker's `missing`, `categorical`, `model_input`, `n_jobs`, and the given weights, and scored on `X_eval`. The block has `scoring`, `models`, `baseline_score` (the report's, `null` when `scoring` names another scorer), `n_groups`, `curve` (`k`, `feature_groups`, `n_features`, `scores`, `median_score` per row), and `failures` (`model`, `stage`, `message`, `k`; a failed model scores `null` there). Report schema: additive.
- `ks` defaults to 1, 2, 3, 5, 10, 20, 50, 100, ... up to the number of consensus groups, plus that number; other values must be positive ints at most that number. `ValueError` for an unfitted ranker, a resampled report, an empty consensus, bad `ks`, or training data that differs from the report's `train_data_sha256`.
- `summarize()` adds an `ablation_curve` info finding (best `k`, and the score at the largest `k`) when the block has a non-null `median_score`.
- CLI: `--ablate`, `--ablate-ks 1,2,5` (needs `--ablate`), and `--drop a,b`, which removes columns from both CSVs before features are read (an unknown name, the target, the `--sample-weight` column, or a `--categorical` column is an error).

### Documentation

- Usage guide: Ablation curve (with the caveat that scoring on the ranking's own evaluation data is optimistic), `.ablate()` parameters, CLI flags, the new finding; the report reference describes the `ablation` block.

## 1.15.0 - 2026-10-22

Dependent shuffling: permutations and shadow columns can shuffle within groups or by a cyclic shift. The default is unchanged.

### Added

- `FeatureRanker(shuffle=...)` and `ConsensusSelector(shuffle=...)`: `"rows"` (default, unchanged), `"within_groups"` (values move only among rows with the same label), or `"circular"` (rows shifted cyclically by a random offset from 1 to n-1; assumes time order). Any other value raises `ValueError`. Applies to permutation importance and shadow columns; correlation and SHAP evidence are unchanged.
- `shuffle_groups=` on `fit()` (shadows) and `rank_features()` (permutations), one label per row, required with `"within_groups"` and an error with another mode. `rank_resampled(groups=)` and `ConsensusSelector.fit(groups=)` supply the labels, sliced per split. A group of one row keeps its values; if every group has one row, `ValueError`.
- Reports gain top-level `shuffle` when it is not `"rows"`. Report schema: additive.

### Documentation

- Usage guide: Shuffling dependent rows; the time-series and grouped-data recipe uses the new modes; the report reference and README describe `shuffle`.

## 1.14.0 - 2026-10-22

Several scorers: check a ranking under more than one metric in one run.

### Added

- `scoring` in `rank_features()`, `rank_resampled()`, and `ConsensusSelector` accepts a list or tuple of scorer names or callables (at least one, unique names; an empty list, a repeated name, or an invalid entry raises before any model is ranked). The first is primary: the dummy gate, consensus, shadow evidence, stability, and selection use it alone, so a one-element list or a plain name gives the same report as before.
- Secondary scorers are computed on the same permuted copies. Each model report gains `secondary`: per secondary scorer, `evaluation_score` and `importance` (`mean`, `std` per group). The report gains `secondary_scoring` and `secondary_baseline_scores` (the dummy's score per secondary scorer, or `null`); resampled reports carry `secondary_scoring` at the top and `secondary` in each split. These keys appear only when more than one scorer is given. Report schema: additive.
- A secondary scorer that raises or returns a non-finite value for a model leaves that model ranked, sets its `secondary[name]` to `null`, and adds a `failures` entry at stage `"evidence"`.
- CLI `--scoring` accepts comma-separated names (`--scoring accuracy,neg_log_loss`).

### Documentation

- Usage guide: Several metrics; the report reference and README describe the new keys.

## 1.13.0 - 2026-10-22

Evaluation subsampling: score on a random share of the evaluation rows to save time. The default is unchanged.

### Added

- `rank_features(..., max_samples=None)` and `rank_resampled(..., max_samples=None)`: a positive int (rows; more than available uses all) or a float in (0, 1] (`max(1, floor(share * rows))`); anything else raises `ValueError` before scoring. One subsample, drawn without replacement with `random_state`, is used for the dummy gate, every baseline, and every permuted score (sample weights follow their rows). Training scores and evidence are unchanged. Reports gain `max_samples` and `n_eval_rows` only when it is given (resampled reports: `max_samples` at the top, `n_eval_rows` in each split's report). Without it, reports are identical to 1.12.
- `ConsensusSelector(max_samples=None)` and CLI `--max-samples` (an int, or a float with a decimal point). Selectors pickled before 1.13 load with `None`.

### Documentation

- Usage guide: Fewer evaluation rows (`max_samples`) under Speed; the report reference and README list the new keys.

## 1.12.0 - 2026-10-22

Repeated use: reports say what they were computed from, the `auto_groups` threshold is an option, `ConsensusSelector` warns when splits did not vote, and `summarize()` reads resampled reports.

### Added

- Every report from `rank_features()` and `rank_resampled()` has `run`: `task`, `group`, and `data_sha256` (a fingerprint of the evaluation data, or of all the data for a resampled report). `rank_features()` reports add `train_data_sha256` (the data passed to `fit()`; `null` for a ranker pickled before 1.12); `rank_resampled()` reports add `cv` (the integer given, or the splitter's `repr`) and `splits_sha256` (a fingerprint of every split's train and evaluation indices). Each split report has its own `run`. A fingerprint is the SHA-256 of the feature names, feature values, and target values; equal inputs under the same library versions give equal fingerprints, and any changed value, name, or row order changes it. Report schema: additive.
- `ConsensusSelector(auto_groups_threshold=0.7)` and CLI `--auto-groups-threshold` (default 0.7) set the cut of automatic grouping. When groups were proposed automatically, `run["auto_groups"]` is `{"threshold": t, "method": "spearman-complete-linkage"}`; otherwise the key is absent. Selectors pickled before 1.12 load with 0.7.
- `ConsensusSelector.fit` warns how many of the splits did not vote (failed, or produced no consensus). Those splits already counted as misses in the shares; selection is unchanged.
- `summarize()` accepts resampled reports and returns `split_failures`, `failures` (aggregated over splits), `gate_off`, `empty_consensus`, `below_shadow`, and `insufficient_splits` findings, in that order. `Report.summary()` works on resampled reports.

### Changed

- `summarize(resampled_report)` no longer raises `ValueError`. `to_prompt()` and `explain()` (deprecated) still show no findings for resampled reports.

### Documentation

- Usage guide: Comparing runs, Summarizing a resampled report, and the threshold option; the report reference lists `run`; README updated.

## 1.11.0 - 2026-10-21

Sample weights: fit, baseline, and every score can use the weights your own model uses. The default is unchanged.

### Added

- `FeatureRanker.fit(..., sample_weight=None)`, `rank_features(..., sample_weight=None)` (evaluation weights), and `rank_resampled(..., sample_weight=None)` (sliced per split for training and evaluation). A weight vector is a 1-D array-like of finite, non-negative numbers, one per row, with a positive sum; anything else raises `ValueError` before any model is fit. Weights are copied. Without weights, behaviour and reports are byte-identical to 1.10.
- Weighted fitting: every model is fit with the training weights, routed to the final step of built-in pipelines (imputer, scaler, and one-hot encoder stay unweighted); the dummy baseline is fit with them. Weighted scoring: with evaluation weights the baseline, every permuted, and the dummy's score are weighted; with `evidence=True` and training weights, the train score uses them.
- A model whose final estimator has no `sample_weight` in `fit()` (built-in `lda`, `qda`, `pls_regression`; the same check applies to custom models and the last step of their pipelines) is not fit unweighted: `failures` gets `{"stage": "fit", "error_type": "SampleWeightNotSupported"}`.
- Reports gain `sample_weight: {"train": bool, "eval": bool}` when any weights are used, also in each resampled split report. Report schema: additive.
- `ConsensusSelector.fit(X, y, groups=None, sample_weight=None)` passes the weights to its ranking; in a `Pipeline` it works with scikit-learn metadata routing (`set_fit_request(sample_weight=True)`). The CLI takes `--sample-weight COLUMN`, read from both CSV files and left out of the features.

### Limitations

- Correlation evidence and the SHAP summary stay unweighted. Custom scorers must accept a `sample_weight=` keyword when weights are given. A split whose evaluation rows all have weight 0 fails as a `"split"` failure.

### Documentation

- Usage guide: a Sample weights section with a runnable example, the `sample_weight` parameter rows, and the CLI flag; the report reference lists `sample_weight`; README updated.

## 1.10.0 - 2026-10-21

Named model input: custom models can select columns by name, and shadow evidence is protected against pipelines that drop the shadow columns. The default is unchanged.

### Added

- `FeatureRanker(model_input="array")`: `"array"` (default) keeps 1.x behaviour, reports unchanged. `"dataframe"` fits, scores (baseline, permuted, train), permutes, and explains (SHAP) every model on a pandas DataFrame whose columns are the feature names, then `shadow__<name>` for each shadow column; the dummy baseline still gets arrays. Importances, scores, and the SHAP summary equal those of array mode for models that do not depend on column names. Any other value raises `ValueError`; `"dataframe"` without pandas raises `ImportError` at construction. Pickles from before 1.10 load with `"array"`.
- Shadow-compatibility check (both modes, shadows on): a model that is or contains (in a `Pipeline`, nested too) a fitted `ColumnTransformer` that sees all input columns but whose transformers and `remainder` leave a shadow column unused is left out of every shadow comparison in that report (no `above_shadow`, `shadow_max`, `shadow_importance`, and not counted in `family_above_shadow` or `above_shadow_share`). `failures` gets `{"stage": "shadows", "error_type": "ShadowColumnsDropped"}` for it; its importances and ranking vote are unchanged. A selector that cannot be resolved gives a `UserWarning` and no failure. Only `ColumnTransformer` is inspected.
- `ConsensusSelector(model_input="array")` stores it unchanged and passes it to its ranker; `rank_resampled()` uses it in every split. Reports gain `model_input: "dataframe"` (top level and in each split report) only in that mode. `summarize()` treats `"shadows"` failures as warnings. The CLI keeps array input. Report schema: additive.

### Limitations

- Built-in estimators fit on DataFrames store `feature_names_in_`; the ranker never mixes modes. A DataFrame is rebuilt from the permuted array for each scoring call, which costs time on wide data. Other column-dropping estimators (feature selectors, custom steps) are not inspected.

### Documentation

- Usage guide: a Columns by name section with a runnable example, the `model_input` parameter rows, and the shadow contract; the report reference lists `model_input` and the `"shadows"` stage. README updated.

## 1.9.0 - 2026-10-21

Categorical features: rank raw text and category columns as single features, with the encoding learned inside each fit. The default is unchanged.

### Added

- `FeatureRanker(categorical=None)`: `None` keeps 1.x behaviour (non-numeric columns raise as before, reports byte-identical to 1.8). `"auto"` treats DataFrame columns of object, string, or category dtype as categorical (NumPy input has no dtypes, so none); a list of feature names marks those columns whatever their dtype, NumPy object arrays included. Any other value raises `ValueError` at construction, an unknown name at `fit()`. Pickles from 1.0-1.8 load with `categorical=None`.
- Fold-local coding: in each `fit()`, a categorical column's levels are the sorted string forms of its training values, coded `0..k-1`, with a missing value (`None`/NaN) as its own last level; values not in training share one "unseen" code `k` in `rank_features()` and in the shadow block. The ranker works on the code matrix, so permutation, shadows, stability, and the selector work by column index and each raw column stays one feature. The fitted ranker exposes `categorical_features_` and `categorical_levels_`. `rank_resampled()` and `ConsensusSelector` validate raw input and slice it per split, so every split learns its own levels.
- Built-in linear models (when categorical columns exist) are `ColumnTransformer` pipelines: `OneHotEncoder(handle_unknown="ignore")` on the categorical code columns (shadow columns included) and the usual scaler, after the median imputer under `missing="impute"`, on the numeric columns; the output is dense. Built-in trees use the codes; custom models receive the code matrix. Without categorical columns the models are built exactly as in 1.8.
- Reports gain `categorical` (`columns`, `train_levels` with the missing level counted, `unseen_eval` as a count of evaluation values), also in each resampled split report. Report schema: additive, and only when categorical columns exist.
- Evidence: categorical columns are left out of Spearman correlations (`evidence.correlation`) and of `auto_groups()`, which gains `categorical=None`. The SHAP summary explains trees on the codes (a categorical group's `direction` is `null`) and lists linear pipelines with a one-hot step as unsupported ("pipeline one-hot encodes categorical columns").
- `ConsensusSelector(categorical=None)` stores it unchanged, passes it to its ranker and to `auto_groups()`, selects raw columns, and allows NaN/None in `transform()` when it is set (the `allow_nan` tag). Pickles from before 1.9 load with `categorical=None`. CLI: `--categorical auto|name,name` reads those CSV columns as text (`auto`: columns of the training CSV with a non-numeric cell); an empty cell there is the missing level; levels come from the training CSV only.

### Changed

- Date-valued data now raises `ValueError` ("X must be numeric") under every policy instead of becoming nanosecond counts: DataFrame columns that hold dates, timestamps, or timedeltas as objects or as category dtype (the 1.7 check covered only datetime64/timedelta64 dtypes), such columns listed in `categorical`, and NumPy `datetime64`/`timedelta64` arrays.

### Limitations

- Levels are compared by string form, so `1` and `"1"` collide. Trees see the codes as ordinal numbers; one-hot or native categorical support needs custom models. A high-cardinality column makes the linear pipelines wide (`train_levels` shows how wide). No target or other supervised encoding. Custom models still receive a plain float array, not column names (next change).

### Documentation

- Usage guide: a Categorical features section with a runnable example; the parameter tables, CLI section, evidence notes, and the report reference list the new option and the `categorical` key. README updated.

## 1.8.0 - 2026-10-21

Stability `above_shadow_share` now divides by `n_splits`, like the family shares and `shadow_hits`. Shadows have never been released, so the meaning is corrected before 1.0.

### Changed

- In `rank_resampled()` reports with shadows on, a stability row's `above_shadow_share` is the sum of the group's per-split consensus shares divided by `n_splits`. Failed splits, splits with an empty consensus, and splits without a consensus row for the group count as 0. Before, it averaged over the splits that had a row, so one lucky split of ten gave a noise group 1.0 (the calibration study measured `min_shadow_share=0.5` selecting 11 % of pure-noise groups in regression `group="fast"`). When every split has a consensus the value is unchanged; reports without shadows and the single-split consensus `above_shadow_share` are unchanged. `ConsensusSelector(min_shadow_share=...)` is stricter where many splits lack a consensus. Report schema: no new keys.

### Documentation

- The report reference and usage guide state the new definition. The benchmark and calibration write-ups note that their tables use the earlier definition and were not re-run.

## 1.7.0 - 2026-10-21

Missing values: rank features on data with NaN without imputing by hand, keeping the imputation inside each training split. The default is unchanged.

### Added

- `FeatureRanker(missing="error")` gains `missing="impute"`. NaN is then accepted in the features of `fit()`, `rank_features()` and `rank_resampled()`; infinite values are still rejected, and any other `missing` value raises `ValueError` at construction. With the default `"error"` NaN raises the same error as before and reports are byte-identical to 1.6.
- With `"impute"`, every built-in model (classification and regression, every group) is a pipeline that starts with `SimpleImputer(strategy="median", keep_empty_features=True)`, so the imputation is learned on each training split only; this includes models that handle NaN natively, for one rule across models and scikit-learn versions. A column missing in every training row is a constant 0 in training and keeps its position; its importance is meaningless. Custom `models=` receive NaN unchanged and must handle it; the dummy baseline is unaffected.
- With `"impute"`, reports gain `missing`: `policy`, `strategy`, and `train_counts` (missing training values per feature). Resampled reports carry it in each split report. Report schema: additive only, and only with `"impute"`.
- Evidence with NaN: Spearman correlations (`evidence.correlation` and `auto_groups()`) use the complete rows of each pair that involves a column with NaN (pairs of complete columns are computed as without NaN), a pair with fewer than 30 such rows counts as uncorrelated (with only a few shared rows, independent columns look strongly correlated by chance), and complete data takes the unchanged fast path with identical numbers. The SHAP summary unwraps a leading `SimpleImputer` (with `add_indicator=False`) as a per-column step, and refuses a pipeline whose steps change the number of columns. `auto_groups()` now accepts NaN with no argument (infinite values are still rejected).
- `ConsensusSelector(missing="error")` passes the policy to its ranker; with `"impute"` it also accepts NaN in `transform()` (the `allow_nan` tag follows the policy), so it works as a Pipeline step before an imputer. CLI: `--missing {error,impute}`; with `impute`, empty feature cells in the CSV files are missing values, and the default error for an empty or `nan` cell points to the option. pandas nullable dtypes (`Float64`, `Int64` with `pd.NA`) are read as NaN from DataFrames: rejected under `"error"` with the usual finite-values error, accepted under `"impute"`. Pickles from 1.0-1.6 load with `missing="error"`.

### Changed

- DataFrames with datetime, timezone-aware datetime, or timedelta columns raise `ValueError` ("X must be numeric") under both policies, instead of the `TypeError` of mixed frames (or, for an all-datetime frame, silently ranking nanosecond counts).

### Limitations

- Median imputation leaves a spike at the median that tree and kernel models can learn, so a feature's importance can mix its values with its missingness (probe with only missingness driving the target: trees about 1.9, SVR about 1.05, ridge 0.006). Use custom models with `add_indicator=True` or native NaN handling to separate them, and check `train_counts`. Native NaN handling of built-in models is not used.
- The CLI reads only empty cells and `nan` as missing; `NA`, `N/A` and `null` are rejected as non-numeric. Missing target values and categorical columns are not supported.

### Documentation

- Usage guide: a Missing values section (fold-local median imputation, custom models see NaN, pairwise correlations and their cost, the informative-missingness limitation) with a runnable example; the report reference lists `missing`.

## 1.6.0 - 2026-10-20

Family evidence: which model families find a group, an opt-in rule that uses it, and a shadow mode option. Defaults and existing rules are unchanged.

### Added

- Per-family shadow evidence. With shadows on, every consensus row gains `family_above_shadow` (family to the fraction of its voting members above that model's strongest shadow; `above_shadow_share` is their mean, unchanged), and every resampled stability row gains `family_above_shadow_share` (family to the sum of its per-split fractions divided by `n_splits`: splits where it did not vote, failed, or were empty count as 0, like `shadow_hits`, so a family that passes the dummy baseline in a few splits cannot reach a majority) and `shadow_families` (the number of families with a share strictly above 0.5). Family keys are sorted.
- `ConsensusSelector(min_shadow_families=None)`: keeps groups with `shadow_families >= min_shadow_families` (a positive int). It turns shadows on, combines with the other rules (filters apply, then `k`), and the no-rule `ValueError` now names all four rules. The docs present 2 as a heuristic starting point, because with 1 a single family fitting noise can pass; false-selection rates are not measured yet (planned: `calibration-study`).
- `FeatureRanker(shadow_mode="joint")` and `ConsensusSelector(shadow_mode="joint")`: `"independent"` shuffles each shadow column with its own row order, drawn column by column from one generator seeded as before, so collinear real columns do not stay collinear in the shadows. `"joint"` reproduces the earlier shadows exactly. With shadows on, reports record `shadow_mode`; an invalid mode raises `ValueError`. Which mode is better is untested.
- Report schema: additive only, and only with shadows on (`family_above_shadow`, `family_above_shadow_share`, `shadow_families`, `shadow_mode`). Reports without shadows are unchanged; with `shadow_mode="joint"` a report with shadows is identical once the new keys are removed. Pickles from 1.0-1.5 load with `shadow_mode="joint"` and `min_shadow_families=None`.

### Documentation

- Usage guide: reading family evidence with a runnable planted-interaction example (the share rule drops one interaction partner, `min_shadow_families=2` keeps both), when to use `min_shadow_families` and its overfitting-family risk, and `shadow_mode`. The report reference lists the new keys. The calibration study is now described as the next change instead of "planned for 1.6".

## 1.5.0 - 2026-10-20

Trust hardening, from a cross-cutting review of 1.1-1.4 (all unreleased, so these are fixes, not behavior changes for released users).

### Fixed

- `fit()`, `rank_features()` and `rank_resampled()` could share memory with the caller's arrays (`np.asarray` on float input), so changing `X` or `y` afterwards could change the fitted models of estimators that store their training data (for example `KNeighborsRegressor`, `KernelRidge`), `train_score`, `overfit_gap`, the correlation evidence, and the SHAP background. Validation now always copies `X` and `y`, so the ranker works on its own data (one extra copy per call).
- `ConsensusSelector` objects pickled before `max_shadow_q` existed (1.3) failed on `get_params()`, `clone()`, and refit. They now load with `max_shadow_q=None`.
- `group="fast"` treated any truthy `fast` value in the registry as fast (for example the string `"false"`); only the boolean `true` marks a model fast.
- SHAP of a `Pipeline` with a `"passthrough"` (or `None`) step is now computed, treating the step as the identity, instead of listing the model as unsupported with a confusing reason.

### Changed

- The shadow hits test is labelled uncalibrated. `shadow_test` gains `calibrated` (always `false`) and `note`; the docs call `shadow_p_value` and `shadow_q_value` binomial reference values (the 0.5 null and false-discovery-rate control are not established for arbitrary estimators or overlapping splits) and `max_shadow_q` a heuristic filter. The earlier "conservative null" claim is removed. Report schema: additive only (the two `shadow_test` keys, only with shadows in a resampled run).
- CLI `--group` help says what `linear` contains (linear, discriminant, naive Bayes and kernel models).

### Documentation

- SHAP `mean_abs` is on the explainer's output scale, which can be a link scale (for example log units for `PoissonRegressor`); linear `direction` is the coefficient's sign when it is not zero and `null` when it is (for example a `Lasso` that dropped the column). Worked example added.
- The overfit gap can come from a train/evaluation shift or a small evaluation sample, not only memorization.
- `group="fast"` gives every family a representative, but a representative can fail or lose to the dummy baseline, so the voting families can differ from `"all"`.
- Benchmark reproduction installs `.[all]` with the recorded package versions; `SelectByShuffling` uses its own 3-fold cross-validation.
- New recipe: time series and grouped data (`TimeSeriesSplit`, `GroupKFold`) with `rank_resampled()` and `ConsensusSelector`.

### Tests

- The fast-preset scaler test covers every fast linear model in both tasks; a CLI test covers an unknown `--group`.

## 1.4.0 - 2026-10-19

### Added

- Shadow hits test: `rank_resampled()` with shadows adds to every stability row `shadow_hits` (splits in which the group's consensus `above_shadow_share` is above 0.5; a split without a consensus row is a non-hit), `shadow_p_value` (one-sided binomial test of the hits against probability 0.5 over all `n_splits`), and `shadow_q_value` (Benjamini-Hochberg across groups). No extra model fits. The p-values are approximate because splits overlap, and the null is conservative; see the usage guide.
- The resampled report gains `shadow_test` (`method`, `hit`, `null_probability`, `correction`, `n_trials`, `min_p_value`). `min_p_value` is `0.5 ** n_splits`, so users can see when too few splits (for example the default `cv=5`, 0.031) leave the test no power; 10 or more splits are recommended.
- `ConsensusSelector(max_shadow_q=None)`: keeps groups with `shadow_q_value <= max_shadow_q` (a number in (0, 1]). It turns shadows on, applies before `k`, and combines with `min_shadow_share`. With no rule set, the `ValueError` now names `k`, `min_shadow_share`, and `max_shadow_q`.
- When `max_shadow_q` selects nothing, the empty-selection warning also gives the number of splits and the smallest possible p-value. The resampled `shadow_test` block is included in the data view of `to_prompt()`/`explain()` when present (prompts for reports without it are unchanged).
- Report schema: additive only, and only when shadows are on in a resampled run (three stability-row keys and one top-level key). Reports without shadows, and single-split reports, are unchanged.
- Uses `scipy.stats.binom` (scipy is already installed with scikit-learn).

## 1.3.0 - 2026-10-19

### Added

- `auto_groups(X, *, threshold=0.7, feature_names=None)`: proposes `feature_groups` from `X` alone (no target). Columns are clustered by Spearman correlation with complete linkage on `1 - |rho|`, cut so every pair inside a group has `|rho| >= threshold`; a constant column is never grouped. Returns only groups of two or more columns, named by their members joined with `+`, ready to pass as `feature_groups`. The cut allows a floating-point tolerance of `1e-12` so exact copies group at `threshold=1`. A bad `threshold`, or a group name that equals a column name or another group's name, raises `ValueError`.
- `ConsensusSelector(feature_groups="auto")` computes the groups with `auto_groups` on the `X` passed to `fit()`, so inside a pipeline they come from the training fold only; `report_["feature_groups"]` shows them. `get_params()` keeps `"auto"`; any other string raises `ValueError` in `fit()`.
- CLI: `--feature-groups auto` computes the groups on the training CSV. Any other value is still a YAML path (a file named `auto` needs `./auto`).
- `"auto"` is not offered on `FeatureRanker.rank_features()` or `rank_resampled()`: call `auto_groups()` on your training data and pass the dict.
- Report schema unchanged; auto groups appear in the existing `feature_groups` key.

## 1.2.0 - 2026-10-19

### Added

- `group="fast"`: one built-in representative per model family, drawn from both the linear and tree groups, so every family still casts its consensus vote with about a third of the models. Classification fits 7 models (22 in `"all"`): `logistic_regression`, `ridge_classifier`, `lda`, `naive_bayes_gaussian`, `decision_tree`, `random_forest`, `hist_gradient_boosting`. Regression fits 5 (18 in `"all"`): `ridge_regression`, `svr_regression`, `decision_tree_regressor`, `random_forest_regressor`, `hist_gradient_boosting_regressor`. All are scikit-learn models, so no optional dependency is needed. The registry marks them with `fast: true`.
- `ConsensusSelector(group="fast")`, `rank_resampled()` on a fast ranker, and CLI `--group fast` use the same set. As for the other non-default groups, `group="fast"` cannot be combined with `models=`.
- Report schema unchanged; reports list whichever models ran.

## 1.1.0 - 2026-10-19

### Added

- Mechanism evidence: `FeatureRanker(..., evidence=True)` keeps the training data in `fit()` and adds to the report, per model, `train_score` and `overfit_gap` (`train_score - evaluation_score`); a top-level `evidence` block with `correlation` (feature-group pairs whose strongest member-column Spearman correlation on the training data is at least 0.5, shadow columns excluded); and, with the new `shap` extra, `models[*].shap` (per group `mean_abs` and `direction` on up to 1000 evenly spaced evaluation rows, for linear models via `LinearExplainer` and tree models via `TreeExplainer`) plus `evidence.shap` (`n_rows`, `unsupported`). Without `shap`, a `UserWarning` is raised and the rest is still computed. A train-scoring error is a `failures` entry at stage `"evidence"` and leaves the ranking untouched. Reports made without `evidence` are identical to 1.0 reports; pickled 1.0 rankers load with `evidence=False`; a ranker fitted with `evidence=True` also pickles its training data.
- `rank_resampled()` passes `evidence` to every split, so each split report carries its own evidence (nothing is aggregated).
- `summarize()` rules `overfit` (a model lost at least half of its training advantage over the dummy on the evaluation data) and `correlated_groups` (importance may be split between correlated groups), both info, after `below_shadow`; failures at stage `evidence` are info.
- CLI: `--evidence`.
- New extra `featranker[shap]` (not part of `[all]`); the `test` extra now installs `shap`.

### Deprecated

- `to_prompt()` and `explain()` emit a `FutureWarning` and will be removed in 2.0. They work unchanged through 1.x. The investigator study found that this evidence, not an LLM reading, carried the interpretive gains.

## 1.0.0 - 2026-10-16

First stable release. It also delivers 0.5–0.9 below, which were never published separately: family-aware consensus, `summarize()`, resampled stability, shadow features, `ConsensusSelector`, and LLM interpretation.

### Added

- `Report`: `rank_features()` and `rank_resampled()` return a `Report`, a `dict` subclass with `summary()`, `to_frame()` (pandas), and `plot()` (matplotlib, `featranker[plot]`). Existing dict use is unchanged; code checking `type(report) is dict` exactly sees the subclass.
- Typing: the package ships `py.typed`; `mypy featranker` runs in CI. `featranker.__version__`.
- Public API and stability policy (`docs/api.md`): semantic versioning, additive report keys in minor releases, deprecation warnings for at least one minor release.
- Release workflow publishing to PyPI with trusted publishing on `v*` tags.
- Package metadata: classifiers (including supported Python versions), project URLs, keywords.

### Removed

- **Breaking:** the `lightgbm` extra; no built-in model used LightGBM. `all` now installs XGBoost and CatBoost.

## 0.9.0 - 2026-10-16

### Added

- `to_prompt(report, *, anonymize=False)`: a deterministic LLM prompt with FeatRanker's interpretation rules, the `summarize()` findings, a compact JSON view of the report (no per-repeat values or params), and a JSON reply format in which every claim cites field paths. Works for single-split and resampled reports; Failure messages are never included, and custom exception class names are sent as `CustomError`. `anonymize=True` aliases feature groups (`g1…`), columns (`c1…`), custom model and family names (`m1…`, `f1…`), and custom scorer names; the reply writes names as `[[g1]]` markers, which are mapped back.
- `explain(report, llm, *, anonymize=False)`: calls your own `llm(prompt) -> str` once, parses the JSON reply, and checks every cited path (dotted or JSON Pointer) and scalar value against a snapshot of the prompt data (numbers within 0.01 or 1%; other types must match exactly), returning per-evidence and per-claim `verified` flags and `all_verified`. FeatRanker makes no network calls and needs no API key; docs include a Claude recipe using the official SDK.

## 0.8.0 - 2026-10-15

### Added

- `ConsensusSelector`: a scikit-learn feature selector (`SelectorMixin`) that runs `rank_resampled()` inside `fit()` on the data it is given and keeps feature groups by `k` (first k in the stability order), `min_shadow_share` (turns shadows on), or both. Use it as a `Pipeline` step so `cross_val_score` and `GridSearchCV` rerank on every training fold. Supports `clone`/`set_params`, DataFrame names (`feature_names_in_`, `get_feature_names_out()`), `report_` and `selected_groups_`; warns and keeps no columns when nothing passes. `fit(X, y, groups=...)` (or `select__groups` in a Pipeline) feeds group-aware inner splitters; NumPy integer `k` works in grids; DataFrame labels are compared as strings; a failed `fit()` clears fitted state; tags declare that `y` is required.

## 0.7.0 - 2026-10-15

### Added

- Shadow features: `FeatureRanker(shadows=True, shadow_seed=...)` trains and ranks every model with a shuffled copy of all features (one row order for the whole shadow block, so group structure is kept). Real groups are still ranked among themselves; each model gains `shadow_importance` and `shadow_max`, each importance entry `above_shadow`, and each consensus row `above_shadow_share` (each voting family contributes its fraction of members beating that model's strongest shadow; descriptive, not a significance test). `shadow_seed` must be a non-negative integer or `None`. Reports are unchanged with shadows off.
- `rank_resampled()` honours shadows (split `i` uses `shadow_seed + i`); stability rows gain the mean `above_shadow_share`.
- `summarize()` rule `below_shadow` (info): groups whose family-weighted above-shadow share is below 0.5 (each family contributes its fraction of voting members beating that model's strongest shadow group).
- Rankers pickled under 0.6 load with shadows off.

## 0.6.0 - 2026-10-15

### Added

- `FeatureRanker.rank_resampled(X, y, *, scoring, cv=5, ...)`: fit and rank on every cross-validation split and report rank stability. `cv` takes an int (shuffled folds, stratified for classification), any scikit-learn splitter (with `groups=`), or `(train, eval)` index pairs. The resampled report (`evaluation_mode: "resampled"`) keeps every split's normal report and adds a `stability` table per feature group, built from the group's place in each split's sorted consensus: `split_ranks`, `median_rank`, `min_rank`, `max_rank`, `n_splits_ranked`, and `top_k_share` (the share of splits where it placed within `top_k`). Splits with overlapping train and evaluation rows are rejected. Split-level failures are recorded with `stage: "split"`; the calling ranker is never modified.

### Changed

- `summarize()` raises `ValueError` with guidance when given a resampled report; summarize each split's report instead.

## 0.5.0 - 2026-10-15

### Changed

- **Consensus gives each model family one vote.** Built-in models carry a `family` (for example `logistic`, `bagging`, `boosting`); ranks are combined within a family by median, then across families. For built-in rankers, `median_rank`, `mean_rank`, `rank_std`, and the consensus order can differ from 0.4. Custom `models=` are each their own family, so their consensus is unchanged.

### Added

- Report fields: `family` per model; `n_families` and `family_ranks` per consensus row. Existing keys are unchanged.
- `summarize(report)`: deterministic findings (`gate_off`, `empty_consensus`, `failures`, `few_families`, `tied_groups`, `within_noise`) implementing the report reading guide.
- `FeatureRanker(verbose=False)` and CLI `--quiet` hide progress bars.

### Compatibility

- Rankers pickled under 0.4 load and run under 0.5: `verbose` defaults to `True` and built-in model families are rebuilt from the registry.

## 0.4.1 - 2026-10-13

### Fixed

- Built-in `sgd_classifier`, `passive_aggressive`, `perceptron`, `linear_svm`, and the `saga` logistic models were unseeded, so their scores changed between runs. They now use `random_state=42` like the other models; their scores differ from 0.4.0.

### Docs

- README restyled and slimmed; reference material moved to `docs/` (usage, report reading guide, models, troubleshooting).

## 0.4.0 - 2026-10-12

### Added

- Dummy baseline (`DummyClassifier`/`DummyRegressor`): only models that strictly beat it vote in the consensus. Reports gain `baseline_score` and per-model `in_consensus`; an empty consensus warns.
- `FeatureRanker(models={...})` to rank with your own estimators or Pipelines (cloned at `fit()`).
- `n_jobs` / `--n-jobs` to fit and rank models in parallel with results identical to sequential runs.
- Report `provenance` (featranker, Python, NumPy, scikit-learn versions) and per-model JSON-safe `params`.
- End-to-end tests of the shipped model registry; CI matrix over Python 3.10–3.13 and scikit-learn 1.5.x/latest.

### Changed

- Built-in linear models standardize features internally (`StandardScaler` fit on training data only).
- Permutation reuses one scratch matrix per model (same results, less copying).
- CLI moved to `featranker/cli.py` (console script `featranker.cli:main`); CSV flags are argparse-required and no arguments prints help.
- Every classification target is label-encoded to 0..n-1; regression targets must be numeric.
- CSV targets are typed by `--task`, so train/eval label sets stay comparable.

### Fixed

- Longer unseen eval labels were silently truncated to the training label width and accepted.
- Mixed numeric/text targets across train and eval CSVs produced all-zero scores without error.
- L1/elastic-net logistic models silently ran as L2 on scikit-learn < 1.8.
- XGBoost failed on classification labels not starting at 0.
- `pls_da` (PLSRegression) removed: it is not a classifier.
- A failed refit left the ranker marked fitted.
- NaN/inf features are rejected; CSV BOMs and blank lines are handled; invalid scorer, `n_repeats`, or feature groups fail before any model is fit.
- Build requires setuptools>=77 for the SPDX license field.

### Removed

- Legacy preparation workflow: `build_ranker()`, `run_ML()`, `rankFeatures()`, `prep_file`/`prep_class`, `--prep-file`/`--prep-class`, in-sample `evaluation_mode`, and `featranker.importance.main`.

### Known issues

- CatBoost 1.2.8 lacks `__sklearn_tags__`; scikit-learn 1.8 may turn the current deprecation warning into per-model failures.

## 0.3.0 - 2026-07-17

### Added

- Held-out CSV CLI: `--train-data`, `--eval-data`, `--target`, `--scoring`, `--feature-groups`, `--n-repeats`, `--random-state`.

## 0.2.0 - 2026-07-16

### Added

- Explicit `fit(X_train, y_train)` and held-out `rank_features(X_eval, y_eval)` workflow.
- Named/callable scoring, grouped permutation, per-model evaluation scores, uncertainty, and rank consensus.
- Structured model failures and machine-readable `evaluation_mode` provenance.
- NumPy and pandas input validation, deterministic permutation seeds, tests, and core CI.
- Optional dependency extras for XGBoost, LightGBM, and CatBoost.

### Changed

- Consensus aggregates model ranks instead of heterogeneous raw importance values.
- Core installation now requires only NumPy, scikit-learn, PyYAML, and tqdm.
- `FeatureRanker` starts unfitted unless an explicit legacy preparation file is supplied.

### Deprecated

- `rankFeatures()` performs marked in-sample ranking and warns; use held-out `rank_features()`.
- `run_ML()` warns; use `fit()`.
