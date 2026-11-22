<div align="center">

# FEATRANKER

<p><b>Leakage-safe, model-agnostic feature ranking for trustworthy ML workflows.</b></p>

<p>
  <a href="https://pypi.org/project/featranker/"><img src="https://img.shields.io/pypi/v/featranker?style=for-the-badge&logo=pypi&logoColor=FFB000&label=pypi&labelColor=1C1B19&color=FFB000" alt="PyPI"></a>
  <a href="https://pypi.org/project/featranker/"><img src="https://img.shields.io/badge/python-%E2%89%A53.10-FFB000?style=for-the-badge&logo=python&logoColor=FFB000&labelColor=1C1B19" alt="Python"></a>
  <a href="https://github.com/CountZero-Error/Featranker/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/CountZero-Error/Featranker/tests.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=FFB000&label=tests&labelColor=1C1B19&color=33FF66" alt="Tests"></a>
  <a href="https://github.com/CountZero-Error/Featranker/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-FFB000?style=for-the-badge&labelColor=1C1B19" alt="MIT License"></a>
</p>

<p>
  <a href="#-01--install"><b>01 Install</b></a> &nbsp;│&nbsp;
  <a href="#-02--quick-start"><b>02 Quick start</b></a> &nbsp;│&nbsp;
  <a href="#-03--how-it-works"><b>03 How it works</b></a> &nbsp;│&nbsp;
  <a href="#-04--docs"><b>04 Docs</b></a> &nbsp;│&nbsp;
  <a href="https://github.com/CountZero-Error/Featranker/releases"><b>Releases</b></a>
</p>

</div>

```text
┌────────────────────────────────────────────────────────┐
│  FEATRANKER  [ FEATURE RANKING TERMINAL ]              │
├────────────────────────────────────────────────────────┤
│  > MODEL ARRAY ............... 40 UNITS      [ OK ]    │
│  > SEPARATE EVAL SET ......... REQUIRED      [ OK ]    │
│  > DUMMY BASELINE GATE ....... ARMED         [ OK ]    │
│  > RANK CONSENSUS ............ ONLINE        [ OK ]    │
├────────────────────────────────────────────────────────┤
│  READY._                                               │
└────────────────────────────────────────────────────────┘
```

FeatRanker trains a set of estimators on explicit training data, measures permutation importance on **separate** evaluation data, and combines each model's ranking into a single consensus. It is built for ML workflows where leakage control, semantic feature groups, and model failures must stay visible.

## ▍ Why FeatRanker?

<table>
  <tr>
    <td width="33%" valign="top">
      <h4>▣ Held-out by design</h4>
      <code>fit()</code> takes training data and <code>rank_features()</code> takes a separate evaluation set. Keeping the two apart is up to you; FeatRanker cannot detect reuse.
    </td>
    <td width="33%" valign="top">
      <h4>▣ Semantic feature groups</h4>
      Permute one-hot, encoded-category, or related feature blocks together with one shared row permutation.
    </td>
    <td width="33%" valign="top">
      <h4>▣ Rank consensus</h4>
      Aggregate <i>ranks</i>, not incomparable raw importance magnitudes. Each model family gets one vote, so near-duplicate built-in models cannot outvote the rest (your own models are separate families unless you map them with <code>families=</code>).
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <h4>▣ Dummy-baseline gate</h4>
      Only models that strictly beat a <code>DummyClassifier</code> / <code>DummyRegressor</code> vote. If the dummy itself fails, every model votes and the report says so.
    </td>
    <td width="33%" valign="top">
      <h4>▣ Parallel &amp; reproducible</h4>
      <code>n_jobs</code> runs models in parallel with joblib. With seeded estimators, a fixed <code>random_state</code> (and a fixed <code>shadow_seed</code> when shadows are on), reports match the sequential run.
    </td>
    <td width="33%" valign="top">
      <h4>▣ Auditable reports</h4>
      Per-model scores, raw repeat importances, fitted params, version provenance, and structured failures, all as strict JSON.
    </td>
  </tr>
</table>

Ships with **40 built-in models** (22 classification, 18 regression), or bring your own scikit-learn estimators and pipelines.

## ▍ 01 · Install

```bash
pip install featranker
```

Requires **Python ≥ 3.10** and scikit-learn ≥ 1.5. Optional extras: `featranker[xgboost]`, `[catboost]`, `[all]` (both), `[plot]` (matplotlib, for `report.plot()`), or `[shap]` (SHAP summaries for `evidence=True`; not part of `[all]`). The `lightgbm` extra was removed in 1.0; install `lightgbm` directly for custom LightGBM estimators. `[all]` now installs only XGBoost and CatBoost.

## ▍ 02 · Quick start

The caller owns splitting and preprocessing. Fit on training data, then rank on a separate evaluation set. This example runs as is:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)

ranker = FeatureRanker(task="reg", group="tree")
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))

report = ranker.rank_features(
    X_eval,
    y_eval,
    scoring="neg_mean_absolute_error",
    feature_groups={"serum": ["s1", "s2", "s3", "s4", "s5", "s6"]},
    random_state=42,
)

for row in report["consensus"]:
    print(f"{row['feature_group']:>6}  median rank {row['median_rank']:.1f}  ({row['n_models']} models)")
```

```text
 serum  median rank 1.0  (7 models)
   bmi  median rank 2.0  (7 models)
    bp  median rank 3.0  (7 models)
   sex  median rank 4.0  (7 models)
   age  median rank 5.0  (7 models)
```

Output shown with the `xgboost` and `catboost` extras installed; a core-only install has fewer tree models, so `n_models` is lower. The six blood-serum columns are ranked as one group. The decision tree scored worse than the dummy baseline, so it stays in the report without a vote. See [Reading the report](https://github.com/CountZero-Error/Featranker/blob/main/docs/report.md#reading-the-report) for what to do next.

For a faster run, `group="fast"` fits one model per family (7 for classification, 5 for regression), so every family has a representative with about a third of the models; a representative can fail, so the voting families can differ from `"all"` ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#speed)).

Or from the command line:

```bash
featranker --task reg --train-data train.csv --eval-data eval.csv \
  --target target --scoring neg_mean_absolute_error --output report.json
```

> [!TIP]
> NumPy training matrices require `feature_names`; pandas DataFrames keep their column names. See [Usage](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md) for every parameter, your own models, `n_jobs`, the CLI, and the input contract.

## ▍ 10-minute recipe

From a ranking to a validated subset, on three disjoint sets of rows: train (fit), validation (rank and choose `k`), and test (touched once, at the end). The script has four steps: rank on the validation rows; check support (how many models voted, what `summarize()` flags); choose `k` from the validation ablation curve; validate the chosen groups against all features on the test rows with `ablate()`. It runs as is and uses only scikit-learn models (`group="fast"`), so the output does not depend on optional extras.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
names = list(data.feature_names)
X_dev, X_test, y_dev, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.4, random_state=0)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(X_val, y_val, scoring="r2", random_state=42)

print("voting models:", report["consensus"][0]["n_models"])
print("findings:", [f["code"] for f in report.summary()])

validation = ranker.ablate(X_train, y_train, X_val, y_val, report, ks=[1, 2, 3, 5, 10])
curve = validation["ablation"]["curve"]
best = max(curve, key=lambda row: row["median_score"])
k = best["k"]
print("chosen groups:", best["feature_groups"], f"(k={k}, validation r2 {best['median_score']:.2f})")

test = ranker.ablate(X_train, y_train, X_test, y_test, report, ks=[k, 10])
scores = {row["k"]: row["median_score"] for row in test["ablation"]["curve"]}
chosen, everything = scores[k], scores[10]
print(f"test r2, chosen {k} groups: {chosen:.2f}   all 10 groups: {everything:.2f}")
```

```text
voting models: 5
findings: ['tied_groups', 'tied_groups', 'within_noise']
chosen groups: ['bmi', 's5', 'bp', 's4', 's3'] (k=5, validation r2 0.46)
test r2, chosen 5 groups: 0.13   all 10 groups: 0.15
```

What this shows. Five models voted, but the validation `summarize()` findings say several groups are tied in rank and three are within noise, so the order below the top is not firm. The validation curve picked `k=5`. On the untouched test rows those five groups score r2 0.13 against 0.15 for all ten: the selection did **not** beat using every feature here, and a smaller feature set is the only gain left (the drop from validation to test is mostly the difference between two small splits, and partly the optimism of choosing `k` on the validation curve; see [Ablation curve](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#ablation-curve)). Report the test number, not the validation one. If the chosen groups do not beat all features on test, keep all features or collect more data.

## ▍ When to use FeatRanker

Use it when you want to know which feature groups a set of different models relies on, with a ranking you can check: a consensus over model families, a dummy baseline, optional shadow features, and a refit curve (`ablate()`) that tests whether the top groups are enough.

Do not use it, or read it with care, when:

- **n is tiny.** Rankings from a few dozen evaluation rows are mostly noise; ties and `within_noise` findings are the signal to stop. Use `rank_resampled()` to see how much the order moves.
- **You need causal claims.** Permutation importance describes how fitted models use a feature, not what happens if you change it.
- **You want one model's importance.** The consensus is over many models by design; for a single model use scikit-learn's `permutation_importance` directly.
- **p is very wide and you have no feature groups.** Every column is its own group and the cost grows with the number of groups; group columns, use `group="fast"`, or see the memory and speed notes in the [usage guide](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#speed).
- **You would shrink a real-data feature set with `min_shadow_share=0.5` alone.** In the [real-data study](https://github.com/CountZero-Error/Featranker/blob/main/docs/realdata.md) that rule kept as little as 1% of the features on some datasets and lost held-out score there; choose `k` from the ablation curve with nested cross-validation and check held-out scores.
- **You would trust the shadow p- or q-values as calibrated.** They are experimental; see the [report guide](https://github.com/CountZero-Error/Featranker/blob/main/docs/report.md) and the [benchmark](https://github.com/CountZero-Error/Featranker/blob/main/docs/benchmark.md), which also shows where FeatRanker loses.

## ▍ 03 · How it works

```text
fit()            train every model + a dummy baseline on the training data
rank_features()  score every model on the evaluation data
                 shuffle each feature group n_repeats times, measure the score drop
                 rank the groups within each model
                 models that beat the dummy vote, one vote per model family
                 ->  median / mean rank consensus
```

For each model, a feature group's importance is how much that model's score drops when the group is shuffled:

```text
importance = model_score - model_score_after_shuffling
```

Every model uses the same scorer. In the report, the unshuffled score is each model's `evaluation_score`; `baseline_score` is the dummy's score, used only for the vote.

To see why a ranking looks the way it does, `FeatureRanker(..., evidence=True)` adds each model's train score and overfit gap, pairs of correlated groups, and (with `featranker[shap]`) how each group pushes predictions ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#evidence)).

Permutation importance splits credit between near-copies. `auto_groups(X)` proposes `feature_groups` from correlation (no target), and `ConsensusSelector(feature_groups="auto")` or the CLI's `--feature-groups auto` computes them on the training data ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#auto-groups)).

To see how stable a ranking is, `rank_resampled()` repeats this over cross-validation splits and reports each group's rank range and how often it lands in the top *k* ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#resampled-ranking)).

With `shadows=True`, each stability row also shows which model families beat their shuffled shadows (`shadow_families`), and `ConsensusSelector(min_shadow_families=2)` keeps groups that at least two families find, for effects such as interactions that only some model types capture. It is a heuristic: its false-selection rate is not yet measured (see the calibration study) ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#family-evidence)).

Built-in linear models are wrapped in a `StandardScaler` pipeline fit on training data only; tree models and your own models are used as given. With `missing="impute"`, NaN is accepted and every built-in model starts with a median imputer fit on its training split only, so evaluation rows never shape the imputation ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#missing-values)).

With `categorical="auto"` (or a list of column names), raw text and category columns are accepted: levels are learned from each fit's training rows only, built-in linear models one-hot encode them and trees use the codes, and every column is still ranked as one feature ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#categorical-features)).

With `model_input="dataframe"`, every model, custom ones included, receives a pandas DataFrame with the feature names (and `shadow__<name>` shadow columns), so your own pipelines can select columns by name; a pipeline that drops the shadow columns is flagged and left out of the shadow evidence ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#columns-by-name)).

With `sample_weight=` on `fit()`, `rank_features()`, and `rank_resampled()` (and `--sample-weight COLUMN` on the CLI), models are fit and every score is computed with your weights; models that cannot take weights (built-in `lda`, `qda`, `pls_regression`) are recorded as fit failures, and correlation and SHAP evidence stay unweighted ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#sample-weights)).

With `max_samples=` on `rank_features()` and `rank_resampled()` (`ConsensusSelector(max_samples=...)`, `--max-samples` on the CLI), permutation scores use one random subsample of the evaluation rows, the same for the dummy gate and every score ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#fewer-evaluation-rows-max_samples)).

For large data, `fit(max_samples=)` fits on a subsample of the training rows (`ConsensusSelector(max_train_samples=)`, `--max-train-samples`), and `max_categories=` caps the one-hot columns per categorical column in the built-in linear models (`--max-categories`) ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#large-data)).

With `scoring=["r2", "neg_mean_absolute_error"]` (a list; `--scoring a,b` on the CLI), the first scorer ranks and the others are scored on the same permuted copies, reported per model as `secondary` ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#several-metrics)).

With `shuffle="within_groups"` (labels passed as `shuffle_groups=`, or `groups=` in `rank_resampled()` and `ConsensusSelector`) or `shuffle="circular"` (rows in time order), permutations and shadows shuffle within groups or by a cyclic shift instead of freely, which respects grouped or time-ordered rows ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#shuffling-dependent-rows)).

With `ranker.ablate(X_train, y_train, X_eval, y_eval, report)`, the voting models are refit on the top-*k* consensus groups for several *k*, giving a curve of how well the top groups alone score; the CLI's `--ablate` adds it, and `--drop id,created_at` removes ID or timestamp columns from both CSVs ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#ablation-curve)).

Every report carries a `run` block (preset, data and split fingerprints, and the `auto_groups` threshold when grouping was automatic) so repeated runs can be compared, and `summarize()` reads resampled reports too ([details](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#comparing-runs)).

> [!IMPORTANT]
> Ranking belongs inside the training portion of validation. In nested validation, fit and rank on inner folds only, and keep the outer test fold untouched until final evaluation. Never use outer-fold or final-test rankings to choose features. [`ConsensusSelector`](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#feature-selection) does this for you as a scikit-learn `Pipeline` step.

## ▍ 04 · Docs

| Page | Contents |
|---|---|
| [**Recipe**](#-10-minute-recipe) and [**When to use**](#-when-to-use-featranker) | Ranking to a validated subset in one script, and when not to use FeatRanker |
| [**Usage**](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md) | API and defaults, working with reports (`summary`, `to_frame`, `plot`), choosing a scorer, your own models, columns by name, missing values, categorical features, sample weights, parallel runs, the CLI, resampled ranking, ablation curve, shadow features, the shadow hits test, per-family evidence and `min_shadow_families`, mechanism evidence, pipeline feature selection, `summarize` (single-split and resampled), comparing runs, labels, troubleshooting |
| [**Report**](https://github.com/CountZero-Error/Featranker/blob/main/docs/report.md) | How to read the results, resampled (stability) reports, full report structure, ranking and failure rules |
| [**Models**](https://github.com/CountZero-Error/Featranker/blob/main/docs/models.md) | All 40 built-in models, optional extras, known issues |
| [**API**](https://github.com/CountZero-Error/Featranker/blob/main/docs/api.md) | Public API, stability policy, typing, releases |
| [**Benchmark**](https://github.com/CountZero-Error/Featranker/blob/main/docs/benchmark.md) | FeatRanker's selector against Boruta, a permutation-importance rule, and `SelectByShuffling` on synthetic data with known signal, including where it loses |

Upgrading from 0.3? The preparation-file workflow was removed in 0.4.0; see the [release notes](https://github.com/CountZero-Error/Featranker/releases/tag/v0.4.0).

## ▍ Scope

`FeatureRanker` does not tune models; it encodes only the columns you mark as `categorical` (inside each fit, with level codes and, for built-in linear models, one-hot) and nothing else; by default it rejects missing values, and `missing="impute"` applies a per-split median imputation inside the built-in models. Its optional SHAP summaries (`evidence=True`) describe how models use features and are not used in the ranking. `ConsensusSelector` performs feature selection inside validation. `rank_resampled()` splits data only to measure ranking stability; it does not evaluate a final model. The one target change it makes is encoding classification labels to `0..n_classes-1` internally (see [Usage](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#labels)). Results depend on the evaluation sample, scorer, estimators, correlated predictors, and permutation design.

## ▍ Development

```bash
git clone https://github.com/CountZero-Error/Featranker.git
cd Featranker
pip install -e '.[test]'
python -m pytest -q
```

## ▍ License

Released under the [MIT License](https://github.com/CountZero-Error/Featranker/blob/main/LICENSE).

---

<div align="center">
<p><sub><code>■ END OF TAPE ■</code> &nbsp;│&nbsp; Built by <a href="https://github.com/CountZero-Error">CountZero</a> &nbsp;│&nbsp; <a href="#featranker">Rewind ↑</a></sub></p>
</div>
