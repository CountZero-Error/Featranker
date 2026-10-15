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
      Aggregate <i>ranks</i>, not incomparable raw importance magnitudes. Each model family gets one vote, so near-duplicate models cannot outvote the rest.
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <h4>▣ Dummy-baseline gate</h4>
      Only models that strictly beat a <code>DummyClassifier</code> / <code>DummyRegressor</code> vote. If the dummy itself fails, every model votes and the report says so.
    </td>
    <td width="33%" valign="top">
      <h4>▣ Parallel &amp; reproducible</h4>
      <code>n_jobs</code> runs models in parallel with joblib. With seeded estimators and a fixed <code>random_state</code>, reports match the sequential run.
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

Requires **Python ≥ 3.10**. Optional extras: `featranker[xgboost]`, `[lightgbm]`, `[catboost]`, or `[all]`.

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

Or from the command line:

```bash
featranker --task reg --train-data train.csv --eval-data eval.csv \
  --target target --scoring neg_mean_absolute_error --output report.json
```

> [!TIP]
> NumPy training matrices require `feature_names`; pandas DataFrames keep their column names. See [Usage](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md) for every parameter, your own models, `n_jobs`, the CLI, and the input contract.

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

Built-in linear models are wrapped in a `StandardScaler` pipeline fit on training data only; tree models and your own models are used as given.

> [!IMPORTANT]
> Ranking belongs inside the training portion of validation. In nested validation, fit and rank on inner folds only, and keep the outer test fold untouched until final evaluation. Never use outer-fold or final-test rankings to choose features.

## ▍ 04 · Docs

| Page | Contents |
|---|---|
| [**Usage**](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md) | API and defaults, choosing a scorer, your own models, parallel runs, the CLI, `summarize`, labels, troubleshooting |
| [**Report**](https://github.com/CountZero-Error/Featranker/blob/main/docs/report.md) | How to read the results, full report structure, ranking and failure rules |
| [**Models**](https://github.com/CountZero-Error/Featranker/blob/main/docs/models.md) | All 40 built-in models, optional extras, known issues |

Upgrading from 0.3? The preparation-file workflow was removed in 0.4.0; see the [release notes](https://github.com/CountZero-Error/Featranker/releases/tag/v0.4.0).

## ▍ Scope

FeatRanker intentionally does **not** do cross-validation, imputation, feature encoding, automatic feature selection, or SHAP-style attribution. The one target change it makes is encoding classification labels to `0..n_classes-1` internally (see [Usage](https://github.com/CountZero-Error/Featranker/blob/main/docs/usage.md#labels)). Results depend on the evaluation sample, scorer, estimators, correlated predictors, and permutation design.

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
