# Performance

English · [简体中文](../zh-CN/guide/performance.md)

This page shows how to make a run faster and keep its memory in check: a smaller model set, fewer evaluation rows, a training subsample, a one-hot cap and parallel models.

[Manual](../README.md) | [Models and scoring](models-and-scoring.md) | [Custom models](custom-models.md) | [Reference](../reference.md)

## Where the time goes

Ranking time is models x groups x repeats x (1 + shadows) x scored rows. Each option below cuts one of these factors.

| To cut | Use |
|---|---|
| Models | `group="fast"` (one model per family, about a third of the models) |
| Groups | `feature_groups` of your own, or [`auto_groups`](selection.md#grouping-correlated-features-with-auto_groups) |
| Repeats | a smaller `n_repeats` |
| Evaluation rows | `rank_features(max_samples=)` |
| Training rows | `fit(max_samples=)` |
| One-hot width | `max_categories` |
| Wall time | `n_jobs`, see [Parallel models](custom-models.md#parallel-models) |

> [!NOTE]
> `svr_regression`, the regression kernel family's only model, takes time quadratic in the number of rows. On large data pass your own `models=` instead (`group="fast"` cannot be combined with `models=`).

## Fewer evaluation rows: max_samples

Permutation importance predicts over every evaluation row once per model, feature group and repeat, so on a large evaluation set most of that time buys little precision. `max_samples` scores on a random subsample instead. An int is a number of rows (more than there are uses all); a float in (0, 1] is a share (`int(max_samples * n)`, rounded down, at least 1: 0.29 of 200 rows is 57). Anything else raises `ValueError` before scoring.

- One subsample, drawn without replacement with `random_state`, is used for the dummy gate, every baseline and every permuted score, so they stay comparable.
- Training scores and evidence are unchanged: SHAP values and shadows still use every row, and `overfit_gap` is the train score minus the model's score on the subsample.
- With `sample_weight` the weights follow their rows. If the drawn rows all have zero weight, `rank_features()` raises `ValueError` naming `max_samples` and `sample_weight` (in `rank_resampled()` that split fails).
- The report gains `max_samples` and `n_eval_rows`. `run["data_sha256"]` still fingerprints all the evaluation rows you passed.
- It works the same in `rank_resampled()` (each split's evaluation rows), `ConsensusSelector(max_samples=...)` and the CLI (`--max-samples`).

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
for max_samples in (None, 0.5, 40):
    report = ranker.rank_features(
        X_eval, y_eval, scoring="r2", random_state=42, max_samples=max_samples
    )
    top = [row["feature_group"] for row in report["consensus"][:3]]
    print(f"max_samples={max_samples}: {report.get('n_eval_rows', len(y_eval))} rows, top 3 {top}")
```

```text
max_samples=None: 133 rows, top 3 ['s5', 'bmi', 'bp']
max_samples=0.5: 66 rows, top 3 ['s5', 'bmi', 's6']
max_samples=40: 40 rows, top 3 ['bmi', 's5', 'sex']
```

A half-size sample already swaps `bp` out of the top three, so judge the ranking's stability before you trade rows for speed.

> [!WARNING]
> The sample is plain random, not stratified. A small one can miss a rare class, and a scorer such as `roc_auc` can then fail. If every model fails, `rank_features()` raises a `RuntimeError` listing the failures. Keep `max_samples` large enough to hold every class; there is no stratified option.

## Large data

Time and memory grow in different places, and each option bounds one of them. The figures below were measured on a 16 GB laptop (Python 3.12, scikit-learn 1.5.1); the raw rows are in [`spikes/memory/FINDINGS.md`](https://github.com/CountZero-Error/Featranker/blob/main/spikes/memory/FINDINGS.md).

| Option | Bounds | Does not bound |
|---|---|---|
| `rank_features(max_samples=)` | Rows scored: the per-model predictions and the scratch copy of the scored matrix. On 100k x 500 with one cheap model, peak RSS fell from 2.7 to 2.0 GB. | Fitting, the shadow block, evidence, or the validated copy of all evaluation rows. |
| `fit(max_samples=)` | Rows the models are fit on, so tree fit time, and the training shadows and evidence copy. On 100k x 500 with the seven `"fast"` models, fitting took 11 s with 5000 rows. | The one validated copy of every row passed in, and the evaluation side. With `fit(max_samples=5000)` alone the run peaked at 3.3 GB; with `rank_features(max_samples=5000)` as well, 1.9 GB. |
| `max_categories` | The one-hot width of built-in linear models: at most that many columns per categorical column. The one-hot block is sparse for most built-in linear models (10k rows, one 1000-level column, Ridge and Lasso: peak 1.4 MB sparse against 193 MB dense). With 10 categorical columns of 1000 levels on 4000 rows and dense output, peak RSS fell from 3.1 GB to 0.37 GB (fit 46 s to 2.7 s); this still applies to the six dense-only estimators. | Tree and custom models, which use one integer code per column anyway. The infrequent levels share one column, so linear scores can differ from the uncapped ones. |
| `n_repeats`, `feature_groups` | Ranking time. `feature_groups` (your own, or `"auto"`) shrinks the number of groups. | Memory. |

The shadow block is built in place (peak 2X of the matrix, not 3X: 2.5 to 2.1 GB at 100k x 500). The Spearman matrix behind `evidence=True` and `auto_groups()` ranks a block of columns at a time, so its peak falls from about 7X of the matrix to 1.2X at 500 columns; `auto_groups` on 100k x 500 peaked at 1.8 GB. A model's own scratch memory, such as `GaussianNB` predicting over 1000 shadow-doubled columns, is outside the package and not reduced.

### Training subsample

`fit(max_samples=)` takes an int or a float in (0, 1], like `rank_features()`'s, and draws the rows with its own `random_state` (a stream distinct from the evaluation draw; `None` is fresh entropy). It keeps the original row order, and under `shuffle="within_groups"` draws whole groups so every sampled group can still be shuffled.

- If the sampled rows' `sample_weight` values are all zero it raises `ValueError`.
- The sample is plain random. If it misses a class of a classification target (every model would fail), `fit()` raises `ValueError` naming the class, so raise `max_samples`.
- `run["n_train_rows"]`, `run["train_max_samples"]`, `run["train_random_state"]` and `run["train_rows_sha256"]` identify the sample. `run["train_data_sha256"]` still fingerprints the full data, so [`ablate()`](selection.md#ablation-curve) accepts the report and refits on the same sampled rows. A ranker fit on other rows (another `random_state` or `max_samples`) is refused with `ValueError`.
- `rank_resampled(max_train_samples=)`, `ConsensusSelector(max_train_samples=)` and `--max-train-samples` apply it to each split's training rows, seeded with the split's seed.
- `max_categories` (`ConsensusSelector(max_categories=)`, `--max-categories`) is recorded in `run["max_categories"]`. It is an int of at least 2 and below 2**63. A level unseen in training shares the infrequent column only for a column with at least `max_categories` levels; otherwise it is all zeros, as without the cap.

This sketch needs a large data set with `X_train`, `X_eval` and a class target, so it is shown without being run:

```text
ranker = FeatureRanker(task="clf", group="fast", categorical="auto", max_categories=20)
ranker.fit(X_train, y_train, max_samples=50_000, random_state=0)
report = ranker.rank_features(X_eval, y_eval, scoring="roc_auc", max_samples=10_000, random_state=0)
print(report["run"]["n_train_rows"], report["run"]["max_categories"])  # rows fitted on, level cap
```

## Putting it together

For a large run combine `group="fast"`, `max_samples`, a smaller `n_repeats`, your own `feature_groups` and [`n_jobs`](custom-models.md#parallel-models). Check the ranking's stability on a smaller problem first, with [`rank_resampled()`](validation.md#resampled-ranking).
