# Selection

English · [简体中文](../zh-CN/guide/selection.md)

This page shows how to turn a ranking into a feature subset: `ConsensusSelector` in a pipeline, the ablation curve, `summarize()`, comparing runs, and grouped permutation with `auto_groups()`.

[Manual](../README.md) | [Validation](validation.md) | [Noise and evidence](noise-and-evidence.md) | [Report](../report.md) | [Reference](../reference.md)

## ConsensusSelector in a pipeline

Choosing features from a ranking of all your data and then cross-validating a model on them overstates its performance. `ConsensusSelector` avoids that by ranking inside `fit()`: as a `Pipeline` step under `cross_val_score` or `GridSearchCV`, it reranks on each training fold and never sees the rows it is scored on.

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featranker import ConsensusSelector

X, y = load_diabetes(return_X_y=True, as_frame=True)
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=4, group="fast", cv=3, n_repeats=3,
        random_state=0,
    )),
    ("model", Ridge()),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"nested-CV MAE: {-scores.mean():.1f} +/- {scores.std():.1f}")

pipe.fit(X, y)  # final model on all development data
print("selected:", pipe.named_steps["select"].selected_groups_)
```

```text
nested-CV MAE: 50.2 +/- 2.1
selected: ['s5', 'bmi', 'bp', 's3']
```

The score is honest because every outer fold chose its own four groups from its own training rows. The last line shows the groups chosen from all the data for the final model.

Set at least one rule: `k`, `min_shadow_share`, `max_shadow_q` or `min_shadow_families`. The shadow filters apply first, then the first `k` groups that pass. See [Reference](../reference.md#consensusselector) for every parameter.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

rng = np.random.default_rng(0)
values = np.hstack([X.to_numpy(), rng.normal(size=(len(X), 2))])  # two pure-noise columns
names = list(X.columns) + ["noise_1", "noise_2"]

selector = ConsensusSelector(
    "reg", "r2", max_shadow_q=0.05, group="fast", n_repeats=3, random_state=0,
    shadow_seed=0, cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0),
)
selector.fit(pd.DataFrame(values, columns=names), y)
print(selector.selected_groups_)
```

```text
['s5', 'bmi']
```

A selected group keeps all its columns, in their original order. Feature names come from DataFrame columns; for arrays they are `x0`, `x1`, ... and `feature_groups` must use those names. After `fit()` you have `selected_groups_`, `report_` (the resampled report from this fit), `n_features_in_`, `feature_names_in_` (DataFrames), and the standard `get_support()` and `get_feature_names_out()`.

- `report_` reflects only the data that fit saw. For the ranking of your whole development set, call `rank_resampled()` directly.
- If any split failed or had no consensus, `fit()` warns how many splits did not vote. Those splits already count as misses in the shares; look at `report_["failures"]` and `report_["splits"]` to see why.
- If no group passes, `fit()` warns and the selector keeps no columns, as scikit-learn's own selectors do. Most downstream estimators then fail on zero columns. Loosen the shadow rules, or use only a positive `k`.
- With a shadow rule, `fit()` raises `ValueError` when no model kept the shadow columns in any split, instead of selecting without shadow evidence.
- Invalid rules raise `ValueError` in `fit()`, not in `__init__`, so `clone`, `set_params` and `GridSearchCV` over `select__k` work.
- For a group-aware inner `cv` such as `GroupKFold`, pass `fit(X, y, groups=...)`; in a `Pipeline`, pass `select__groups=...` (for example `cross_val_score(..., params={"select__groups": groups})`). With `shuffle="within_groups"` the same groups also confine the shuffles.
- DataFrame column labels are compared as strings in `fit()` and `transform()`. A failed `fit()` clears any earlier fitted state.
- `min_shadow_share=0.5` kept as little as 1% of the features on some of the 13 datasets in the [real-data study](../realdata.md) and lost held-out score there. `max_shadow_q` is experimental and needs `cv` with 10 or more splits; `min_shadow_families` is a heuristic ([Noise and evidence](noise-and-evidence.md#family-evidence)).

> [!NOTE]
> Each `fit()` is one `rank_resampled()` run: `cv` rankings, doubled with shadows. Inside `cross_val_score(cv=n)` that is `n x cv` rankings, plus one more run when you fit the final pipeline. `GridSearchCV` multiplies the cost by the number of candidates, plus its final refit. Use `group="fast"` or `"tree"`, a small `cv` and fewer `n_repeats` while searching.

## Ablation curve

Permutation importance shows how much a fitted model loses when a group becomes noise. It does not show whether the top groups are enough: correlated groups share credit, so each can look small while removing both hurts. `ablate()` checks the ranking by refitting. For each `k`, every model that voted in the consensus is fit again, with the configuration `fit()` used (later `set_params` on your estimators has no effect), on the columns of the top-`k` groups, and scored on the evaluation data. The curve answers "how many groups do I need": where it flattens, later groups add little.

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from featranker import FeatureRanker

data = load_diabetes()
names = list(data.feature_names)
X_dev, X_test, y_dev, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.4, random_state=0
)

ranker = FeatureRanker(task="reg", group="fast", verbose=False)
ranker.fit(X_train, y_train, feature_names=names)
report = ranker.rank_features(X_val, y_val, scoring="r2", random_state=42)

ks = [1, 2, 3, 5, 10]  # same refits, scored on validation rows and on untouched test rows
validation = ranker.ablate(X_train, y_train, X_val, y_val, report, ks=ks)
test = ranker.ablate(X_train, y_train, X_test, y_test, report, ks=ks)
for on_val, on_test in zip(validation["ablation"]["curve"], test["ablation"]["curve"]):
    print(f"k={on_val['k']:>2}  validation {on_val['median_score']:.2f}  "
          f"test {on_test['median_score']:.2f}  added {on_val['feature_groups'][-1]}")
```

```text
k= 1  validation 0.26  test 0.04  added bmi
k= 2  validation 0.43  test 0.16  added s5
k= 3  validation 0.41  test 0.17  added bp
k= 5  validation 0.46  test 0.13  added s3
k=10  validation 0.44  test 0.15  added sex
```

Both curves refit the same models on the same training rows and the same top-`k` groups; only the scored rows differ. The validation curve rises from 0.26 at `k=1` to 0.43 at `k=2` and then stays within 0.41 to 0.46, so the first two or three groups carry most of it. The test rows score lower at every `k`, including `k=10`, where no groups were left out: 0.15 against 0.44. Most of that gap is the difference between two small splits, not selection. If you pick `k=3` from the validation curve, the validation rows chose `k`, so 0.41 is a validation score. The untouched test rows (0.17 for the same groups) are the number to report.

> [!WARNING]
> By default the curve is scored on the evaluation data that also produced the ranking, so it is optimistic: the top groups were chosen because they scored well there. A third data set as `X_eval` and `y_eval` removes that optimism for a `k` you fixed in advance. Once you use the curve to choose `k`, that data set has been used for selection and its best score is a validation score. Report final performance on another untouched test set, or with outer cross-validation around the whole procedure (a `ConsensusSelector` in a `Pipeline` under `cross_val_score`). Choose `k` inside that loop, for example `GridSearchCV(pipe, {"select__k": [2, 3, 5]})` as the estimator passed to `cross_val_score`. A fixed `k` picked from the curve and then cross-validated is still optimistic, because the same rows chose it.

How it works:

- **Groups and columns.** The top `k` rows of `report["consensus"]` are kept, in list order. Their columns are the union of their `feature_groups` entries, in training column order. Groups are never split.
- **Models.** Only the voting models (`in_consensus: true`) are refit. An empty consensus raises `ValueError`.
- **Same preprocessing.** Each `k` builds a fresh `FeatureRanker` with the ranker's `task`, `group` or `models`, `n_jobs`, `missing`, `model_input` and `categorical` columns (those among the kept ones), without shadows or evidence. Imputation, categorical coding, named input and sample weights behave as in the ranking run.
- **Cost.** `len(ks)` refits of every voting model. A shorter `ks` list saves time.
- **Failures.** A model that fails to fit or score at some `k` gets `null` there and an entry in `ablation["failures"]`; other models and ks continue. A model that cannot fit few columns (the built-in `pls_regression` with one column) fails at small `k` this way. `median_score` is the median over the models that did not fail, and `null` if all did.
- **Paired median.** The models that fail can change from one `k` to the next, so `median_score` can move although no model's score did. Each row therefore also has `n_models_scored` and `paired_median_score`, the median over `paired_models`, the models with a score at every `k` (`null` if none). Prefer `paired_median_score` when `n_models_scored` varies.
- **Provenance.** The block records `data_sha256` (a [fingerprint](#comparing-runs) of `X_eval` and `y_eval`), `weights_sha256` (`null` without weights), `n_eval_rows`, and `same_holdout`: `true` when that data and weights are the ones the report was ranked on.
- **Baseline.** `baseline_score` is the dummy's score from the refits themselves (same training rows and weights, every `X_eval` row, the scorer used), so it is comparable to the curve. It can differ from the report's `baseline_score`, which used the report's own rows, scorer and `max_samples`. It is `null` only if no refit scored.
- **Summary.** `summarize()` adds an `ablation_curve` finding and, when the set of scored models differs between rows, an `ablation_cohort_changed` warning.
- **Weights.** The ranker does not remember weights, so pass `sample_weight` and `eval_sample_weight` again; they are checked against the report's `run["train_weights_sha256"]` and `run["weights_sha256"]`. A third data set takes any weights.

The curve is descriptive. FeatRanker does not choose `k` for you, and it does not drop one group at a time. For a resampled report there is no ablation; ablate a held-out report. A ranker with user models pickled before 1.22 refits a clone of its fitted model, not the current template.

> [!NOTE]
> `ablate()` refuses a report that does not match the ranker with `ValueError`: a resampled report, an empty consensus, a voting model of another class or with other parameters, other training data, other weights, or a voting model the ranker's latest `fit()` did not fit. A report made under another scikit-learn version is compared on the parameters both versions have and only warns when they differ. See [Reference](../reference.md#input-rules).

## Summarize a report

`summarize(report)` applies the [reading guide](../report.md#reading-the-report) for you. It returns a list of findings and does not change the report. It works on the dict from `rank_features()`, on a report loaded from the CLI's JSON and on a resampled report.

```python
from featranker import summarize

for finding in summarize(report):
    print(finding["severity"], finding["code"])
```

```text
info tied_groups
info tied_groups
info within_noise
```

Each finding is a dict with `code`, `severity` (`"info"` or `"warning"`), `message`, JSON-safe `evidence`, `basis` (`"descriptive"`, or `"experimental"` when it rests on shadow p- or q-values; today only `insufficient_splits`) and `scoring`. Codes are public; messages are not. Findings come in this order:

| Code | Severity | When |
|---|---|---|
| `gate_off` | warning | The dummy baseline failed, so every model voted. |
| `empty_consensus` | warning | No model beat the dummy. |
| `failures` | warning for `fit` and `ranking`; info for `initialization` and `evidence` | One finding per failed stage, listing the models. |
| `few_families` | warning | Fewer than 3 model families voted. |
| `tied_groups` | info | Groups share the same `median_rank`; one finding per tie. It does not measure ranking uncertainty. |
| `within_noise` | info | In most voting models the group's mean importance is no greater than its standard deviation across shuffles. With `n_repeats=1` it only catches importances of 0 or less. |
| `below_shadow` | info | With shadows on, groups whose `above_shadow_share` is below 0.5. Descriptive, not a significance test. |
| `overfit` | info | With evidence on: models that lost at least half of their training advantage over the dummy on the evaluation data, `(train_score - evaluation_score) / (train_score - baseline_score) >= 0.5`. Skipped when `baseline_score` is `None`. |
| `correlated_groups` | info | With evidence on: groups correlated on the training data (absolute Spearman rho of at least 0.5). |
| `constant_within_groups` | warning | A report ranked with `shuffle="within_groups"` has groups constant within every shuffle group; their importance is zero by construction. |
| `ablation_curve` | info | The report has an `ablation` block with a score: the `k` with the highest `paired_median_score` (smallest on ties) and its score at the largest `k`. Evidence: `best_k`, `best_median_score`, `max_k`, `max_k_median_score`. |
| `ablation_cohort_changed` | warning | The curve's rows have different sets of scored models, so the unpaired `median_score` compares different sets. |

An empty list means nothing stood out, not that the ranking is proven. The checks cover one evaluation set.

### Summarizing a resampled report

A resampled report has its own findings, in this order, each only when its condition holds. To read one split, summarize `report["splits"][i]["report"]`.

| Code | Severity | When |
|---|---|---|
| `split_failures` | warning | Splits that failed outright, with indices and error types. They count as misses in every share. |
| `failures` | as for a single split | Model failures aggregated over splits, each model with the splits it failed in. |
| `gate_off` | warning | Splits whose `baseline_score` is `None`. |
| `empty_consensus` | warning | Splits whose consensus is empty. |
| `below_shadow` | info | With shadows on, groups whose stability `above_shadow_share` is below 0.5. |
| `constant_within_groups` | warning | Groups constant within every shuffle group in some splits (evidence `groups` and `splits`). |
| `insufficient_splits` | info | With shadows on, the hits test cannot reach a q-value of 0.05 with this many splits (`shadow_test.min_p_value > 0.05`, below 5 splits). |

```python
ranker = FeatureRanker(task="reg", group="fast", shadows=True, shadow_seed=0, verbose=False)
resampled = ranker.rank_resampled(
    data.data, data.target, feature_names=names,
    scoring="r2", cv=3, n_repeats=3, random_state=0,
)
for finding in summarize(resampled):
    print(finding["severity"], finding["code"])
```

```text
info below_shadow
info insufficient_splits
```

The per-split findings (`few_families`, `tied_groups` and the rest) are not repeated here.

## Comparing runs

Every report has a `run` block that says what it was computed from, so two reports can be told apart without keeping the data.

- `task`, `group`: the preset.
- `fingerprint_scheme`: `2`. Integer and boolean columns (class labels, for example) are hashed exactly, tagged by dtype kind, so integers above 2**53 differ. Booleans hash as 0 and 1 and `int32` equals `int64` for the same values; an integer column and a float column with the same numbers still differ. `ablate()` reads a report without this key as scheme 1.
- `data_sha256`: a fingerprint of the evaluation data for `rank_features()`, and of all the data passed to `rank_resampled()`.
- `train_data_sha256`: a fingerprint of the data passed to `fit()` (`rank_features()` reports only; `null` for a ranker pickled before 1.12).
- `weights_sha256` (and `train_weights_sha256`): a fingerprint of the sample weights, present only when weights were given.
- `shuffle_groups_sha256` (and `train_shuffle_groups_sha256`): a fingerprint of the shuffle labels, present only when labels were given.
- `rank_resampled()` only: `cv` (the integer given, the splitter's `repr` without memory addresses, or `"iterable of splits"`) and `splits_sha256`, a fingerprint of every split's train and evaluation row indices.
- `auto_groups`: `{"threshold": 0.7, "method": "spearman-complete-linkage"}`, only when `feature_groups="auto"` chose the groups.

A fingerprint is the SHA-256 digest of the feature names, the feature values and the target values. Equal inputs under the same library versions give equal fingerprints. Changing any value, a name or the row order changes it. Fingerprints may differ across NumPy or pandas versions, notably for text and other object columns (the versions are in `provenance`); a categorical `None` in NumPy input and a NaN from pandas hash differently. They do not cover model settings.

```python
X_train, X_eval, y_train, y_eval = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0
)


def rank(X_eval, y_eval):
    ranker = FeatureRanker(task="reg", group="fast", verbose=False)
    ranker.fit(X_train, y_train, feature_names=names)
    return ranker.rank_features(X_eval, y_eval, scoring="r2", n_repeats=3, random_state=0)


first = rank(X_eval, y_eval)
print("same inputs:", first["run"] == rank(X_eval, y_eval)["run"])

changed = y_eval.copy()
changed[0] += 1.0
other = rank(X_eval, changed)
print("evaluation data differs:", first["run"]["data_sha256"] != other["run"]["data_sha256"])
print("training data differs:", first["run"]["train_data_sha256"] != other["run"]["train_data_sha256"])
```

```text
same inputs: True
evaluation data differs: True
training data differs: False
```

To compare two resampled reports, check `run["data_sha256"]`, `run["cv"]` and `run["splits_sha256"]` first: with all three equal, the stability tables came from the same rows in the same splits.

## Grouped permutation

Related columns can be shuffled as one unit with `feature_groups`:

```text
feature_groups = {
    "segment": ["segment_red", "segment_blue", "segment_other"],
    "channel": ["channel_web", "channel_store", "channel_partner"],
}
```

Every column in a multi-column group receives the same row permutation during a repeat. That keeps the relations inside the block while breaking its relation with the target. Features left out of `feature_groups` become singleton groups. Empty groups, unknown features, duplicate membership, overlaps and group-name collisions raise errors.

### Grouping correlated features with auto_groups

Permutation importance splits credit between near-copies: shuffling one of two correlated columns barely hurts a model that can use the other, so both look unimportant. `auto_groups(X)` proposes the groups from `X` alone (it never sees a target) and returns a dict you can pass as `feature_groups`.

```python
from featranker import auto_groups

Xf, _ = load_diabetes(return_X_y=True, as_frame=True)

print(auto_groups(Xf))
print(auto_groups(Xf, threshold=0.4))
print(auto_groups(Xf, threshold=0.9))
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{'bmi+s5': ['bmi', 's5'], 's1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
{}
```

- Columns are clustered by Spearman correlation with complete linkage, so every pair inside a group has absolute correlation of at least `threshold` (up to a tolerance of `1e-12`, which lets exact copies group at `threshold=1`). Lowering it makes bigger groups. A constant column is never grouped.
- Use enough rows. With few rows the sample correlation is inflated (with two rows every non-constant pair has absolute correlation 1), so near-unrelated columns can be grouped.
- Only groups of two or more columns are returned, named by their members joined with `+`, in column order. A name equal to an existing column or another group raises `ValueError`.
- `X` is validated like `fit()` input. Names come from a DataFrame, else `feature_names`, else `x0`, `x1`, .... With `categorical` those columns are accepted as text and never grouped.
- Call it on your training or development data and pass the result to `rank_features()` or `rank_resampled()`. The correlation evidence lists group pairs from 0.5, which shows what sits near the cut.
- The correlation matrix and clustering are O(p squared) in the number of columns, which is fine for hundreds of columns.

Where you cannot compute the groups yourself, use `"auto"`: `ConsensusSelector(feature_groups="auto")` in a pipeline, or `--feature-groups auto` on the command line. Both use the default threshold of 0.7; change it with `ConsensusSelector(auto_groups_threshold=...)` or `--auto-groups-threshold`. The report records the threshold in `run["auto_groups"]`, and only when the grouping was automatic. Under `cross_val_score` each outer training fold computes its own groups, so evaluation rows never shape them.

```python
pipe = Pipeline([
    ("select", ConsensusSelector(
        "reg", "neg_mean_absolute_error", k=3, group="fast", cv=3, n_repeats=3,
        feature_groups="auto", random_state=0,
    )),
    ("model", Ridge()),
])
pipe.fit(Xf, y)

select = pipe.named_steps["select"]
print({name: cols for name, cols in select.report_["feature_groups"].items() if len(cols) > 1})
print("selected:", select.selected_groups_)
```

```text
{'s1+s2': ['s1', 's2'], 's3+s4': ['s3', 's4']}
selected: ['s5', 'bmi', 's3+s4']
```
