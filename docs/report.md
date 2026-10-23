# FEATRANKER ▍ REPORT

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <a href="usage.md">Usage</a> &nbsp;│&nbsp; <b>Report</b> &nbsp;│&nbsp; <a href="models.md">Models</a> &nbsp;│&nbsp; <a href="api.md">API</a></sub>

`rank_features()` returns a [`Report`](usage.md#working-with-reports), a `dict` subclass, that keeps the raw evidence from each model and builds the consensus from ranks only. The CLI writes the same structure as JSON.

<a id="reading-the-report"></a>

## ▍ Reading the report

Check these in order. `featranker.summarize(report)` runs the same checks and returns them as structured findings; see [Summarize a report](usage.md#summarize).

**1. Was the dummy gate active?** `baseline_score` is the score of a dummy model that ignores the features. If it is `None` and `failures` has an entry for `dummy_baseline`, the gate was off: every model that ranked has `in_consensus: True`, whether or not it beat the dummy. Treat that consensus with extra caution.

**2. Did enough models learn anything?** When the gate is active, compare each model's `evaluation_score` with `baseline_score`. Models that beat it have `in_consensus: True` and vote. When the gate is off, skip this comparison: votes carry no guarantee of beating a dummy. If none vote, `consensus` is empty: the features may not predict the target, or the evaluation set may be too small.

**3. Did anything fail?** Any model in `failures` was skipped. Read `stage` and `message` to see why.

**4. Read the consensus.** `consensus` is sorted from most to least important. Rank 1 is the most important.

Each model family casts one vote. Voting models are grouped by `family` (for example, the four logistic regressions form one family). A family's rank for a group is the median of its members' ranks, and the consensus combines those family ranks. A family with five members therefore counts as much as a family with one. Your own models are each their own family.

| Field | How to read it |
|---|---|
| `median_rank` | The main ranking: the median of the family ranks, averaging the middle two for an even count. With one or two voting families, it offers little protection against an unusual family rank. |
| `mean_rank` | Breaks ties in `median_rank`. A big gap from `median_rank` means a few families rank this group very differently. |
| `rank_std` | How much the families disagree about this group's position. Near 0 means they agree. It does not measure sampling uncertainty: a different evaluation set can move a group even when families agree. |
| `n_families` | How many families voted. With fewer than 3, the consensus rests on very few independent views. |
| `family_ranks` | Each voting family's rank for this group. Use it to see which families drive or disagree with the consensus. |
| `n_models` | How many models voted. Every voting model ranks every group, so it is the same for all groups. It measures coverage, not confidence; `n_families` is the better count of independent views. |
| `above_shadow_share` | Only with [shadows](usage.md#shadow-features) on. Each voting family contributes its fraction of members whose mean importance beat that model's strongest shadow group; this is their average. Near 1 means it beats the shuffled reference almost everywhere. It is relative and descriptive, not a p-value or proof of useful signal. |
| `family_above_shadow` | Only with shadows on: family name to the fraction of that family's voting members that beat their model's strongest shadow, with keys sorted by family. `above_shadow_share` is the mean of these values. See [family evidence](usage.md#family-evidence). |

**5. Look at the importances behind a rank.** A rank is relative: it says a group's shuffle hurt the score more than other groups' shuffles did, not that it hurt a lot. For a group you care about, open `models[<name>]["importance"][<group>]` for a few models:

- `mean` is the average score drop on this evaluation set with this scorer. Near 0 or negative means shuffling the group did not measurably hurt the score. That does not prove the model ignores the feature: a correlated feature can stand in for it (see the caveats below), or the scorer may not respond to the change.
- `std` is the spread of the score drop across the `n_repeats` shuffles. A large spread relative to `mean` means individual shuffles disagree. More repeats make `mean` more precise but do not shrink this spread, and neither says whether the ranking holds on other data. To check that, rerun on a different evaluation split.
- With shadows on, `above_shadow` says whether `mean` beat the model's `shadow_max`, the largest mean importance among the shuffled shadow groups. Each model's `shadow_importance` holds the shadow groups' `values`, `mean`, and `std` for comparison.

### Caveats

- **Correlated features share credit.** If two features carry the same information, shuffling one leaves the other, so both can look unimportant. Put them in one `feature_groups` entry to rank them together; with `evidence=True`, `evidence.correlation` lists such pairs. See scikit-learn's [notes on correlated features](https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features).
- **Rankings depend on the evaluation data and scorer.** A different split or metric can reorder close groups. Treat groups with similar `median_rank` as tied.
- **Use rankings inside validation only.** Pick or drop features using inner-fold results, never the final test set. See the [leakage warning](../README.md#-03--how-it-works).

<a id="resampled-reports"></a>

## ▍ Resampled reports

[`rank_resampled()`](usage.md#resampled-ranking) returns a report with `evaluation_mode: "resampled"`:

```python
{
    "evaluation_mode": "resampled",
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 10,
    "random_state": 0,
    "feature_groups": {"bmi": ["bmi"], "...": ["..."]},
    "provenance": {"featranker": "0.6.0", "...": "..."},
    "run": {
        "task": "reg",
        "group": "all",
        "data_sha256": "9f2c...",
        "cv": 5,                      # or the splitter's repr
        "splits_sha256": "41ab...",
    },
    "top_k": 3,
    "n_splits": 5,
    "splits": [
        {"split": 0, "n_train": 353, "n_eval": 89, "report": {...}},  # a normal report
    ],
    "stability": [
        {
            "feature_group": "bmi",
            "median_rank": 2.0,
            "min_rank": 1,
            "max_rank": 2,
            "split_ranks": [2, 1, 1, 2, 2],
            "n_splits_ranked": 5,
            "top_k_share": 1.0,
        }
    ],
    "failures": [],
}
```

Each `splits[i]["report"]` is an ordinary report for that split; read it with the guide above or [`summarize()`](usage.md#summarize). `summarize(report)` on the resampled report itself lists the [split-level findings](usage.md#summarize-resampled): failed splits, splits without a gate or a consensus, groups below their shadows, and too few splits for the shadow test.

How to read `stability`, which is sorted by `median_rank`, then `top_k_share` (higher first), then name:

| Field | How to read it |
|---|---|
| `split_ranks` | The group's place in each split's sorted consensus (1 = first), in split order; `null` where that split had an empty consensus. Groups tied on both `median_rank` and `mean_rank` share the better place. |
| `median_rank` | The median place across ranked splits. Use it for the overall order. |
| `min_rank`, `max_rank` | The best and worst place seen. A wide range means the position depends on which rows were used for evaluation. |
| `top_k_share` | Fraction of attempted splits in which the group's place is at most `top_k`. Failed and empty-consensus splits contribute zero. Inspect `n_splits_ranked` and failures first: an intermediate share can reflect missing rankings, changing ranks, or both; no fixed band establishes statistical uncertainty. |
| `n_splits_ranked` | How many splits produced a consensus. It is the same for every group. |
| `above_shadow_share` | Only with shadows on: the sum of the group's per-split `above_shadow_share` divided by `n_splits`. A split that failed, had an empty consensus, or has no consensus row for the group counts as 0 (as for `shadow_hits`), so one lucky split out of ten gives at most 0.1. When every split has a consensus, this is the mean over splits. |
| `shadow_hits` | Only with shadows on: the number of splits in which the group's consensus `above_shadow_share` was above 0.5. A split with no consensus row for the group is a non-hit. |
| `shadow_p_value` | Only with shadows on: the one-sided binomial probability of at least `shadow_hits` hits in `n_splits` trials at probability 0.5. An uncalibrated reference value: splits overlap and the 0.5 null is not established for arbitrary estimators. See [the hits test](usage.md#shadow-test). |
| `shadow_q_value` | Only with shadows on: `shadow_p_value` adjusted with Benjamini-Hochberg across all groups. Also a reference value, not a calibrated false-discovery rate. |
| `family_above_shadow_share` | Only with shadows on: family name to the sum of that family's per-split `family_above_shadow` divided by `n_splits`, for each family that voted in at least one split. A split in which it did not vote, failed, or was empty counts as 0 (as for `shadow_hits`). Keys are sorted by family. Unlike in a consensus row, the mean of these values equals the stability `above_shadow_share` only when the same families vote in every split. |
| `shadow_families` | Only with shadows on: the number of families whose `family_above_shadow_share` is strictly above 0.5. `ConsensusSelector(min_shadow_families=N)` keeps groups with at least `N`. |

With shadows on, the report also has a top-level `shadow_test`: `method` (`"binomial"`), `hit` (`"above_shadow_share > 0.5"`), `null_probability` (0.5), `correction` (`"benjamini-hochberg"`), `n_trials` (equal to `n_splits`), `min_p_value` (`0.5 ** n_splits`, the smallest p-value the test can give; with `cv=5` it is 0.03125, so use 10 or more splits), `calibrated` (always `false`), and `note` (a sentence saying that the values are binomial reference values whose null and false-discovery-rate control are not established for arbitrary estimators or overlapping splits). Read the p and q values as reference values, not significance.

Within each split, `rank_std` still measures disagreement between model families, not between splits; between-split variation is what `min_rank`, `max_rank`, and `top_k_share` show. With few splits (for example `cv=3`), these numbers are coarse. Every split also draws on the same data, so a stable ranking here is not a guarantee on new data.

`failures` holds every split's failures, each with a `split` index. A split that failed entirely appears with `"model": null` and `"stage": "split"`.

<a id="structure"></a>

## ▍ Structure

`evaluation_mode` is always `"held_out"`. It names the workflow (scoring on data passed to `rank_features()`); FeatRanker cannot check that this data is really separate from the training data.

Every report has a `run` block (after `provenance`): `task`, `group`, `data_sha256`, and, for `rank_features()`, `train_data_sha256` (`null` for a ranker pickled before 1.12); for `rank_resampled()`, `cv` and `splits_sha256` instead, with each split's report carrying its own `run`. When `feature_groups="auto"` chose the groups (`ConsensusSelector`, `--feature-groups auto`), `run["auto_groups"]` is `{"threshold": ..., "method": "spearman-complete-linkage"}`; otherwise the key is absent. See [Comparing runs](usage.md#comparing-runs).

With `shadows=True`, the report also has top-level `shadows: true`, `shadow_seed`, and `shadow_mode` (`"joint"` or `"independent"`; resampled reports add the `shadow_test` block, and their stability rows gain `shadow_hits`, `shadow_p_value`, `shadow_q_value`, `family_above_shadow_share`, and `shadow_families`); each model gains `shadow_importance` and `shadow_max`; each importance entry gains `above_shadow`; and each consensus row gains `above_shadow_share` and `family_above_shadow`. Without shadows, none of these keys appear.

With `missing="impute"`, the report has a top-level `missing` block: `policy` (`"impute"`), `strategy` (`"median"`), and `train_counts` (every feature name to the number of its missing training values). Resampled reports carry it in each split's report, counted on that split's training rows. With the default `missing="error"` there is no `missing` key. See [Missing values](usage.md#missing-values).

When the data has categorical columns (`categorical="auto"` or a list of names), the report has a top-level `categorical` block: `columns` (the categorical feature names), `train_levels` (name to the number of levels learned from the training rows, the missing-value level included), and `unseen_eval` (name to the number of evaluation values that got the shared "unseen" code). Resampled reports carry it in each split's report, learned on that split's training rows. Without categorical columns there is no `categorical` key. See [Categorical features](usage.md#categorical-features).

With `model_input="dataframe"` the report (and each resampled split report) has `"model_input": "dataframe"`; in the default `"array"` mode the key is absent. See [Columns by name](usage.md#columns-by-name).

When any [sample weights](usage.md#sample-weights) are used, the report (and each resampled split report) has `"sample_weight": {"train": bool, "eval": bool}`, saying whether training and evaluation weights were given; without weights the key is absent. A model that cannot take weights has a `failures` entry with stage `"fit"` and `error_type` `"SampleWeightNotSupported"`. Correlation evidence and the SHAP summary stay unweighted.

With [`max_samples`](usage.md#fewer-evaluation-rows-max_samples), the report has top-level `max_samples` (as given) and, for `rank_features()` (and each resampled split report), `n_eval_rows`, the number of evaluation rows actually scored; without it neither key appears. `run["data_sha256"]` still covers all the evaluation rows passed.

With `shuffle="within_groups"` or `"circular"` (see [Shuffling dependent rows](usage.md#shuffling-dependent-rows)), the report (and each resampled split report) has top-level `"shuffle"` with that value; with the default `"rows"` the key is absent.

<a id="ablation"></a>

A report returned by [`FeatureRanker.ablate()`](usage.md#ablation-curve) (or written by the CLI with `--ablate`) has one more top-level key, `ablation`; every other key is a copy of the report it was made from, and without `ablate()` the key is absent.

```python
"ablation": {
    "scoring": "r2",                  # scorer used for the curve
    "models": ["ridge", "random_forest"],   # the voting models that were refit
    "baseline_score": -0.01,          # the refits' dummy score; null only if no refit scored
    "n_groups": 10,                   # consensus groups; the largest possible k
    "curve": [
        {
            "k": 2,
            "feature_groups": ["s5", "bmi"],   # the top-k consensus groups, in order
            "n_features": 2,                   # columns they hold
            "scores": {"ridge": 0.31, "random_forest": 0.17},   # null where the model failed
            "median_score": 0.24,              # median of the non-null scores; null if none
        }
    ],
    "failures": [{"model": "pls_regression", "stage": "fit", "message": "...", "k": 1}],
}
```

Scores are the models' scores on the data passed as `X_eval`, higher is better. `baseline_score` is the dummy's score from the refits themselves, on every row of `X_eval` with the scorer named in `scoring`, so it can differ from the report's own `baseline_score`; it is `null` only if no refit scored. Ablation `failures` entries have `model`, `stage` (`"fit"` or `"ranking"`), `message`, and `k`, and no `error_type`; they are separate from the report's own `failures`. All numbers are finite or `null`.

<a id="several-scorers"></a>

With [several scorers](usage.md#several-metrics) (`scoring=["r2", "neg_mean_absolute_error"]`), `scoring` names the primary scorer, and the report adds `secondary_scoring` (the other names, in order) and `secondary_baseline_scores` (name to the fitted dummy's score, or `null` if it could not be scored). Each model gains `secondary`: per secondary scorer, `{"evaluation_score": ..., "importance": {group: {"mean": ..., "std": ...}}}` computed on the same permuted copies as the primary `importance`, or `null` if that scorer failed for the model (the model stays ranked, with a `failures` entry at stage `"evidence"` naming the scorer). `consensus`, `stability`, shadows, and everything else use the primary scorer only. Resampled reports have `secondary_scoring` at the top and `secondary` in each split's models. With one scorer none of these keys appear.

With `evidence=True`, each model also has `train_score` and `overfit_gap` (`null` if the training score could not be computed, with a `failures` entry at stage `"evidence"`), the report has a top-level `evidence` block, and models SHAP can explain have `shap`. Without `evidence`, none of these keys appear. See [Evidence](usage.md#evidence).

```python
{
    "models": {
        "ridge": {
            # ... the keys above, plus:
            "train_score": 0.55,
            "overfit_gap": 0.16,          # train_score - evaluation_score
            "shap": {                     # only when shap is installed and supports the model
                "explainer": "linear",    # or "tree"
                "groups": {
                    "bmi": {"mean_abs": 21.4, "direction": 1.0},
                    "segment": {"mean_abs": 3.2, "direction": None},  # multi-column group
                },
            },
        }
    },
    "evidence": {
        "correlation": {
            "method": "spearman",
            "data": "train",
            "threshold": 0.5,
            "pairs": [{"groups": ["s1", "s2"], "rho": 0.87}],   # sorted by |rho|
        },
        "shap": {
            "available": True,        # {"available": False} when shap is not installed
            "n_rows": 133,            # evaluation rows explained (at most 1000)
            "unsupported": [{"model": "knn", "reason": "InvalidModelError: ..."}],
        },
    },
}
```

`mean_abs` is on the explainer's output scale, which may be a link scale rather than the prediction scale (a linear model's linear predictor, for example log units for `PoissonRegressor`), so compare it between groups of one model, not across models. `direction` is `null` for multi-column groups, constant features, and multiclass models; for binary classification the explained output is the positive class (the second label in sorted order). For linear models `direction` is the coefficient's sign (+1 or -1) when the coefficient is not zero, even when `mean_abs` is near 0, and `null` when it is zero.

```python
{
    "evaluation_mode": "held_out",
    "scoring": "neg_mean_absolute_error",
    "n_repeats": 20,
    "random_state": 42,
    "feature_groups": {
        "segment": ["segment_red", "segment_blue", "segment_other"]
    },
    "baseline_score": -14.7,
    "provenance": {
        "featranker": "0.5.0",
        "python": "3.13.1",
        "numpy": "2.2.0",
        "scikit-learn": "1.7.1",
    },
    "run": {
        "task": "reg",
        "group": "tree",
        "data_sha256": "9f2c...",         # evaluation data
        "train_data_sha256": "c07e...",   # data passed to fit()
    },
    "models": {
        "random_forest_regressor": {
            "family": "bagging",
            "evaluation_score": -8.2,
            "in_consensus": True,
            "params": {"n_estimators": 200, "max_depth": None},
            "importance": {
                "segment": {
                    "values": [1.1, 1.3],
                    "mean": 1.2,
                    "std": 0.1,
                    "rank": 1,
                }
            },
        }
    },
    "consensus": [
        {
            "feature_group": "segment",
            "median_rank": 1.0,
            "mean_rank": 1.0,
            "rank_std": 0.0,
            "n_models": 1,
            "n_families": 1,
            "family_ranks": {"bagging": 1.0},
        }
    ],
    "failures": [],
}
```

## ▍ Ranking

- Raw importance values stay model-specific and are never averaged into the primary consensus.
- Exactly equal mean importances share the minimum rank. Group names give a deterministic output order.
- Each voting model's rank is first combined within its family (median), then across families: `median_rank`, `mean_rank`, and `rank_std` are the median, mean, and population standard deviation of the family ranks.
- Built-in families are listed in [Models](models.md). Each model passed through `models=` is its own family, so with your own models every model votes separately.
- The consensus sorts by `median_rank`, then `mean_rank`, then group name.

## ▍ Dummy baseline

- `baseline_score` is the evaluation score of a default `DummyClassifier()` or `DummyRegressor()` fit on the same training data and scored with the same scorer.
- A model has `in_consensus: True` only if its `evaluation_score` is strictly greater than `baseline_score` (scikit-learn scorers are higher-is-better). Models at or below the baseline stay in `models` but do not vote in `consensus`.
- If no model beats the baseline, `consensus` is an empty list and a `UserWarning` is emitted; nothing is raised.
- If the baseline cannot be fit or scored, `baseline_score` is `None`, every model has `in_consensus: True`, and `failures` gains an entry for model `dummy_baseline`.

## ▍ Failures

- Failures contain `model`, `stage`, `error_type`, and `message`. The stages are `initialization`, `fit`, `ranking`, (with `evidence=True`) `evidence`, and (with `shadows=True`) `shadows`. An `evidence` failure leaves the model's ranking untouched. A `shadows` failure (`error_type` `ShadowColumnsDropped`) means a `ColumnTransformer` in a custom model ignores some shadow columns: the model keeps its importances and ranking vote but has no shadow evidence in that report (see [Columns by name](usage.md#columns-by-name)).
- Failed models get no zero placeholder and do not enter the consensus.
- `fit()` raises if no model fits; `rank_features()` raises if every model fails ranking.

## ▍ Provenance and reproducibility

- `provenance` records the installed `featranker` version (`None` if the package is not installed), plus the Python, NumPy, and scikit-learn versions.
- `run` records the preset and fingerprints of the data and splits, so two reports can be checked for the same inputs; see [Comparing runs](usage.md#comparing-runs).
- Each model's `params` is its fitted estimator's `get_params(deep=True)`, limited to strings, integers, booleans, `None`, and finite floats. Nested estimator objects are left out, while pipeline entries such as `ridge__alpha` remain.
- Reports are valid strict JSON.
- `random_state` controls permutation reproducibility. Estimator randomness stays estimator-specific.
