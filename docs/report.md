# FEATRANKER ▍ REPORT

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <a href="usage.md">Usage</a> &nbsp;│&nbsp; <b>Report</b> &nbsp;│&nbsp; <a href="models.md">Models</a></sub>

`rank_features()` returns a dict that keeps the raw evidence from each model and builds the consensus from ranks only. The CLI writes the same structure as JSON.

<a id="reading-the-report"></a>

## ▍ Reading the report

Check these in order.

**1. Was the dummy gate active?** `baseline_score` is the score of a dummy model that ignores the features. If it is `None` and `failures` has an entry for `dummy_baseline`, the gate was off: every model that ranked has `in_consensus: True`, whether or not it beat the dummy. Treat that consensus with extra caution.

**2. Did enough models learn anything?** When the gate is active, compare each model's `evaluation_score` with `baseline_score`. Models that beat it have `in_consensus: True` and vote. When the gate is off, skip this comparison: votes carry no guarantee of beating a dummy. If none vote, `consensus` is empty: the features may not predict the target, or the evaluation set may be too small.

**3. Did anything fail?** Any model in `failures` was skipped. Read `stage` and `message` to see why.

**4. Read the consensus.** `consensus` is sorted from most to least important. Rank 1 is the most important.

| Field | How to read it |
|---|---|
| `median_rank` | The main ranking. The middle rank across voting models, so one odd model cannot move it much. |
| `mean_rank` | Breaks ties in `median_rank`. A big gap from `median_rank` means a few models rank this group very differently. |
| `rank_std` | How much the voting models disagree about this group's position. Near 0 means they agree. It does not measure sampling uncertainty: a different evaluation set can move a group even when models agree. |
| `n_models` | How many models voted. Every voting model ranks every group, so it is the same for all groups. It measures coverage, not confidence: built-in models include close variants that tend to agree, so many votes are not many independent confirmations. |

**5. Look at the importances behind a rank.** A rank is relative: it says a group's shuffle hurt the score more than other groups' shuffles did, not that it hurt a lot. For a group you care about, open `models[<name>]["importance"][<group>]` for a few models:

- `mean` is the average score drop on this evaluation set with this scorer. Near 0 or negative means shuffling the group did not measurably hurt the score. That does not prove the model ignores the feature: a correlated feature can stand in for it (see the caveats below), or the scorer may not respond to the change.
- `std` is the spread of the score drop across the `n_repeats` shuffles. A large spread relative to `mean` means individual shuffles disagree. More repeats make `mean` more precise but do not shrink this spread, and neither says whether the ranking holds on other data. To check that, rerun on a different evaluation split.

### Caveats

- **Correlated features share credit.** If two features carry the same information, shuffling one leaves the other, so both can look unimportant. Put them in one `feature_groups` entry to rank them together. See scikit-learn's [notes on correlated features](https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features).
- **Rankings depend on the evaluation data and scorer.** A different split or metric can reorder close groups. Treat groups with similar `median_rank` as tied.
- **Use rankings inside validation only.** Pick or drop features using inner-fold results, never the final test set. See the [leakage warning](../README.md#-03--how-it-works).

## ▍ Structure

`evaluation_mode` is always `"held_out"`. It names the workflow (scoring on data passed to `rank_features()`); FeatRanker cannot check that this data is really separate from the training data.

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
        "featranker": "0.4.0",
        "python": "3.13.1",
        "numpy": "2.2.0",
        "scikit-learn": "1.7.1",
    },
    "models": {
        "random_forest_regressor": {
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
        }
    ],
    "failures": [],
}
```

## ▍ Ranking

- Raw importance values stay model-specific and are never averaged into the primary consensus.
- Exactly equal mean importances share the minimum rank. Group names give a deterministic output order.
- The consensus sorts by `median_rank`, then `mean_rank`, then group name.

## ▍ Dummy baseline

- `baseline_score` is the evaluation score of a default `DummyClassifier()` or `DummyRegressor()` fit on the same training data and scored with the same scorer.
- A model has `in_consensus: True` only if its `evaluation_score` is strictly greater than `baseline_score` (scikit-learn scorers are higher-is-better). Models at or below the baseline stay in `models` but do not vote in `consensus`.
- If no model beats the baseline, `consensus` is an empty list and a `UserWarning` is emitted; nothing is raised.
- If the baseline cannot be fit or scored, `baseline_score` is `None`, every model has `in_consensus: True`, and `failures` gains an entry for model `dummy_baseline`.

## ▍ Failures

- Failures contain `model`, `stage`, `error_type`, and `message`.
- Failed models get no zero placeholder and do not enter the consensus.
- `fit()` raises if no model fits; `rank_features()` raises if every model fails ranking.

## ▍ Provenance and reproducibility

- `provenance` records the installed `featranker` version (`None` if the package is not installed), plus the Python, NumPy, and scikit-learn versions.
- Each model's `params` is its fitted estimator's `get_params(deep=True)`, limited to strings, integers, booleans, `None`, and finite floats. Nested estimator objects are left out, while pipeline entries such as `ridge__alpha` remain.
- Reports are valid strict JSON.
- `random_state` controls permutation reproducibility. Estimator randomness stays estimator-specific.
