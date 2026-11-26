# Concepts

This page explains how FeatRanker decides a ranking, and what each number in a report means.

[Manual](README.md) | [Getting started](getting-started.md) | [Report](report.md) | [Reference](reference.md)

## The pipeline in one picture

```text
fit()            train every model and a dummy baseline on the training rows
rank_features()  score every model on the evaluation rows
                 shuffle each feature group n_repeats times and measure the score drop
                 rank the groups inside each model
                 models that beat the dummy vote, one vote per model family
                 ->  median and mean rank consensus
```

## Models and families

FeatRanker fits many models, not one. It ships 40 (22 for classification, 18 for regression; 4 of them need the optional `xgboost` or `catboost` package), and you can bring your own ([Custom models](guide/custom-models.md)). Each built-in model belongs to a family, such as `logistic`, `bagging` or `boosting`; see [Models](models.md).

- `group` chooses which built-in models run: `"linear"`, `"tree"`, `"fast"` (one per family) or `"all"`.
- A family casts one vote, however many models it has. Four logistic regressions are one vote.
- Your own models are separate families unless you map them with `families=`.

## Permutation importance on held-out rows

For one model, the importance of a feature group is how much its score drops when the group is shuffled:

```text
importance = model_score - model_score_after_shuffling
```

- The score is computed on the evaluation rows, never on the training rows.
- All columns of a group get the same row shuffle, so the relations inside the group stay intact.
- A feature you leave out of `feature_groups` is its own group.
- `n_repeats` shuffles are made per group. More repeats give a more precise mean and take longer.
- Every model uses the same scorer.

## The dummy baseline gate

A dummy model that ignores the features is fit and scored too. A model votes only if its evaluation score is strictly better than the dummy's.

- `baseline_score` is the dummy's score. It is used only for the vote.
- If no model beats the dummy, `consensus` is empty. The features may not predict the target, or the evaluation set may be too small.
- If the dummy itself fails, every model votes and the report says so (`gate_off`). Treat that consensus with extra care.

## Consensus

Inside each model the groups are ranked by mean importance. A family's rank for a group is the median of its members' ranks. The consensus combines the family ranks.

| Number | Meaning |
|---|---|
| `median_rank` | The main ranking: the median of the family ranks. 1 is the most important. |
| `mean_rank` | Breaks ties in `median_rank`. |
| `rank_std` | How much the families disagree. It is not sampling uncertainty. |
| `n_families` | How many families voted. Fewer than 3 is a thin vote. |
| `n_models` | How many models voted. A coverage count, not confidence. |

Ranks are relative: a rank says that shuffling the group hurt more than shuffling others, not that it hurt a lot. Look at `models[name]["importance"][group]["mean"]` and `["std"]` for the size. Full definitions are in [Report](report.md#reading-the-report).

## Shadows and evidence

A rank does not say a group carries signal. Two optional tools add that.

- **Shadow features** (`shadows=True`) add a shuffled copy of every feature. A group is above shadow when it beats the strongest shadow. See [Noise and evidence](guide/noise-and-evidence.md#shadow-features).
- **Evidence** (`evidence=True`) adds train scores, overfit gaps, correlated groups and a SHAP summary. See [Noise and evidence](guide/noise-and-evidence.md#evidence).

## What a ranking cannot tell you

> [!WARNING]
> Permutation importance describes how fitted models use a feature. It does not say what happens if you change the feature, so it is no causal claim.

- Correlated features share credit. Group them, or use [`auto_groups()`](guide/selection.md#grouping-correlated-features-with-auto_groups).
- Rankings depend on the evaluation sample, the scorer and the models. Treat groups with a similar `median_rank` as tied.
- With a few dozen evaluation rows a ranking is mostly noise. Use [`rank_resampled()`](guide/validation.md#resampled-ranking) to see how much it moves.
- Choose features inside your validation folds. A ranking from your final test set must never pick features.
