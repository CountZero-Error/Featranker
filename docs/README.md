# FEATRANKER ▍ DOCS

<sub><a href="../README.md">◂ Back to README</a></sub>

```text
┌────────────────────────────────────────────────────────┐
│  FEATRANKER  [ REFERENCE MANUAL ]                      │
└────────────────────────────────────────────────────────┘
```

| Page | Contents |
|---|---|
| [**10-minute recipe**](../README.md#-10-minute-recipe) and [**When to use FeatRanker**](../README.md#-when-to-use-featranker) | Rank, check support, choose a subset, and validate it on untouched rows; and when not to use FeatRanker |
| [**Usage**](usage.md) | API and defaults, choosing task/group/scorer, your own models, parallel runs, the CLI, resampled ranking, the ablation curve, shadow features and per-family evidence, mechanism evidence, pipeline feature selection, summarizing a report, comparing runs, grouped permutation, label encoding, and troubleshooting |
| [**Report**](report.md) | How to read the results, resampled (stability) reports, the full report structure, and the rules for ranking, failures, and reproducibility |
| [**Models**](models.md) | All 40 built-in models and their families, optional estimator libraries, and known issues |
| [**API**](api.md) | What is public, the 1.0 stability and deprecation policy, typing, and how releases are published |
| [**Benchmark**](benchmark.md) | FeatRanker's selector against Boruta, a permutation-importance rule, and `SelectByShuffling` on synthetic data with known signal: method, results, and where FeatRanker loses |
| [**Real-data study**](realdata.md) | FeatRanker's selection against all features and a random-forest comparator on 13 OpenML datasets, held-out scores, and what the shadow-share rule dropped |

New to FeatRanker? Start with the [quick start](../README.md#-02--quick-start).
