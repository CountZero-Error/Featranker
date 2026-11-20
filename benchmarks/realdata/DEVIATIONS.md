# Real-data study: deviations from PREREG.md

Recorded 2026-11-20 after the script review and BEFORE any study run. PREREG.md itself is unchanged (sha256 339712d2...).

1. `shadow_seed=seed` on both FeatRanker selectors. PREREG did not fix it; the default (fresh entropy) would make reruns differ.
2. `n_jobs=-1` on the selectors and the random forests. Speed only.
3. `LogisticRegression(max_iter=2000)`, so the linear model converges on unscaled-heavy datasets. PREREG named the model, not its settings.
4. An empty selection (a selector keeps no feature) is recorded as an error row with error type `EmptySelection`, and its outcome is scored with a dummy model (`DummyClassifier(strategy="prior")` / `DummyRegressor()`), stored as `dummy_score`; the summary uses that score, so an empty selection counts as the loss it is instead of disappearing.
5. Median imputation is applied in every downstream pipeline and before the `rf_median` forest on all datasets, not only those with missing cells. On a dataset with no missing cells it is the identity.
6. The "counts" outcome is over (dataset, downstream model) pairs, not datasets, because each dataset has three models. Per-dataset tables are reported as well.
7. A dataset that fails to load is written as error rows for every arm and model; `--resume` does not retry it.
8. The script records the imported FeatRanker tree (path, source sha256, commit given) and checks loaded shapes against the PREREG table; a shape mismatch is recorded, not fatal.
