"""Checked by mypy in CI (not run by pytest): public annotations must expose
the Report helpers and accept the documented argument types."""

import numpy as np

from featranker import ConsensusSelector, FeatureRanker, Report

ranker = FeatureRanker(task="reg")
fast_ranker = FeatureRanker(task="reg", group="fast")
single: Report = ranker.rank_features(np.ones((4, 1)), np.ones(4), scoring="r2")
single.summary()
single.to_frame()
single.plot()
resampled: Report = ranker.rank_resampled(np.ones((4, 1)), np.ones(4), scoring="r2", cv=2)
resampled.to_frame()

selector = ConsensusSelector("reg", "r2", k=np.int64(2))
fast_selector = ConsensusSelector("reg", "r2", k=2, group="fast")
selector.report_.plot()
q_selector = ConsensusSelector("reg", "r2", max_shadow_q=0.05)
family_selector = ConsensusSelector(
    "reg", "r2", min_shadow_families=2, shadow_mode="independent"
)
independent_ranker = FeatureRanker(task="reg", shadows=True, shadow_mode="independent")
missing_ranker = FeatureRanker(task="reg", missing="impute")
missing_selector = ConsensusSelector("reg", "r2", k=2, missing="impute")
categorical_ranker = FeatureRanker(task="reg", categorical="auto")
listed_ranker = FeatureRanker(task="reg", categorical=["city"])
categorical_selector = ConsensusSelector("reg", "r2", k=2, categorical=["city"])
named_ranker = FeatureRanker(task="reg", model_input="dataframe")
named_selector = ConsensusSelector("reg", "r2", k=2, model_input="dataframe")
