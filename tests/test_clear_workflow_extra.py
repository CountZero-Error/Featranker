import numpy as np
from sklearn.linear_model import Ridge

from featranker import FeatureRanker, summarize


def test_findings_have_null_scoring_and_experimental_only_for_shadow_codes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    ranker = FeatureRanker("reg", models={"r": Ridge()}, verbose=False).fit(
        X, y, feature_names=["a", "b", "c"]
    )
    report = ranker.rank_features(X, y, scoring="r2", n_repeats=2, random_state=0)
    assert all(f["basis"] == "descriptive" for f in summarize(report))
