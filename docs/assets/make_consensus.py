"""Regenerate docs/assets/consensus.png from the README quick-start example.

Run from the repo root: PYTHONPATH=. python docs/assets/make_consensus.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

ax = report.plot()
ax.set_title("Consensus ranking, diabetes quick start")
ax.figure.savefig(Path(__file__).with_name("consensus.png"), dpi=150, bbox_inches="tight")
