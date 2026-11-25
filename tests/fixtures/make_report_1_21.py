"""Made tests/fixtures/report_1_21_categorical.json; run with PYTHONPATH at a checkout of 951088f (1.21)."""
import json
import sys
import warnings

import numpy as np
import pandas as pd

from featranker import FeatureRanker


def data(task):
    rng = np.random.default_rng(123)
    x, z = np.round(rng.normal(size=(2, 150)), 3)
    frame = pd.DataFrame({"x": x, "z": z, "c": np.array(["red", "blue", "green"])[rng.integers(0, 3, 150)]})
    target = np.round(3 * x + 2 * (frame["c"] == "red").to_numpy() - z, 3)
    return frame, target if task == "reg" else (target > 0.5).astype(int)


CASES = [("reg", "fast", None), ("reg", "linear", None), ("reg", "linear", 2),
         ("clf", "fast", None), ("clf", "linear", None), ("clf", "tree", None)]


def key(task, group, cap):
    return f"{task}-{group}-{cap}"


def fitted(task, group, cap):
    X, y = data(task)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker = FeatureRanker(task, group, categorical=["c"], max_categories=cap, verbose=False)
        return ranker.fit(X[:100], y[:100]), X, y


def main(path):
    for optional in ("xgboost", "lightgbm", "catboost"):
        sys.modules[optional] = None  # reports without optional models work on every install
    reports = {}
    for case in CASES:
        ranker, X, y = fitted(*case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reports[key(*case)] = ranker.rank_features(
                X[100:], y[100:], scoring="r2" if case[0] == "reg" else "accuracy",
                n_repeats=1, random_state=1)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=1, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
