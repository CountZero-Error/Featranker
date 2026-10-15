import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Lasso, Ridge

from featranker import FeatureRanker
from featranker.importance import _build_consensus

CONFIG = Path(__file__).parents[1] / "featranker" / "importance_config.yaml"


def test_every_registry_entry_declares_one_family_per_group():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    for task, groups in config.items():
        group_of_family = {}
        for group, entries in groups.items():
            for entry in entries:
                family = entry.get("family")
                assert isinstance(family, str) and family, (task, entry["name"])
                assert group_of_family.setdefault(family, group) == group, (
                    task, family
                )


def _report(family, ranks, in_consensus=True):
    return {
        "family": family,
        "in_consensus": in_consensus,
        "importance": {g: {"mean": 0.0, "rank": r} for g, r in ranks.items()},
    }


def _rows(model_reports):
    return {row["feature_group"]: row for row in _build_consensus(model_reports)}


def test_large_family_cannot_outvote_small_families():
    rows = _rows({
        "m1": _report("F1", {"x": 1, "y": 2}),
        "m2": _report("F1", {"x": 1, "y": 2}),
        "m3": _report("F1", {"x": 1, "y": 2}),
        "t1": _report("F2", {"x": 3, "y": 1}),
        "t2": _report("F3", {"x": 3, "y": 1}),
    })

    assert rows["x"]["family_ranks"] == {"F1": 1.0, "F2": 3.0, "F3": 3.0}
    assert rows["x"]["median_rank"] == 3.0  # per-model median would be 1.0
    assert rows["x"]["n_models"] == 5
    assert rows["x"]["n_families"] == 3


def test_one_model_per_family_matches_per_model_consensus():
    reports = {
        "a": _report("a", {"x": 1, "y": 2, "z": 3}),
        "b": _report("b", {"x": 2, "y": 1, "z": 3}),
        "c": _report("c", {"x": 1, "y": 3, "z": 2}),
    }
    consensus = _build_consensus(reports)

    for row in consensus:
        ranks = [r["importance"][row["feature_group"]]["rank"] for r in reports.values()]
        assert row["median_rank"] == float(np.median(ranks))
        assert row["mean_rank"] == float(np.mean(ranks))
        assert row["rank_std"] == float(np.std(ranks))
    assert [row["feature_group"] for row in consensus] == ["x", "y", "z"]


def test_gated_out_models_do_not_vote_within_family():
    rows = _rows({
        "voter": _report("F", {"x": 1, "y": 2}),
        "gated": _report("F", {"x": 2, "y": 1}, in_consensus=False),
    })

    assert rows["x"]["family_ranks"] == {"F": 1.0}
    assert rows["x"]["n_models"] == 1
    assert rows["x"]["n_families"] == 1


def test_custom_models_are_their_own_family():
    ranker = FeatureRanker(task="reg", models={"a": Ridge(), "b": Lasso()})
    ranker.fit(np.arange(12.0).reshape(6, 2), np.arange(6.0), ["p", "q"])
    report = ranker.rank_features(
        np.arange(12.0).reshape(6, 2), np.arange(6.0), scoring="r2", n_repeats=2,
        random_state=0,
    )

    assert {n: m["family"] for n, m in report["models"].items()} == {"a": "a", "b": "b"}
    json.dumps(report, allow_nan=False)


def _as_v04_pickle(ranker):
    """Round-trip a fitted ranker as if it were pickled by 0.4 (no family or verbose state)."""
    import pickle

    for attribute in ("model_families_", "verbose"):
        del ranker.__dict__[attribute]
    return pickle.loads(pickle.dumps(ranker))


def test_ranker_pickled_by_v04_rebuilds_registry_families():
    X, y = np.arange(24.0).reshape(8, 3), np.arange(8.0)
    ranker = FeatureRanker(task="reg", group="linear", verbose=False)
    ranker.fit(X, y, ["a", "b", "c"])

    loaded = _as_v04_pickle(ranker)
    report = loaded.rank_features(X, y, scoring="r2", n_repeats=1, random_state=0)

    assert loaded.verbose is True
    assert report["models"]["ridge_regression"]["family"] == "linear"
    assert report["models"]["kernel_ridge_regression"]["family"] == "linear"
    assert report["models"]["svr_regression"]["family"] == "kernel"
    loaded.fit(X, y, ["a", "b", "c"])


def test_custom_ranker_pickled_by_v04_uses_model_names():
    X, y = np.arange(12.0).reshape(6, 2), np.arange(6.0)
    ranker = FeatureRanker(task="reg", models={"a": Ridge()}, verbose=False)
    ranker.fit(X, y, ["p", "q"])

    report = _as_v04_pickle(ranker).rank_features(X, y, scoring="r2", n_repeats=1)

    assert report["models"]["a"]["family"] == "a"
