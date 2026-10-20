import json
import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from featranker import FeatureRanker

SHADOW_KEYS = {"shadows", "shadow_seed", "shadow_importance", "shadow_max",
               "above_shadow", "above_shadow_share"}

RNG = np.random.default_rng(0)
X_DATA = np.column_stack([np.arange(40.0), RNG.normal(size=40), RNG.normal(size=40)])
Y_DATA = 3.0 * X_DATA[:, 0] + 0.1 * RNG.normal(size=40)
NAMES = ["signal", "noise_a", "noise_b"]
SEEN = []


class RecordingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        SEEN.append(np.array(X))
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def ranker(models=None, **kwargs):
    return FeatureRanker(
        task="reg", models=models or {"lin": LinearRegression()}, verbose=False, **kwargs
    )


def run(r, X=X_DATA, y=Y_DATA, **kwargs):
    r.fit(X[:30], y[:30], NAMES)
    kwargs = {"scoring": "r2", "n_repeats": 3, "random_state": 0, **kwargs}
    return r.rank_features(X[30:], y[30:], **kwargs)


def all_keys(obj):
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in all_keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in all_keys(v)}
    return set()


@pytest.mark.parametrize(
    "kwargs, match",
    [({"shadows": "yes"}, "shadows"), ({"shadows": 1}, "shadows"),
     ({"shadows": True, "shadow_seed": 1.5}, "shadow_seed"),
     ({"shadows": True, "shadow_seed": True}, "shadow_seed")],
)
def test_invalid_shadow_options(kwargs, match):
    with pytest.raises(TypeError, match=match):
        ranker(**kwargs)


def test_shadows_off_is_unchanged():
    default = run(ranker())
    explicit = run(ranker(shadows=False))

    assert default == explicit
    assert not all_keys(default) & SHADOW_KEYS


def test_models_train_on_real_plus_shadow_columns_from_training_rows():
    SEEN.clear()
    r = ranker({"rec": RecordingRegressor()}, shadows=True, shadow_seed=0)
    r.fit(X_DATA[:30], Y_DATA[:30], NAMES)

    seen = SEEN[0]
    assert seen.shape == (30, 6)
    np.testing.assert_array_equal(seen[:, :3], X_DATA[:30])
    real_rows = {tuple(row) for row in X_DATA[:30]}
    assert all(tuple(row) in real_rows for row in seen[:, 3:])
    assert not np.array_equal(seen[:, 3:], seen[:, :3])  # shuffled
    assert r.baseline_model_.n_features_in_ == 3
    assert r.n_features_in_ == 3


def test_ranker_pickled_without_shadow_state_loads_with_shadows_off():
    r = ranker()
    r.fit(X_DATA[:30], Y_DATA[:30], NAMES)
    for attribute in ("shadows", "shadow_seed"):
        del r.__dict__[attribute]

    loaded = pickle.loads(pickle.dumps(r))
    report = loaded.rank_features(X_DATA[30:], Y_DATA[30:], scoring="r2", n_repeats=1)

    assert loaded.shadows is False and loaded.shadow_seed is None
    assert not all_keys(report) & SHADOW_KEYS


class PredictRecorder(RecordingRegressor):
    def predict(self, X):
        SEEN.append(np.array(X))
        return super().predict(X)


def test_eval_shadow_block_comes_from_eval_rows():
    SEEN.clear()
    run(ranker({"rec": PredictRecorder()}, shadows=True, shadow_seed=1), n_repeats=1)

    unpermuted = SEEN[1]  # SEEN[0] is fit; SEEN[1] is the unshuffled eval score
    assert unpermuted.shape == (10, 6)
    np.testing.assert_array_equal(unpermuted[:, :3], X_DATA[30:])
    eval_rows = {tuple(row) for row in X_DATA[30:]}
    assert all(tuple(row) in eval_rows for row in unpermuted[:, 3:])


def test_report_has_shadow_fields():
    report = run(ranker(shadows=True, shadow_seed=0))

    assert report["shadows"] is True and report["shadow_seed"] == 0
    model = report["models"]["lin"]
    assert set(model["shadow_importance"]) == set(NAMES)
    assert set(model["shadow_importance"]["signal"]) == {"values", "mean", "std"}
    assert model["shadow_max"] == max(s["mean"] for s in model["shadow_importance"].values())
    for name in NAMES:
        entry = model["importance"][name]
        assert entry["above_shadow"] == (entry["mean"] > model["shadow_max"])
    json.dumps(report, allow_nan=False)


def test_strong_signal_beats_every_shadow():
    report = run(ranker(shadows=True, shadow_seed=0))

    assert report["models"]["lin"]["importance"]["signal"]["above_shadow"] is True


def test_real_ranks_unchanged_by_shadows_for_linear_model():
    X = RNG.normal(size=(60, 3))
    y = 5.0 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * X[:, 2] + 0.01 * RNG.normal(size=60)

    def ranks(**kwargs):
        report = run(ranker(**kwargs), X=X, y=y)
        return {g: v["rank"] for g, v in report["models"]["lin"]["importance"].items()}

    assert ranks() == ranks(shadows=True, shadow_seed=0) == {
        "signal": 1, "noise_a": 2, "noise_b": 3,
    }


def test_seeded_shadow_runs_are_reproducible():
    assert run(ranker(shadows=True, shadow_seed=3)) == run(ranker(shadows=True, shadow_seed=3))


from featranker.importance import _build_consensus, _build_stability


def _model(family, above, in_consensus=True):
    return {
        "family": family,
        "in_consensus": in_consensus,
        "shadow_max": 0.1,
        "importance": {"g": {"mean": 0.5, "rank": 1, "above_shadow": above}},
    }


def test_above_shadow_share_is_family_weighted():
    rows = _build_consensus({
        "a1": _model("F1", True),
        "a2": _model("F1", True),
        "b1": _model("F2", False),
        "gated": _model("F3", True, in_consensus=False),
    })

    assert rows[0]["above_shadow_share"] == 0.5


def test_no_share_without_shadows():
    model = {"family": "F", "in_consensus": True,
             "importance": {"g": {"mean": 0.5, "rank": 1}}}

    assert "above_shadow_share" not in _build_consensus({"m": model})[0]


def _split(index, share):
    consensus = [] if share is None else [
        {"feature_group": "g", "median_rank": 1.0, "mean_rank": 1.0,
         "above_shadow_share": share}
    ]
    return {"split": index, "n_train": 4, "n_eval": 2,
            "report": {"feature_groups": {"g": ["g"]}, "consensus": consensus}}


def test_stability_share_is_mean_over_ranked_splits():
    rows = _build_stability([_split(0, 1.0), _split(1, 0.5), _split(2, None)], 3, 1)

    assert rows[0]["above_shadow_share"] == 0.75


def test_stability_has_no_share_without_shadows():
    split = _split(0, 1.0)
    del split["report"]["consensus"][0]["above_shadow_share"]

    assert "above_shadow_share" not in _build_stability([split], 1, 1)[0]


def test_resampled_passes_shadow_seed_plus_split_index():
    r = ranker(shadows=True, shadow_seed=10)

    report = r.rank_resampled(X_DATA, Y_DATA, feature_names=NAMES, scoring="r2",
                              cv=3, n_repeats=1, random_state=0)

    assert [s["report"]["shadow_seed"] for s in report["splits"]] == [10, 11, 12]
    assert report["shadows"] is True and report["shadow_seed"] == 10
    assert "above_shadow_share" in report["stability"][0]
    json.dumps(report, allow_nan=False)


def test_negative_shadow_seed_is_rejected_at_construction():
    with pytest.raises(ValueError, match="non-negative"):
        ranker(shadows=True, shadow_seed=-1)


# --- shadow hits test ---------------------------------------------------------

from scipy.stats import binom
from sklearn.model_selection import RepeatedKFold

from featranker.importance import _benjamini_hochberg

TEST_KEYS = {"shadow_hits", "shadow_p_value", "shadow_q_value"}


def _shares_split(index, shares):
    """A split whose consensus gives each group this above_shadow_share (None = absent)."""
    rows = [
        {"feature_group": g, "median_rank": float(i), "mean_rank": float(i),
         "above_shadow_share": s}
        for i, (g, s) in enumerate(shares.items(), start=1) if s is not None
    ]
    return {"split": index, "n_train": 4, "n_eval": 2,
            "report": {"feature_groups": {g: [g] for g in shares}, "consensus": rows}}


def test_hits_and_p_value_over_ten_splits():
    splits = [_shares_split(i, {"g": 1.0 if i < 9 else 0.0}) for i in range(10)]

    row = _build_stability(splits, 10, 1)[0]

    assert row["shadow_hits"] == 9
    assert row["shadow_p_value"] == pytest.approx(binom.sf(8, 10, 0.5))
    assert row["shadow_p_value"] == pytest.approx(0.0107, abs=1e-4)


def test_missing_split_is_a_non_hit_with_n_splits_trials():
    splits = [_shares_split(i, {"g": 1.0 if i else None}) for i in range(5)]

    row = _build_stability(splits, 5, 1)[0]

    assert row["shadow_hits"] == 4
    assert row["shadow_p_value"] == pytest.approx(binom.sf(3, 5, 0.5))


def test_share_of_exactly_half_is_not_a_hit():
    splits = [_shares_split(i, {"g": 0.5}) for i in range(4)]

    row = _build_stability(splits, 4, 1)[0]

    assert row["shadow_hits"] == 0 and row["shadow_p_value"] == 1.0


def test_benjamini_hochberg_example():
    assert _benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_benjamini_hochberg_caps_at_one_and_is_monotone():
    q = _benjamini_hochberg([1.0, 0.9, 0.001])

    assert q.max() <= 1.0 and q[2] == pytest.approx(0.003)


def test_stability_q_values_are_bh_across_groups():
    splits = [_shares_split(i, {"a": 1.0, "b": 1.0 if i < 8 else 0.0, "c": 0.0})
              for i in range(10)]

    rows = {r["feature_group"]: r for r in _build_stability(splits, 10, 1)}

    p = [rows[g]["shadow_p_value"] for g in "abc"]
    assert [rows[g]["shadow_q_value"] for g in "abc"] == pytest.approx(
        _benjamini_hochberg(p)
    )


def test_stability_has_no_test_keys_without_shadows():
    split = _split(0, 1.0)
    del split["report"]["consensus"][0]["above_shadow_share"]

    assert not TEST_KEYS & set(_build_stability([split], 1, 1)[0])


def test_resampled_report_has_shadow_test_block_for_cv5():
    report = ranker(shadows=True).rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=5, n_repeats=1,
        random_state=0)

    block = dict(report["shadow_test"])
    assert "not calibrated" in block.pop("note")
    assert block == {
        "method": "binomial", "hit": "above_shadow_share > 0.5",
        "null_probability": 0.5, "correction": "benjamini-hochberg",
        "n_trials": 5, "min_p_value": 0.03125, "calibrated": False,
    }
    assert all(TEST_KEYS <= set(row) for row in report["stability"])
    json.dumps(report, allow_nan=False)


def test_shadow_test_note_says_what_is_not_established():
    report = ranker(shadows=True).rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1,
        random_state=0)

    note = report["shadow_test"]["note"]
    assert report["shadow_test"]["calibrated"] is False
    assert "not calibrated" in note and "false-discovery" in note
    assert "arbitrary estimators" in note and "overlapping splits" in note


def test_resampled_report_without_shadows_has_no_test_keys():
    report = ranker().rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1,
        random_state=0)

    assert "shadow_test" not in report
    assert not TEST_KEYS & all_keys(report)


def test_planted_signal_gets_small_q_and_noise_large_q():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=60)

    report = ranker(shadows=True, shadow_seed=0).rank_resampled(
        X, y, feature_names=NAMES, scoring="r2", n_repeats=2, random_state=0,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=0))

    rows = {r["feature_group"]: r for r in report["stability"]}
    assert report["shadow_test"]["n_trials"] == 15
    assert rows["signal"]["shadow_q_value"] < 0.01
    assert rows["noise_a"]["shadow_q_value"] > 0.2
    assert rows["noise_b"]["shadow_q_value"] > 0.2


def test_benjamini_hochberg_ties_share_a_q_value_and_keep_order():
    q = _benjamini_hochberg([0.03125, 0.03125, 1.0, 0.03125])

    assert q == pytest.approx([0.03125 * 4 / 3] * 2 + [1.0, 0.03125 * 4 / 3])


def test_benjamini_hochberg_single_p_value_is_unchanged():
    assert _benjamini_hochberg([0.2]) == pytest.approx([0.2])


def test_fewer_split_reports_than_n_splits_still_uses_n_splits_trials():
    splits = [_shares_split(i, {"g": 1.0}) for i in range(3)]

    row = _build_stability(splits, 5, 1)[0]

    assert row["shadow_hits"] == 3
    assert row["shadow_p_value"] == pytest.approx(binom.sf(2, 5, 0.5))
