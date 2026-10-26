import json
import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from featranker import FeatureRanker

SHADOW_KEYS = {"shadows", "shadow_seed", "shadow_mode", "shadow_importance", "shadow_max",
               "above_shadow", "above_shadow_share", "family_above_shadow",
               "family_above_shadow_share", "shadow_families"}

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


def test_stability_share_counts_a_split_without_consensus_as_zero():
    rows = _build_stability([_split(0, 1.0), _split(1, 0.5), _split(2, None)], 3, 1)

    assert rows[0]["above_shadow_share"] == 0.5


def test_stability_share_of_one_lucky_split_in_ten_is_a_tenth():
    splits = [_split(0, 1.0)] + [_split(i, None) for i in range(1, 10)]

    assert _build_stability(splits, 10, 1)[0]["above_shadow_share"] == pytest.approx(0.1)


def test_stability_share_counts_failed_splits_as_zero():
    # Only 2 of 4 splits produced a report; the failed ones are absent from the list.
    rows = _build_stability([_split(0, 1.0), _split(2, 0.5)], 4, 1)

    assert rows[0]["above_shadow_share"] == 0.375


def test_stability_share_counts_a_group_missing_from_a_consensus_as_zero():
    splits = [_shares_split(0, {"a": 1.0, "b": 0.5}), _shares_split(1, {"a": 1.0, "b": None})]

    rows = {r["feature_group"]: r for r in _build_stability(splits, 2, 1)}

    assert rows["a"]["above_shadow_share"] == 1.0
    assert rows["b"]["above_shadow_share"] == 0.25


def test_stability_share_equals_the_plain_mean_when_every_split_votes():
    shares = [0.0, 1 / 3, 0.5, 0.7, 1.0, 0.1, 0.9]
    splits = [_split(i, s) for i, s in enumerate(shares)]

    assert _build_stability(splits, 7, 1)[0]["above_shadow_share"] == float(np.mean(shares))


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
        "experimental": True,
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


# --- per-family shadow evidence -------------------------------------------------

FAMILY_KEYS = {"family_above_shadow", "family_above_shadow_share", "shadow_families"}


def test_family_above_shadow_is_the_fraction_of_voting_members():
    rows = _build_consensus({
        "a1": _model("F1", True),
        "a2": _model("F1", True),
        "b1": _model("F2", False),
        "gated": _model("F3", True, in_consensus=False),
    })

    assert rows[0]["family_above_shadow"] == {"F1": 1.0, "F2": 0.0}
    assert rows[0]["above_shadow_share"] == 0.5


def test_above_shadow_share_is_the_mean_of_family_fractions_with_sorted_keys():
    rows = _build_consensus({
        "z1": _model("Z", True),
        "z2": _model("Z", False),
        "a1": _model("A", True),
    })

    families = rows[0]["family_above_shadow"]
    assert list(families) == ["A", "Z"]
    assert families == {"A": 1.0, "Z": 0.5}
    assert rows[0]["above_shadow_share"] == np.mean(list(families.values())) == 0.75


def test_no_family_evidence_without_shadows():
    model = {"family": "F", "in_consensus": True,
             "importance": {"g": {"mean": 0.5, "rank": 1}}}

    assert not FAMILY_KEYS & set(_build_consensus({"m": model})[0])


def _family_split(index, families):
    """A split whose consensus has one group g with these family fractions (None = no row)."""
    consensus = [] if families is None else [
        {"feature_group": "g", "median_rank": 1.0, "mean_rank": 1.0,
         "above_shadow_share": float(np.mean(list(families.values()))),
         "family_above_shadow": families}
    ]
    return {"split": index, "n_train": 4, "n_eval": 2,
            "report": {"feature_groups": {"g": ["g"]}, "consensus": consensus}}


def test_stability_family_share_divides_by_n_splits():
    splits = [
        _family_split(0, {"F1": 1.0, "F2": 1.0}),
        _family_split(1, {"F1": 1.0}),
        _family_split(2, {"F1": 0.0}),
    ]

    row = _build_stability(splits, 3, 1)[0]

    assert row["family_above_shadow_share"] == pytest.approx({"F1": 2 / 3, "F2": 1 / 3})
    assert row["shadow_families"] == 1


def test_a_family_that_rarely_votes_cannot_pass_alone():
    splits = [_family_split(i, {"F": 1.0} if i < 2 else None) for i in range(10)]

    row = _build_stability(splits, 10, 1)[0]

    assert row["family_above_shadow_share"] == pytest.approx({"F": 0.2})
    assert row["shadow_families"] == 0


def test_failed_splits_count_as_zero_in_family_share():
    splits = [_family_split(i, {"F": 1.0}) for i in range(3)]  # 3 reports of 5 splits

    row = _build_stability(splits, 5, 1)[0]

    assert row["family_above_shadow_share"] == pytest.approx({"F": 0.6})
    assert row["shadow_families"] == 1


def test_shadow_families_counts_shares_strictly_above_half():
    splits = [_family_split(0, {"A": 1.0, "B": 0.0}), _family_split(1, {"A": 0.0, "B": 1.0}),
              _family_split(2, {"A": 1.0, "B": 1.0, "C": 0.2})]

    row = _build_stability(splits, 3, 1)[0]

    assert list(row["family_above_shadow_share"]) == ["A", "B", "C"]
    assert row["family_above_shadow_share"]["A"] == pytest.approx(2 / 3)
    assert row["family_above_shadow_share"]["C"] == pytest.approx(0.2 / 3)
    assert row["shadow_families"] == 2  # A and B; C is below
    halves = _build_stability([_family_split(0, {"A": 1.0}), _family_split(1, {"A": 0.0})], 2, 1)
    assert halves[0]["family_above_shadow_share"] == {"A": 0.5}
    assert halves[0]["shadow_families"] == 0


def test_stability_has_no_family_keys_without_shadows():
    split = _split(0, 1.0)
    del split["report"]["consensus"][0]["above_shadow_share"]

    assert not FAMILY_KEYS & set(_build_stability([split], 1, 1)[0])


FAMILY_MODELS = {"lin": LinearRegression(), "ridge": Ridge()}


def test_resampled_report_has_family_evidence_that_matches_existing_columns():
    report = ranker(FAMILY_MODELS, shadows=True, shadow_seed=0).rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1, random_state=0)

    for split in report["splits"]:
        for row in split["report"]["consensus"]:
            fractions = row["family_above_shadow"]
            assert row["above_shadow_share"] == pytest.approx(np.mean(list(fractions.values())))
            assert list(fractions) == sorted(fractions)
    for row in report["stability"]:
        assert {"family_above_shadow_share", "shadow_families"} <= set(row)
        assert row["shadow_families"] == sum(
            v > 0.5 for v in row["family_above_shadow_share"].values())
    json.dumps(report, allow_nan=False)


def test_resampled_report_without_shadows_has_no_family_keys():
    report = ranker(FAMILY_MODELS).rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1, random_state=0)

    assert not all_keys(report) & SHADOW_KEYS


# --- shadow mode ----------------------------------------------------------------

from featranker.importance import _with_shadows

COPY_X = np.column_stack([X_DATA[:, 0], X_DATA[:, 0], X_DATA[:, 1]])  # e, f = e, noise


@pytest.mark.parametrize("mode", ["per-column", "Joint", "", None, 1, ["joint"]])
def test_invalid_shadow_mode_is_rejected_at_construction(mode):
    with pytest.raises(ValueError, match="shadow_mode"):
        ranker(shadow_mode=mode)
    with pytest.raises(ValueError, match="shadow_mode"):
        ranker(shadows=True, shadow_mode=mode)


@pytest.mark.parametrize("seed", [0, 7, [3, 1]])
def test_default_joint_mode_keeps_the_original_random_stream(seed):
    expected = np.hstack([X_DATA, X_DATA[np.random.default_rng(seed).permutation(40)]])

    np.testing.assert_array_equal(_with_shadows(X_DATA, seed), expected)
    np.testing.assert_array_equal(_with_shadows(X_DATA, seed, "joint"), expected)


def test_independent_shadows_are_per_column_permutations_from_one_generator():
    block = _with_shadows(COPY_X, 5, "independent")

    rng = np.random.default_rng(5)
    for column in range(3):
        np.testing.assert_array_equal(
            block[:, 3 + column], COPY_X[rng.permutation(40), column])
    np.testing.assert_array_equal(block[:, :3], COPY_X)
    np.testing.assert_array_equal(block, _with_shadows(COPY_X, 5, "independent"))  # seeded


def test_independent_mode_breaks_an_exact_copy_pair_that_joint_keeps():
    joint = _with_shadows(COPY_X, 5, "joint")
    independent = _with_shadows(COPY_X, 5, "independent")

    np.testing.assert_array_equal(joint[:, 3], joint[:, 4])
    assert not np.array_equal(independent[:, 3], independent[:, 4])
    np.testing.assert_array_equal(np.sort(independent[:, 3]), np.sort(COPY_X[:, 0]))
    np.testing.assert_array_equal(np.sort(independent[:, 4]), np.sort(COPY_X[:, 1]))


def test_fit_trains_on_independent_shadow_block():
    SEEN.clear()
    r = ranker({"rec": RecordingRegressor()}, shadows=True, shadow_seed=2,
               shadow_mode="independent")
    r.fit(X_DATA[:30], Y_DATA[:30], NAMES)

    np.testing.assert_array_equal(SEEN[0], _with_shadows(X_DATA[:30], 2, "independent"))


def test_eval_shadow_block_is_independent_per_column_from_eval_rows():
    SEEN.clear()
    run(ranker({"rec": PredictRecorder()}, shadows=True, shadow_seed=1,
               shadow_mode="independent"), n_repeats=1)

    unpermuted = SEEN[1]
    np.testing.assert_array_equal(unpermuted, _with_shadows(X_DATA[30:], [1, 1], "independent"))


def test_report_records_shadow_mode_only_with_shadows():
    off = run(ranker(shadow_mode="independent"))
    default = run(ranker(shadows=True, shadow_seed=0))
    independent = run(ranker(shadows=True, shadow_seed=0, shadow_mode="independent"))

    assert "shadow_mode" not in all_keys(off)
    assert default["shadow_mode"] == "joint"
    assert independent["shadow_mode"] == "independent"
    assert default == run(ranker(shadows=True, shadow_seed=0, shadow_mode="joint"))


def test_independent_seeded_runs_are_reproducible():
    def make():
        return run(ranker(shadows=True, shadow_seed=3, shadow_mode="independent"))

    assert make() == make()


def test_ranker_pickled_without_shadow_mode_loads_as_joint():
    r = ranker(shadows=True, shadow_seed=0)
    r.fit(X_DATA[:30], Y_DATA[:30], NAMES)
    del r.__dict__["shadow_mode"]  # what a 1.5 pickle holds

    loaded = pickle.loads(pickle.dumps(r))
    report = loaded.rank_features(X_DATA[30:], Y_DATA[30:], scoring="r2", n_repeats=1)

    assert loaded.shadow_mode == "joint" and report["shadow_mode"] == "joint"


def test_resampled_passes_shadow_mode_to_every_split():
    report = ranker(shadows=True, shadow_seed=0, shadow_mode="independent").rank_resampled(
        X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1, random_state=0)

    assert report["shadow_mode"] == "independent"
    assert [s["report"]["shadow_mode"] for s in report["splits"]] == ["independent"] * 3
    assert "shadow_mode" not in all_keys(
        ranker().rank_resampled(X_DATA, Y_DATA, feature_names=NAMES, scoring="r2", cv=3,
                                n_repeats=1, random_state=0))
