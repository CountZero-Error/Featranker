import json
import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, ShuffleSplit

from featranker import ConsensusSelector, FeatureRanker
from featranker.importance import _shuffle_index

N_TRAIN, N_EVAL = 12, 12
RNG = np.random.default_rng(0)
# Column values are unique row ids, so a shuffled column shows where each value went.
X_DATA = np.column_stack(
    [np.arange(24.0), np.arange(24.0) * 10 + 1, RNG.normal(size=24)]
)
Y_DATA = 3.0 * X_DATA[:, 0] + 0.1 * RNG.normal(size=24)
NAMES = ["a", "b", "c"]
LABELS = np.repeat([0, 1, 2], 4)
FIT_SEEN, PREDICT_SEEN = [], []


class Recorder(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        FIT_SEEN.append(np.array(X))
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        PREDICT_SEEN.append(np.array(X))
        return self.model_.predict(X)


def ranker(**kwargs):
    return FeatureRanker(
        task="reg", models={"rec": Recorder()}, verbose=False, **kwargs
    )


def run(r, shuffle_groups=None, fit_groups=None, **kwargs):
    FIT_SEEN.clear()
    PREDICT_SEEN.clear()
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES, shuffle_groups=fit_groups)
    kwargs = {"scoring": "r2", "n_repeats": 3, "random_state": 0, **kwargs}
    return r.rank_features(
        X_DATA[N_TRAIN:], Y_DATA[N_TRAIN:], shuffle_groups=shuffle_groups, **kwargs
    )


def permuted_columns(column):
    """Every predict-time version of eval column `column` that differs from the original."""
    original = X_DATA[N_TRAIN:, column]
    return [m[:, column] for m in PREDICT_SEEN if not np.array_equal(m[:, column], original)]


@pytest.mark.parametrize("value", ["group", "ROWS", None, 1, ["rows"]])
def test_invalid_shuffle_raises(value):
    with pytest.raises(ValueError, match="shuffle"):
        ranker(shuffle=value)


def test_default_is_rows_and_report_has_no_shuffle_key():
    plain = run(ranker(shadows=True, shadow_seed=1))
    explicit = run(ranker(shadows=True, shadow_seed=1, shuffle="rows"))

    assert "shuffle" not in plain and "shuffle" not in explicit
    assert plain["models"] == explicit["models"]


def test_rows_index_is_the_plain_permutation():
    drawn = _shuffle_index(np.random.default_rng(3), 10, "rows", None)

    assert np.array_equal(drawn, np.random.default_rng(3).permutation(10))


def test_within_groups_permutation_keeps_values_in_their_group():
    report = run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS)

    columns = permuted_columns(0)
    assert columns
    for column in columns:
        for label in (0, 1, 2):
            rows = LABELS == label
            assert sorted(column[rows]) == sorted(X_DATA[N_TRAIN:, 0][rows])
    assert report["shuffle"] == "within_groups"


@pytest.mark.parametrize("shadow_mode", ["joint", "independent"])
def test_within_groups_shadows_keep_values_in_their_group(shadow_mode):
    r = ranker(
        shuffle="within_groups", shadows=True, shadow_seed=0, shadow_mode=shadow_mode
    )
    run(r, shuffle_groups=LABELS, fit_groups=LABELS)

    for matrix in FIT_SEEN:
        for column in range(3):
            shadow = matrix[:, 3 + column]
            for label in (0, 1, 2):
                rows = LABELS == label
                assert sorted(shadow[rows]) == sorted(matrix[rows, column])
        assert not np.array_equal(matrix[:, :3], matrix[:, 3:])
    # The evaluation shadows are shuffled by the evaluation labels too.
    shadowed = [m for m in PREDICT_SEEN if m.shape[1] == 6]
    assert shadowed
    for matrix in shadowed:
        for label in (0, 1, 2):
            rows = LABELS == label
            assert sorted(matrix[rows, 3]) == sorted(matrix[rows, 0])


def test_single_row_groups_keep_their_values():
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    run(ranker(shuffle="within_groups"), shuffle_groups=labels, fit_groups=LABELS)

    original = X_DATA[N_TRAIN:, 0]
    columns = permuted_columns(0)
    assert columns
    for column in columns:
        assert np.array_equal(column[8:], original[8:])
        assert not np.array_equal(column[:8], original[:8])


def test_all_singleton_groups_raise():
    with pytest.raises(ValueError, match="group"):
        run(ranker(shuffle="within_groups"), shuffle_groups=np.arange(N_EVAL), fit_groups=LABELS)
    with pytest.raises(ValueError, match="group"):
        run(
            ranker(shuffle="within_groups", shadows=True),
            shuffle_groups=LABELS,
            fit_groups=np.arange(N_TRAIN),
        )


def test_all_singleton_groups_after_subsampling_raise():
    labels = np.arange(N_EVAL)
    labels[1] = 0  # one pair; the 2 sampled rows (seed 0) are not that pair
    with pytest.raises(ValueError, match="group"):
        run(
            ranker(shuffle="within_groups"), shuffle_groups=labels, fit_groups=LABELS,
            max_samples=2, random_state=0,
        )


@pytest.mark.parametrize(
    "bad",
    [[0, None] * 6, [0.0, np.nan] * 6, np.array([0, "a"] * 6, dtype=object), [0, 1.5, "a", None] * 3],
    ids=["none", "nan", "mixed", "mixed-none"],
)
def test_unusable_labels_raise_value_error(bad):
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups"), shuffle_groups=bad, fit_groups=LABELS)
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=bad)


def test_string_labels_work():
    labels = np.repeat(["x", "y", "z"], 4)

    report = run(ranker(shuffle="within_groups"), shuffle_groups=labels, fit_groups=labels)

    assert report["shuffle"] == "within_groups"


def test_singleton_labels_pass_fit_without_shadows():
    # Labels are unused by fit() then; the all-singleton check is for shadows only.
    run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=np.arange(N_TRAIN))


@pytest.mark.parametrize(
    "bad", [LABELS[:5], np.tile(LABELS, 2), LABELS.reshape(3, 4)], ids=["short", "long", "2d"]
)
def test_mismatched_labels_raise(bad):
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups"), shuffle_groups=bad, fit_groups=LABELS)
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=bad)


def test_missing_labels_raise():
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups", shadows=True), shuffle_groups=LABELS)
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle="within_groups"), fit_groups=LABELS)


@pytest.mark.parametrize("shuffle", ["rows", "circular"])
def test_labels_with_another_mode_raise(shuffle):
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle=shuffle), fit_groups=LABELS)
    with pytest.raises(ValueError, match="shuffle_groups"):
        run(ranker(shuffle=shuffle), shuffle_groups=LABELS)


def test_within_groups_with_subsample_slices_the_labels():
    report = run(
        ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS,
        max_samples=8, random_state=1,
    )

    assert report["n_eval_rows"] == 8
    moved = [m for m in PREDICT_SEEN if not np.array_equal(m[:, 0], (m[:, 1] - 1) / 10)]
    assert moved
    for m in moved:
        ids = ((m[:, 1] - 1) / 10).astype(int)
        for label in set(LABELS[ids - N_TRAIN]):
            rows = LABELS[ids - N_TRAIN] == label
            assert sorted(m[rows, 0]) == sorted(ids[rows])


def sampled_ids():
    """Eval row ids (column 0 values) of every predict call, and its permuted column 0."""
    return [((m[:, 1] - 1) / 10).astype(int) for m in PREDICT_SEEN]


def test_within_groups_subsample_keeps_whole_groups():
    report = run(
        ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS,
        max_samples=0.25, random_state=1,
    )

    groups = [set(range(12, 16)), set(range(16, 20)), set(range(20, 24))]
    assert report["n_eval_rows"] == 4
    ids = sampled_ids()
    assert ids
    assert all(set(i) in groups for i in ids)
    moved = [m for m in PREDICT_SEEN if not np.array_equal(m[:, 0], (m[:, 1] - 1) / 10)]
    assert moved
    for m in moved:
        assert sorted(m[:, 0]) == sorted((m[:, 1] - 1) / 10)


def test_within_groups_subsample_may_exceed_the_target_by_whole_groups():
    report = run(
        ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS,
        max_samples=5, random_state=1,
    )

    assert report["n_eval_rows"] == 8  # two whole groups cover 5 rows


def test_circular_subsample_keeps_row_order():
    run(ranker(shuffle="circular"), max_samples=8, random_state=1)

    originals = [m for m in PREDICT_SEEN if len(m) == 8]
    assert originals
    for m in originals:
        assert any(np.all(np.diff(np.roll(m[:, 1], -k)) > 0) for k in range(8))


def test_circular_permutation_is_a_roll():
    report = run(ranker(shuffle="circular"))

    original = X_DATA[N_TRAIN:, 0]
    columns = permuted_columns(0)
    assert columns
    for column in columns:
        assert any(np.array_equal(column, np.roll(original, k)) for k in range(1, N_EVAL))
    assert report["shuffle"] == "circular"


def test_circular_offsets_cover_one_to_n_minus_one():
    offsets = {
        int(np.argmax(_shuffle_index(np.random.default_rng(seed), 5, "circular", None) == 0))
        for seed in range(200)
    }

    assert offsets == {1, 2, 3, 4}


def test_circular_joint_shadows_share_one_offset_and_independent_draw_each():
    joint = ranker(shuffle="circular", shadows=True, shadow_seed=0)
    run(joint)
    matrix = FIT_SEEN[0]
    originals = matrix[:, :3]
    ks = [
        next(k for k in range(1, N_TRAIN) if np.array_equal(matrix[:, 3 + j], np.roll(originals[:, j], k)))
        for j in range(3)
    ]
    assert len(set(ks)) == 1

    independent = ranker(
        shuffle="circular", shadows=True, shadow_seed=0, shadow_mode="independent"
    )
    run(independent)
    matrix = FIT_SEEN[0]
    originals = matrix[:, :3]
    ks = [
        next(k for k in range(1, N_TRAIN) if np.array_equal(matrix[:, 3 + j], np.roll(originals[:, j], k)))
        for j in range(3)
    ]
    assert len(set(ks)) > 1


def test_circular_with_fewer_than_two_rows_raises():
    r = ranker(shuffle="circular")
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES)
    with pytest.raises(ValueError, match="2 rows"):
        r.rank_features(X_DATA[12:13], Y_DATA[12:13], scoring="r2")
    with pytest.raises(ValueError, match="2 rows"):
        ranker(shuffle="circular", shadows=True).fit(X_DATA[:1], Y_DATA[:1], NAMES)


def test_shuffle_changes_the_numbers_but_not_the_evidence_keys():
    plain = run(ranker(evidence=True), random_state=0)
    circular = run(ranker(evidence=True, shuffle="circular"), random_state=0)

    assert plain["evidence"]["correlation"] == circular["evidence"]["correlation"]
    assert plain["models"]["rec"]["importance"] != circular["models"]["rec"]["importance"]


def test_works_with_weights_scorers_and_shadows():
    r = FeatureRanker(
        task="reg", models={"lin": LinearRegression()}, verbose=False,
        shuffle="within_groups", shadows=True, shadow_seed=0,
    )
    weights = np.linspace(1, 2, N_EVAL)
    r.fit(X_DATA[:N_TRAIN], Y_DATA[:N_TRAIN], NAMES, sample_weight=weights, shuffle_groups=LABELS)
    report = r.rank_features(
        X_DATA[N_TRAIN:], Y_DATA[N_TRAIN:], scoring=["r2", "neg_mean_absolute_error"],
        shuffle_groups=LABELS, sample_weight=weights, n_repeats=2, random_state=0,
    )

    assert report["shuffle"] == "within_groups"
    assert report["models"]["lin"]["secondary"]["neg_mean_absolute_error"] is not None
    assert "shadow_max" in report["models"]["lin"]
    json.dumps(dict(report), allow_nan=False)


def test_report_is_strict_json_for_every_mode():
    for r, kwargs in (
        (ranker(shuffle="circular"), {}),
        (ranker(shuffle="within_groups"), {"shuffle_groups": LABELS, "fit_groups": LABELS}),
    ):
        json.dumps(dict(run(r, **kwargs)), allow_nan=False)


def test_ranker_pickled_before_shuffle_loads_as_rows():
    r = ranker()
    del r.__dict__["shuffle"]  # what a 1.14 pickle holds

    loaded = pickle.loads(pickle.dumps(r))

    assert loaded.shuffle == "rows"
    assert "shuffle" not in run(loaded)


def test_circular_works_in_parallel():
    serial = run(ranker(shuffle="circular", n_jobs=1))
    parallel = run(ranker(shuffle="circular", n_jobs=2))

    assert parallel["shuffle"] == "circular"
    assert parallel["models"] == serial["models"]


# --- resampling and selector ---------------------------------------------------

GROUPS = np.repeat(np.arange(6), 4)


def resampled(r, **kwargs):
    FIT_SEEN.clear()
    PREDICT_SEEN.clear()
    kwargs = {"scoring": "r2", "cv": GroupKFold(3), "n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_resampled(X_DATA, Y_DATA, feature_names=NAMES, **kwargs)


def test_rank_resampled_slices_groups_per_split():
    r = ranker(shuffle="within_groups", shadows=True, shadow_seed=0)

    report = resampled(r, groups=GROUPS)

    assert report["shuffle"] == "within_groups"
    assert all(s["report"]["shuffle"] == "within_groups" for s in report["splits"])
    assert not report["failures"]
    # Shadows (fit) kept the values of each training group together.
    for matrix in FIT_SEEN:
        ids = matrix[:, 0]
        group_of = {v: GROUPS[int(v)] for v in ids}
        for column in range(3):
            for g in set(group_of.values()):
                rows = np.array([group_of[v] == g for v in ids])
                assert sorted(matrix[rows, 3 + column]) == sorted(matrix[rows, column])


def test_rank_resampled_circular_sorts_each_split_before_rolling():
    cv = ShuffleSplit(3, test_size=0.25, random_state=0)  # unordered indices

    resampled(ranker(shuffle="circular"), cv=cv)

    assert FIT_SEEN
    for matrix in FIT_SEEN:
        assert np.all(np.diff(matrix[:, 0]) > 0)
    for matrix in PREDICT_SEEN:
        ids = matrix[:, 1]  # the column group "b" is the one never rolled with "a"
        assert any(np.all(np.diff(np.roll(ids, -k)) > 0) for k in range(len(ids)))


def test_rank_resampled_within_groups_needs_groups():
    with pytest.raises(ValueError, match="groups"):
        resampled(ranker(shuffle="within_groups"), cv=3)
    with pytest.raises(ValueError, match="groups"):
        resampled(ranker(shuffle="within_groups"), groups=GROUPS[:5])


def test_rank_resampled_circular_and_default_keys():
    circular = resampled(ranker(shuffle="circular"), cv=3)
    plain = resampled(ranker(), cv=3)

    assert circular["shuffle"] == "circular"
    assert all(s["report"]["shuffle"] == "circular" for s in circular["splits"])
    assert "shuffle" not in plain
    assert all("shuffle" not in s["report"] for s in plain["splits"])
    json.dumps(dict(circular), allow_nan=False)


def test_selector_with_groups():
    selector = ConsensusSelector(
        "reg", "r2", k=2, shuffle="within_groups", cv=GroupKFold(3), group="fast",
        n_repeats=2, random_state=0,
    )

    selector.fit(X_DATA, Y_DATA, groups=GROUPS)

    assert selector.report_["shuffle"] == "within_groups"
    assert all(s["report"]["shuffle"] == "within_groups" for s in selector.report_["splits"])


def test_selector_with_shadows_and_circular():
    selector = ConsensusSelector(
        "reg", "r2", min_shadow_share=0.0, shuffle="circular", cv=3, group="fast",
        n_repeats=2, random_state=0, shadow_seed=0,
    )

    selector.fit(X_DATA, Y_DATA)

    assert selector.report_["shuffle"] == "circular"


def test_selector_without_groups_raises_and_clones():
    from sklearn.base import clone

    selector = ConsensusSelector("reg", "r2", k=1, shuffle="within_groups", cv=3)
    copy = clone(selector)

    assert copy.get_params()["shuffle"] == "within_groups"
    with pytest.raises(ValueError, match="groups"):
        copy.fit(X_DATA, Y_DATA)


def test_selector_pickled_before_shuffle_loads_as_rows():
    selector = ConsensusSelector("reg", "r2", k=1)
    state = dict(selector.__dict__)
    del state["shuffle"]  # what a 1.14 pickle holds
    loaded = ConsensusSelector.__new__(ConsensusSelector)
    loaded.__setstate__(state)

    assert loaded.shuffle == "rows"


# --- run record and multi-column groups ---------------------------------------


def test_changing_only_the_labels_changes_the_run_block():
    other = np.repeat([0, 1, 2], 4)[::-1].copy()
    other[0], other[5] = other[5], other[0]
    first = run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS)
    again = run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS)
    eval_changed = run(ranker(shuffle="within_groups"), shuffle_groups=other, fit_groups=LABELS)
    train_changed = run(ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=other)

    assert first["run"] == again["run"]
    assert first["run"]["shuffle_groups_sha256"] != eval_changed["run"]["shuffle_groups_sha256"]
    assert first["run"]["train_shuffle_groups_sha256"] == eval_changed["run"]["train_shuffle_groups_sha256"]
    assert first["run"]["train_shuffle_groups_sha256"] != train_changed["run"]["train_shuffle_groups_sha256"]
    assert first["run"] != eval_changed["run"] and first["run"] != train_changed["run"]


@pytest.mark.parametrize("shuffle", ["rows", "circular"])
def test_run_block_has_no_label_fingerprint_without_labels(shuffle):
    report = run(ranker(shuffle=shuffle))

    assert not [key for key in report["run"] if "shuffle_groups" in key]


def test_rank_resampled_run_block_fingerprints_groups():
    r = ranker(shuffle="within_groups")
    first = resampled(r, groups=GROUPS)
    other = resampled(r, groups=np.roll(GROUPS, 1))

    assert first["run"]["shuffle_groups_sha256"] != other["run"]["shuffle_groups_sha256"]
    assert "shuffle_groups_sha256" not in resampled(ranker(), cv=3)["run"]


def test_multi_column_feature_group_moves_together_inside_label_groups():
    run(
        ranker(shuffle="within_groups"), shuffle_groups=LABELS, fit_groups=LABELS,
        feature_groups={"ab": ["a", "b"]},
    )

    moved = [m for m in PREDICT_SEEN if not np.array_equal(m[:, 0], X_DATA[N_TRAIN:, 0])]
    assert moved
    for m in moved:
        # b = 10 * a + 1 in the data, so one row index for both columns keeps it true.
        assert np.array_equal(m[:, 1], m[:, 0] * 10 + 1)
        for label in (0, 1, 2):
            rows = LABELS == label
            assert sorted(m[rows, 0]) == sorted(X_DATA[N_TRAIN:, 0][rows])
