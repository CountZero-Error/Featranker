import json

import numpy as np
import pytest

from featranker.importance import _build_stability


def _split(index, scores):
    """A split entry whose consensus has these (median_rank, mean_rank) scores."""
    rows = [
        {"feature_group": group, "median_rank": score[0], "mean_rank": score[1]}
        for group, score in scores.items()
    ]
    rows.sort(key=lambda row: (row["median_rank"], row["mean_rank"], row["feature_group"]))
    groups = {name: [name] for name in ("w", "x", "y", "z")}
    return {"split": index, "n_train": 4, "n_eval": 2,
            "report": {"feature_groups": groups, "consensus": rows}}


def _ordered(index, order):
    """A split whose consensus lists groups in this order with distinct scores."""
    return _split(index, {g: (float(i + 1), float(i + 1)) for i, g in enumerate(order)})


def _rows(split_reports, n_splits, top_k):
    return {
        row["feature_group"]: row
        for row in _build_stability(split_reports, n_splits, top_k)
    }


def test_stability_uses_consensus_positions():
    splits = [_ordered(0, "xyzw"), _ordered(1, "yxzw"), _ordered(2, "yzwx")]

    row = _rows(splits, n_splits=3, top_k=2)["x"]

    assert row["split_ranks"] == [1, 2, 4]
    assert row["median_rank"] == 2.0
    assert row["min_rank"] == 1 and row["max_rank"] == 4
    assert row["n_splits_ranked"] == 3
    assert row["top_k_share"] == pytest.approx(2 / 3)


def test_position_ignores_score_scale():
    # x is first by position even though its median_rank score is 4.5
    splits = [_split(0, {"x": (4.5, 4.0), "y": (5.0, 5.0), "z": (6.0, 6.0), "w": (7.0, 7.0)})]

    assert _rows(splits, n_splits=1, top_k=1)["x"]["split_ranks"] == [1]


def test_ties_on_both_scores_share_the_better_position():
    splits = [_split(0, {"x": (1.0, 1.0), "y": (2.0, 2.0), "z": (2.0, 2.0), "w": (2.0, 3.0)})]

    rows = _rows(splits, n_splits=1, top_k=2)

    assert [rows[g]["split_ranks"][0] for g in "xyzw"] == [1, 2, 2, 4]
    assert rows["z"]["top_k_share"] == 1.0


def test_split_with_empty_consensus_is_null_and_not_top_k():
    splits = [_ordered(0, "xyzw"), _split(1, {}), _ordered(2, "xyzw")]

    row = _rows(splits, n_splits=3, top_k=1)["x"]

    assert row["split_ranks"] == [1, None, 1]
    assert row["n_splits_ranked"] == 2
    assert row["top_k_share"] == pytest.approx(2 / 3)


def test_no_split_with_consensus_warns_and_is_empty():
    with pytest.warns(UserWarning, match="No split"):
        assert _build_stability([_split(0, {}), _split(1, {})], 2, 1) == []


def test_failed_splits_lower_top_k_share():
    row = _rows([_ordered(0, "xyzw")], n_splits=4, top_k=1)["x"]

    assert row["top_k_share"] == 0.25


def test_stability_sort_order():
    splits = [_ordered(0, "xyzw"), _ordered(1, "xyzw"), _ordered(2, "yxzw")]

    # x: [1,1,2] median 1; y: [2,2,1] median 2; z: [3,3,3]; w: [4,4,4]
    stability = _build_stability(splits, 3, 1)
    assert [r["feature_group"] for r in stability] == ["x", "y", "z", "w"]
    json.dumps(stability, allow_nan=False)


def test_equal_median_rank_sorts_by_higher_top_k_share_before_name():
    x_first = {"x": (1.0, 1.0), "y": (2.0, 2.0), "w": (3.0, 3.0), "z": (4.0, 4.0)}
    w_x_tied = {"y": (1.0, 1.0), "z": (2.0, 2.0), "w": (3.0, 3.0), "x": (3.0, 3.0)}
    splits = [_split(i, x_first if i < 2 else w_x_tied) for i in range(5)]

    rows = _rows(splits, n_splits=5, top_k=1)
    # both median 3; x is top-1 in 2/5 splits, w in 0/5 -> x before w despite name
    assert rows["x"]["split_ranks"] == [1, 1, 3, 3, 3]
    assert rows["w"]["split_ranks"] == [3, 3, 3, 3, 3]
    names = [r["feature_group"] for r in _build_stability(splits, 5, 1)]
    assert names.index("x") < names.index("w")

from sklearn.model_selection import GroupKFold

from featranker.importance import _resolve_splits

X12 = np.arange(24.0).reshape(12, 2)


def test_integer_cv_is_stratified_for_classification():
    y = np.array(["a"] * 8 + ["b"] * 4)

    pairs = _resolve_splits(3, X12, y, None, "clf", 0)

    assert len(pairs) == 3
    assert all(set(y[test]) == {"a", "b"} for _, test in pairs)


def test_integer_cv_is_shuffled_and_seeded_for_regression():
    y = np.arange(12.0)

    first = _resolve_splits(3, X12, y, None, "reg", 7)
    again = _resolve_splits(3, X12, y, None, "reg", 7)

    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, again))
    assert not np.array_equal(first[0][1], np.arange(4))  # shuffled, not 0..3


def test_group_splitter_keeps_groups_disjoint():
    groups = np.repeat([0, 1, 2, 3], 3)

    pairs = _resolve_splits(GroupKFold(2), X12, np.arange(12.0), groups, "reg", None)

    assert all(not set(groups[tr]) & set(groups[te]) for tr, te in pairs)


def test_explicit_index_pairs():
    pairs = _resolve_splits([([0, 1, 2, 3], [4, 5])], X12, np.arange(12.0), None, "reg", None)

    assert len(pairs) == 1
    assert pairs[0][0].tolist() == [0, 1, 2, 3]
    assert pairs[0][1].tolist() == [4, 5]


@pytest.mark.parametrize(
    "cv",
    [1, 0, True, "five", 2.5, [([0, 1], [])], [([], [2])], [([0, 1], [99])],
     [([0.5, 1.0], [2])], [], [(1, 2, 3)]],
)
def test_invalid_cv_raises_value_error(cv):
    with pytest.raises(ValueError, match="cv|split"):
        _resolve_splits(cv, X12, np.arange(12.0), None, "reg", None)


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge

from featranker import FeatureRanker

FIT_ROWS = []


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Linear fit that records the first-column values of its training rows."""

    def fit(self, X, y):
        FIT_ROWS.append(sorted(X[:, 0].tolist()))
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class FailOnMarker(RecordingRegressor):
    """Fails whenever the marker row (first column == 0) is in the training rows."""

    def fit(self, X, y):
        if (X[:, 0] == 0).any():
            raise ValueError("marker row in training data")
        return super().fit(X, y)


class AlwaysFail(RecordingRegressor):
    def fit(self, X, y):
        raise ValueError("always fails")


RNG = np.random.default_rng(0)
X_DATA = np.column_stack([np.arange(30.0), RNG.normal(size=30), RNG.normal(size=30)])
Y_DATA = 2.0 * X_DATA[:, 0] + 0.1 * RNG.normal(size=30)
NAMES = ["signal", "noise_a", "noise_b"]


def ranker(models=None, **kwargs):
    return FeatureRanker(
        task="reg", models=models or {"ridge": Ridge()}, verbose=False, **kwargs
    )


def resample(r, **kwargs):
    kwargs = {"scoring": "r2", "cv": 3, "n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_resampled(X_DATA, Y_DATA, feature_names=NAMES, **kwargs)


def test_fitted_ranker_state_is_untouched():
    r = ranker().fit(X_DATA, Y_DATA, NAMES)
    trained, names = r.trained_models, r.feature_names_
    fitted_model = trained["ridge"]

    resample(r)

    assert r.trained_models is trained and r.trained_models["ridge"] is fitted_model
    assert r.feature_names_ is names and r.is_fitted_ is True


def test_unfitted_ranker_works_and_stays_unfitted():
    r = ranker()

    report = resample(r)

    assert report["n_splits"] == 3 and len(report["splits"]) == 3
    assert r.is_fitted_ is False


def test_each_split_fits_only_on_its_training_rows():
    FIT_ROWS.clear()
    train = [list(range(0, 20)), list(range(10, 30))]
    evaluation = [list(range(20, 30)), list(range(0, 10))]

    report = resample(ranker({"rec": RecordingRegressor()}),
                      cv=list(zip(train, evaluation)))

    assert FIT_ROWS == [[float(i) for i in rows] for rows in train]
    assert [(s["n_train"], s["n_eval"]) for s in report["splits"]] == [(20, 10), (20, 10)]


def test_split_seeds_are_random_state_plus_index():
    report = resample(ranker(), random_state=5)

    assert [s["report"]["random_state"] for s in report["splits"]] == [5, 6, 7]


def test_reproducible_with_seeded_builtin_models():
    def run():
        r = FeatureRanker(task="reg", group="linear", verbose=False)
        with _quiet():
            return resample(r)

    assert run() == run()


class _quiet:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def test_failing_split_is_recorded_and_run_continues():
    pairs = [(list(range(1, 20)), list(range(20, 30))),   # no marker -> ok
             (list(range(0, 20)), list(range(20, 30))),   # marker -> fails
             (list(range(10, 30)), list(range(0, 10)))]   # no marker -> ok

    report = resample(ranker({"m": FailOnMarker()}), cv=pairs)

    assert report["n_splits"] == 3
    assert [s["split"] for s in report["splits"]] == [0, 2]
    split_failures = [f for f in report["failures"] if f["stage"] == "split"]
    assert len(split_failures) == 1
    assert split_failures[0]["split"] == 1 and split_failures[0]["model"] is None
    assert split_failures[0]["error_type"] == "RuntimeError"


def test_all_splits_failing_raises():
    with pytest.raises(RuntimeError, match="always fails"):
        resample(ranker({"bad": AlwaysFail()}))


def test_model_failures_carry_split_index():
    report = resample(ranker({"ok": Ridge(), "bad": AlwaysFail()}))

    fit_failures = [f for f in report["failures"] if f["stage"] == "fit"]
    assert sorted(f["split"] for f in fit_failures) == [0, 1, 2]
    assert all(f["model"] == "bad" for f in fit_failures)


def test_dataframe_column_names_are_kept():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X_DATA, columns=NAMES)

    report = ranker().rank_resampled(frame, Y_DATA, scoring="r2", cv=3, n_repeats=1)

    assert {row["feature_group"] for row in report["stability"]} == set(NAMES)


def test_report_shape_and_strict_json():
    report = resample(ranker(), top_k=1)

    assert set(report) == {
        "evaluation_mode", "scoring", "n_repeats", "random_state", "feature_groups",
        "provenance", "top_k", "n_splits", "splits", "stability", "failures",
    }
    assert report["evaluation_mode"] == "resampled"
    assert report["top_k"] == 1
    assert report["stability"][0]["feature_group"] == "signal"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [{"cv": 1}, {"cv": "five"}, {"top_k": 0}, {"top_k": True}, {"top_k": 1.5},
     {"n_repeats": 0}, {"feature_groups": {"g": ["missing"]}}],
)
def test_invalid_arguments_fail_before_any_fit(kwargs):
    FIT_ROWS.clear()

    with pytest.raises(ValueError):
        resample(ranker({"rec": RecordingRegressor()}), **kwargs)

    assert FIT_ROWS == []


def test_overlapping_train_and_eval_rows_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        _resolve_splits([([0, 1, 2, 3], [3, 4])], X12, np.arange(12.0), None, "reg", None)


def test_overlap_across_splits_is_allowed():
    pairs = [([0, 1, 2], [3, 4]), ([5, 6, 7], [3, 4])]  # same eval rows, like RepeatedKFold

    assert len(_resolve_splits(pairs, X12, np.arange(12.0), None, "reg", None)) == 2


def test_integer_cv_stratifies_fractional_class_labels():
    y = np.tile([0.1, 0.2], 6)

    pairs = _resolve_splits(2, X12, y, None, "clf", 0)

    assert all(set(y[test]) == {0.1, 0.2} for _, test in pairs)
