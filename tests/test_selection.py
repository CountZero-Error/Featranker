import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge

from featranker import ConsensusSelector, FeatureRanker

RNG = np.random.default_rng(0)
X_DATA = np.column_stack([np.arange(60.0), RNG.normal(size=60), RNG.normal(size=60)])
Y_DATA = 2.0 * X_DATA[:, 0] + 0.1 * RNG.normal(size=60)


def selector(**kwargs):
    kwargs = {"k": 1, "models": {"ridge": Ridge()}, "cv": 2, "n_repeats": 1,
              "random_state": 0, **kwargs}
    return ConsensusSelector("reg", "r2", **kwargs)


def test_parameters_are_stored_raw_and_clone_round_trips():
    s = ConsensusSelector("reg", "r2", k=3, cv=4, min_shadow_share=0.5, max_shadow_q=0.05)

    assert s.k == 3 and s.cv == 4 and s.min_shadow_share == 0.5 and s.max_shadow_q == 0.05
    assert clone(s).get_params() == s.get_params()
    assert s.set_params(k=5).k == 5


@pytest.mark.parametrize(
    "kwargs",
    [{"k": None}, {"k": 0}, {"k": 2.5}, {"k": True},
     {"k": None, "min_shadow_share": 1.5}, {"k": None, "min_shadow_share": -0.1},
     {"k": None, "min_shadow_share": "half"}, {"k": None, "min_shadow_share": float("nan")},
     {"k": 1, "max_shadow_q": 0}, {"k": 1, "max_shadow_q": 1.5}, {"k": 1, "max_shadow_q": -0.1},
     {"k": 1, "max_shadow_q": "low"}, {"k": 1, "max_shadow_q": float("nan")},
     {"k": 1, "max_shadow_q": True}],
)
def test_invalid_rule_raises_before_ranking(kwargs, monkeypatch):
    def fail(*args, **kw):
        raise AssertionError("ranking must not start")

    monkeypatch.setattr(FeatureRanker, "rank_resampled", fail)

    with pytest.raises(ValueError):
        selector(**kwargs).fit(X_DATA, Y_DATA)


CALLS = []


def fake_resampled(stability, groups=None):
    """Patch rank_resampled to return this stability table and record its call."""

    def rank_resampled(self, X, y, **kwargs):
        CALLS.append({"shadows": self.shadows, "shadow_seed": self.shadow_seed,
                      "X": np.asarray(X), "y": np.asarray(y), **kwargs})
        names = kwargs.get("feature_names") or list(X.columns)
        return {
            "evaluation_mode": "resampled",
            "feature_groups": groups or {n: [n] for n in names},
            "stability": stability,
        }

    return rank_resampled


def _rows(order, shares=None, qs=None):
    return [
        {"feature_group": g,
         **({} if shares is None else {"above_shadow_share": s}),
         **({} if qs is None else {"shadow_q_value": q})}
        for g, s, q in zip(order, shares or [None] * len(order), qs or [None] * len(order))
    ]


@pytest.fixture
def patched(monkeypatch):
    CALLS.clear()

    def patch(stability, groups=None):
        monkeypatch.setattr(FeatureRanker, "rank_resampled", fake_resampled(stability, groups))

    return patch


def test_report_comes_from_exactly_the_fit_data(patched):
    patched(_rows(["x0", "x1", "x2"]))

    s = selector(k=2).fit(X_DATA, Y_DATA)

    np.testing.assert_array_equal(CALLS[0]["X"], X_DATA)
    np.testing.assert_array_equal(CALLS[0]["y"], Y_DATA)
    assert CALLS[0]["top_k"] == 2 and CALLS[0]["cv"] == 2
    assert s.report_["evaluation_mode"] == "resampled"


def test_shadows_on_only_with_min_shadow_share(patched):
    patched(_rows(["x0", "x1", "x2"], [1.0, 1.0, 1.0]))

    selector(k=2).fit(X_DATA, Y_DATA)
    selector(k=None, min_shadow_share=0.5, shadow_seed=4).fit(X_DATA, Y_DATA)

    assert [(c["shadows"], c["shadow_seed"]) for c in CALLS] == [(False, None), (True, 4)]


def test_shadows_on_with_max_shadow_q_alone(patched):
    patched(_rows(["x0", "x1", "x2"], qs=[0.1, 0.1, 0.1]))

    s = selector(k=None, max_shadow_q=0.2, shadow_seed=4).fit(X_DATA, Y_DATA)

    assert [(c["shadows"], c["shadow_seed"]) for c in CALLS] == [(True, 4)]
    assert s.selected_groups_ == ["x0", "x1", "x2"]


def test_no_rule_error_names_all_three(patched):
    with pytest.raises(ValueError, match="k, min_shadow_share, or max_shadow_q"):
        selector(k=None).fit(X_DATA, Y_DATA)


def test_q_rule_keeps_groups_at_or_below_threshold(patched):
    patched(_rows(["x0", "x1", "x2"], qs=[0.01, 0.2, 0.04]))

    s = selector(k=None, max_shadow_q=0.05).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x0", "x2"]


def test_q_filter_then_k(patched):
    patched(_rows(["x0", "x1", "x2"], qs=[0.01, 0.2, 0.04]))

    s = selector(k=1, max_shadow_q=0.05).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x0"]


def test_both_filters_apply(patched):
    patched(_rows(["x0", "x1", "x2"], [0.9, 0.9, 0.2], [0.5, 0.01, 0.01]))

    s = selector(k=None, min_shadow_share=0.5, max_shadow_q=0.05).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x1"]


def test_top_k_rule(patched):
    patched(_rows(["x2", "x0", "x1"]))

    assert selector(k=2).fit(X_DATA, Y_DATA).selected_groups_ == ["x2", "x0"]


def test_k_larger_than_group_count_keeps_all(patched):
    patched(_rows(["x2", "x0", "x1"]))

    assert selector(k=10).fit(X_DATA, Y_DATA).selected_groups_ == ["x2", "x0", "x1"]


def test_shadow_rule_keeps_groups_at_or_above_threshold(patched):
    patched(_rows(["x0", "x1", "x2"], [0.4, 0.5, 0.8]))

    s = selector(k=None, min_shadow_share=0.5).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x1", "x2"]


def test_shadow_filter_then_k(patched):
    patched(_rows(["x0", "x1", "x2"], [0.4, 0.9, 0.8]))

    s = selector(k=1, min_shadow_share=0.5).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x1"]


def test_empty_selection_warns_and_supports_nothing(patched):
    patched(_rows(["x0", "x1", "x2"], [0.2, 0.3, 0.4]))

    with pytest.warns(UserWarning, match="No feature group"):
        s = selector(k=None, min_shadow_share=1.0).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == []
    assert s.get_support().tolist() == [False, False, False]


def test_group_selects_all_its_columns_in_original_order(patched):
    patched(_rows(["seg", "x1"]), groups={"seg": ["x0", "x2"], "x1": ["x1"]})

    s = selector(k=1, feature_groups={"seg": ["x0", "x2"]}).fit(X_DATA, Y_DATA)

    assert s.get_support().tolist() == [True, False, True]
    np.testing.assert_array_equal(s.transform(X_DATA), X_DATA[:, [0, 2]])


def test_array_input_uses_x_names_with_real_ranking():
    s = selector(k=1, feature_groups={"pair": ["x1", "x2"]}).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == ["x0"]
    assert s.get_feature_names_out().tolist() == ["x0"]
    assert not hasattr(s, "feature_names_in_")
    assert s.n_features_in_ == 3


def test_dataframe_names_in_and_out():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X_DATA, columns=["p", "q", "r"])

    s = selector(k=1).fit(frame, Y_DATA)

    assert s.feature_names_in_.tolist() == ["p", "q", "r"]
    assert s.get_feature_names_out().tolist() == ["p"]
    assert list(s.transform(frame).shape) == [60, 1]


from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline

SEEN_ROWS = []


class RowRecorder(RegressorMixin, BaseEstimator):
    """Records the row ids (column 0) of every training set it is fit on."""

    def fit(self, X, y):
        SEEN_ROWS.append(set(X[:, 0].tolist()))
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def pipeline(**kwargs):
    return Pipeline([("select", selector(**kwargs)), ("model", Ridge())])


def test_cross_val_score_gives_finite_scores():
    scores = cross_val_score(pipeline(k=1), X_DATA, Y_DATA, cv=3)

    assert len(scores) == 3 and np.isfinite(scores).all()


def test_selector_only_sees_outer_training_rows():
    SEEN_ROWS.clear()
    outer = KFold(3, shuffle=True, random_state=1)

    cross_val_score(pipeline(k=1, models={"rec": RowRecorder()}), X_DATA, Y_DATA, cv=outer)

    train_folds = [set(X_DATA[train, 0].tolist()) for train, _ in outer.split(X_DATA)]
    # inner cv=2 -> two selector fits per outer fold, in outer-fold order
    assert len(SEEN_ROWS) == 6
    for index, rows in enumerate(SEEN_ROWS):
        assert rows <= train_folds[index // 2]


def test_grid_search_over_k():
    search = GridSearchCV(pipeline(k=1), {"select__k": [1, 2]}, cv=2).fit(X_DATA, Y_DATA)

    assert search.best_params_["select__k"] in {1, 2}


def test_diabetes_end_to_end():
    data = load_diabetes()
    s = ConsensusSelector("reg", "r2", k=3, group="tree", cv=3, n_repeats=2,
                          random_state=0).fit(data.data, data.target)

    assert len(s.selected_groups_) == 3
    assert s.transform(data.data).shape == (len(data.data), 3)
    assert set(s.selected_groups_) <= {f"x{i}" for i in range(10)}


from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold


def test_failed_refit_clears_fitted_state():
    pd = pytest.importorskip("pandas")
    s = selector(k=1).fit(pd.DataFrame(X_DATA, columns=["a", "b", "c"]), Y_DATA)

    with pytest.raises(ValueError):
        s.fit(pd.DataFrame(X_DATA, columns=["c", "b", "a"]), Y_DATA[:-1])

    for attribute in ("support_", "selected_groups_", "report_", "n_features_in_",
                      "feature_names_in_"):
        assert not hasattr(s, attribute)
    with pytest.raises(NotFittedError):
        s.get_support()


@pytest.mark.parametrize("columns", [[0, "b", "c"], [0, 1, 2]])
def test_non_string_dataframe_columns_are_normalized(columns):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X_DATA, columns=columns)

    out = selector(k=1).fit_transform(frame, Y_DATA)

    assert out.shape == (60, 1)
    assert list(frame.columns) == columns  # caller's frame untouched


def test_reordered_integer_columns_are_rejected_on_transform():
    pd = pytest.importorskip("pandas")
    s = selector(k=1).fit(pd.DataFrame(X_DATA, columns=[0, 1, 2]), Y_DATA)

    with pytest.raises(ValueError, match="feature names"):
        s.transform(pd.DataFrame(X_DATA[:, ::-1], columns=[2, 1, 0]))


def test_numpy_integer_k_and_grid():
    assert selector(k=np.int64(1)).fit(X_DATA, Y_DATA).selected_groups_ == ["x0"]

    search = GridSearchCV(pipeline(k=1), {"select__k": np.arange(1, 3)}, cv=2)
    assert search.fit(X_DATA, Y_DATA).best_params_["select__k"] in {1, 2}


ROW_GROUPS = np.repeat(np.arange(6), 10)


def test_groups_are_forwarded_to_the_splitter():
    s = selector(k=1, cv=GroupKFold(2)).fit(X_DATA, Y_DATA, groups=ROW_GROUPS)

    for split in s.report_["splits"]:
        assert split["n_train"] + split["n_eval"] == 60


def test_groups_route_through_pipeline_and_outer_cv():
    pipe = pipeline(k=1, cv=GroupKFold(2))

    scores = cross_val_score(pipe, X_DATA, Y_DATA, cv=KFold(3),
                             params={"select__groups": ROW_GROUPS})

    assert np.isfinite(scores).all()


def test_selector_requires_y_tag():
    s = selector()
    if hasattr(s, "__sklearn_tags__") and hasattr(BaseEstimator, "__sklearn_tags__"):
        assert s.__sklearn_tags__().target_tags.required is True
    else:
        assert s._get_tags()["requires_y"] is True


def test_colliding_labels_rejected_on_transform():
    pd = pytest.importorskip("pandas")
    s = selector(k=1).fit(X_DATA, Y_DATA)

    with pytest.raises(ValueError, match="unique"):
        s.transform(pd.DataFrame(X_DATA, columns=[1, "1", "c"]))


# -- feature_groups="auto" ----------------------------------------------------

AUTO_X = np.column_stack([X_DATA[:, 0], X_DATA[:, 0] + 0.05 * RNG.normal(size=60), X_DATA[:, 2]])


def test_auto_groups_the_near_copy_pair_in_the_report():
    s = selector(feature_groups="auto").fit(AUTO_X, Y_DATA)

    assert s.report_["feature_groups"] == {
        "x0+x1": ["x0", "x1"],
        "x2": ["x2"],
    }
    assert s.get_params()["feature_groups"] == "auto"
    assert clone(s).get_params()["feature_groups"] == "auto"
    assert s.selected_groups_ == ["x0+x1"]
    assert s.support_.tolist() == [True, True, False]


def test_auto_with_dataframe_names():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(AUTO_X, columns=["a", "b", "c"])

    s = selector(feature_groups="auto").fit(frame, Y_DATA)

    assert s.report_["feature_groups"]["a+b"] == ["a", "b"]


def test_other_feature_groups_string_raises_in_fit():
    s = selector(feature_groups="manual")

    assert s.feature_groups == "manual"  # not validated in __init__
    with pytest.raises(ValueError, match="feature_groups"):
        s.fit(AUTO_X, Y_DATA)


def test_auto_groups_come_from_each_pipeline_training_fold():
    # The pair is a copy only in the first half of the rows.
    n = 100
    rng = np.random.default_rng(1)
    e = rng.normal(size=n)
    f = rng.normal(size=n)
    f[:50] = e[:50] + 0.02 * rng.normal(size=50)
    X = np.column_stack([e, f, rng.normal(size=n)])
    y = e + 0.1 * rng.normal(size=n)

    result = cross_validate(
        pipeline(feature_groups="auto"), X, y, cv=KFold(2), return_estimator=True
    )

    folds = [est["select"].report_["feature_groups"] for est in result["estimator"]]
    assert "x0+x1" not in folds[0]  # trained on rows 50..99
    assert folds[1]["x0+x1"] == ["x0", "x1"]  # trained on rows 0..49


def test_non_dict_feature_groups_error_mentions_auto():
    with pytest.raises(TypeError, match="auto"):
        selector(feature_groups=5).fit(AUTO_X, Y_DATA)


@pytest.mark.parametrize("make", [np.array, lambda v: __import__("pandas").Series(v)])
def test_array_or_series_feature_groups_raise_the_auto_type_error(make):
    with pytest.raises(TypeError, match="auto"):
        selector(feature_groups=make(["x0", "x1"])).fit(AUTO_X, Y_DATA)


def test_max_shadow_q_alone_gives_report_with_shadow_test():
    s = selector(k=None, max_shadow_q=1.0, shadow_seed=0).fit(X_DATA, Y_DATA)

    assert "shadow_test" in s.report_
    assert all("shadow_q_value" in row for row in s.report_["stability"])
    assert s.selected_groups_ == [r["feature_group"] for r in s.report_["stability"]]


def test_empty_q_selection_warning_names_splits_and_min_p_value():
    with pytest.warns(UserWarning, match="No feature group") as caught:
        s = selector(k=None, max_shadow_q=0.01, shadow_seed=0, cv=5).fit(X_DATA, Y_DATA)

    assert s.selected_groups_ == []
    message = str(caught[0].message)
    assert "5 splits" in message and "0.031" in message and "RepeatedKFold" in message


def test_empty_share_selection_warning_is_unchanged(patched):
    patched(_rows(["x0", "x1", "x2"], [0.2, 0.3, 0.4]))

    with pytest.warns(UserWarning) as caught:
        selector(k=None, min_shadow_share=1.0).fit(X_DATA, Y_DATA)

    assert str(caught[0].message) == (
        "No feature group was selected; transform() will return no columns."
    )


def test_selector_pickled_without_max_shadow_q_loads_clones_and_refits():
    fitted = selector(k=2).fit(X_DATA, Y_DATA)
    expected = fitted.transform(X_DATA)
    del fitted.__dict__["max_shadow_q"]  # what a 1.3 pickle holds
    old = pickle.loads(pickle.dumps(fitted))

    np.testing.assert_array_equal(old.transform(X_DATA), expected)
    assert old.get_params()["max_shadow_q"] is None
    refit = clone(old).fit(X_DATA, Y_DATA)
    assert refit.max_shadow_q is None
    np.testing.assert_array_equal(refit.transform(X_DATA), expected)
    old.fit(X_DATA, Y_DATA)  # refitting the loaded object itself also works
