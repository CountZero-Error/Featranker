import datetime
import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker, auto_groups
from featranker.evidence import correlation_evidence

N = 150
N_TRAIN = 100
RNG = np.random.default_rng(0)
CITY = RNG.choice(["a", "b", "c"], N)
# Not monotone in the sorted level order, so only one-hot (not codes) fits linearly.
EFFECT = {"a": 2.0, "b": -2.0, "c": 2.0}
X_NUM = RNG.normal(size=N)
Z_NUM = RNG.normal(size=N)
Y_REG = np.array([EFFECT[c] for c in CITY]) + 0.2 * RNG.normal(size=N)
Y_CLF = (Y_REG > 0).astype(int)
FRAME = pd.DataFrame({"x": X_NUM, "city": CITY, "z": Z_NUM})
NAMES = list(FRAME.columns)

NUMERIC_ERROR = "X must be numeric"


def ranker(task="reg", group="fast", **kwargs):
    return FeatureRanker(task, group, verbose=False, **{"categorical": "auto", **kwargs})


def fit_rank(r, X=FRAME, y=Y_REG, scoring="r2", **kwargs):
    r.fit(X.iloc[:N_TRAIN], y[:N_TRAIN])
    kwargs = {"n_repeats": 2, "random_state": 0, **kwargs}
    return r.rank_features(X.iloc[N_TRAIN:], y[N_TRAIN:], scoring=scoring, **kwargs)


class Recorder(RegressorMixin, BaseEstimator):
    """Keeps the training matrix and the first matrix it is asked to predict."""

    def fit(self, X, y):
        self.fit_X_ = np.array(X)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        if not hasattr(self, "eval_X_"):
            self.eval_X_ = np.array(X)
        return np.full(len(X), self.mean_)


def recording(**kwargs):
    return FeatureRanker("reg", models={"rec": Recorder()}, verbose=False, **kwargs)


# -- option and detection (1.1) ------------------------------------------------


def test_categorical_defaults_to_none():
    assert FeatureRanker("reg", verbose=False).categorical is None


@pytest.mark.parametrize("value", ["city", "AUTO", True, 3, ["a", 1], [""], {"a"}, ("a", None)])
def test_invalid_categorical_raises_at_construction(value):
    with pytest.raises(ValueError, match="categorical"):
        FeatureRanker("reg", verbose=False, categorical=value)


def test_unknown_name_raises_at_fit():
    r = FeatureRanker("reg", verbose=False, categorical=["nope"])

    with pytest.raises(ValueError, match="nope"):
        r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])


def test_default_rejects_strings_as_before():
    with pytest.raises(ValueError, match="could not convert string"):
        FeatureRanker("reg", "fast", verbose=False).fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])


def test_auto_detects_by_dtype():
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "count": [1, 2, 3, 4],
            "flag": [True, False, True, False],
            "obj": ["a", "b", "a", "b"],
            "text": pd.array(["a", "b", "a", "b"], dtype="string"),
            "cat": pd.Categorical(["u", "v", "u", "v"]),
        }
    )
    r = recording(categorical="auto")

    r.fit(frame, np.arange(4.0))

    assert r.categorical_features_ == ["obj", "text", "cat"]


def test_auto_example_from_the_spec():
    r = ranker()

    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    assert r.categorical_features_ == ["city"]


def test_explicit_list_on_numpy_object_array():
    X = np.array([[1.0, "p"], [2.0, "q"], [3.0, "p"], [4.0, "q"]], dtype=object)
    r = recording(categorical=["city"])

    r.fit(X, np.arange(4.0), ["x", "city"])

    assert r.categorical_features_ == ["city"]
    np.testing.assert_array_equal(r.trained_models["rec"].fit_X_, [[1, 0], [2, 1], [3, 0], [4, 1]])


def test_explicit_list_makes_a_numeric_column_categorical():
    frame = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5], "code": [10, 20, 10, 20]})
    r = recording(categorical=["code"])

    r.fit(frame, np.arange(4.0))

    assert r.categorical_features_ == ["code"]
    np.testing.assert_array_equal(r.trained_models["rec"].fit_X_[:, 1], [0, 1, 0, 1])


def test_auto_on_numpy_finds_no_categorical_columns():
    r = recording(categorical="auto")

    r.fit(np.arange(8.0).reshape(4, 2), np.arange(4.0), ["a", "b"])

    assert r.categorical_features_ == []


def test_auto_on_numeric_frame_matches_default_and_adds_no_key():
    numeric = FRAME[["x", "z"]]
    default = fit_rank(FeatureRanker("reg", "fast", verbose=False), X=numeric)
    auto = fit_rank(ranker(), X=numeric)

    assert "categorical" not in auto
    assert json.dumps(default, sort_keys=True) == json.dumps(auto, sort_keys=True)


def test_ranker_unpickled_without_categorical_behaves_as_none():
    r = FeatureRanker("reg", "fast", verbose=False)
    r.fit(FRAME[["x", "z"]].iloc[:N_TRAIN], Y_REG[:N_TRAIN])
    state = pickle.loads(pickle.dumps(r)).__dict__.copy()
    for key in ("categorical", "categorical_features_", "categorical_levels_"):
        state.pop(key, None)  # what a 1.8 pickle holds
    old = FeatureRanker.__new__(FeatureRanker)

    old.__setstate__(state)

    assert old.categorical is None
    report = old.rank_features(FRAME[["x", "z"]].iloc[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2")
    assert "categorical" not in report


# -- level coding (1.2) --------------------------------------------------------


def test_levels_are_sorted_with_missing_last_and_coded():
    frame = pd.DataFrame({"city": ["b", None, "a", np.nan, "c", "a"]})
    r = recording(categorical=["city"])

    r.fit(frame, np.arange(6.0))

    assert r.categorical_levels_ == {"city": ["a", "b", "c", None]}
    np.testing.assert_array_equal(r.trained_models["rec"].fit_X_[:, 0], [1, 3, 0, 3, 2, 0])


def test_levels_use_the_sorted_string_form():
    frame = pd.DataFrame({"v": [10, 9, 2, 10]})
    r = recording(categorical=["v"])

    r.fit(frame, np.arange(4.0))

    assert r.categorical_levels_ == {"v": ["10", "2", "9"]}


def test_levels_come_from_training_rows_only_and_unseen_share_one_code():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "city": ["a", "b", "a", "b"]})
    evaluation = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "city": ["a", "c", "d", "c"]}
    )
    r = recording(categorical=["city"])
    r.fit(train, np.arange(4.0))

    report = r.rank_features(evaluation, np.arange(4.0), scoring="r2", n_repeats=1)

    assert report["categorical"]["train_levels"] == {"city": 2}
    assert report["categorical"]["unseen_eval"] == {"city": 3}
    np.testing.assert_array_equal(r.trained_models["rec"].eval_X_[:, 1], [0, 2, 2, 2])


def test_integral_levels_match_across_int_and_float_dtypes():
    rng = np.random.default_rng(1)
    code = rng.integers(0, 3, 120)
    y = np.array([2.0, -2.0, 2.0])[code] + 0.1 * rng.normal(size=120)
    frame = pd.DataFrame({"code": code, "z": rng.normal(size=120)})
    evaluation = frame.iloc[60:].copy()
    evaluation["code"] = evaluation["code"].astype(float)
    evaluation.iloc[0, 0] = np.nan
    r = ranker("reg", "fast", categorical=["code"])
    r.fit(frame.iloc[:60], y[:60])

    report = r.rank_features(evaluation, y[60:], scoring="r2", n_repeats=2, random_state=0)

    assert report["categorical"]["unseen_eval"] == {"code": 1}
    assert report["consensus"][0]["feature_group"] == "code"


def test_pandas_na_and_nat_count_as_missing_in_object_arrays():
    train = np.array([["a", 1.0], [pd.NA, 2.0], ["b", 3.0], [pd.NaT, 4.0]], dtype=object)
    r = recording(categorical=["city"])
    r.fit(train, np.arange(4.0), ["city", "x"])

    assert r.categorical_levels_ == {"city": ["a", "b", None]}


def test_nat_in_a_categorical_object_column_is_not_a_date():
    train = np.array([["a", 1.0], [pd.NaT, 2.0], ["b", 3.0], ["a", 4.0]], dtype=object)
    r = recording(categorical=["city"])
    r.fit(train, np.arange(4.0), ["city", "x"])

    assert r.categorical_features_ == ["city"]


def test_missing_in_evaluation_is_unseen_when_training_had_none():
    train = pd.DataFrame({"city": ["a", "b", "a", "b"]})
    evaluation = pd.DataFrame({"city": ["a", None, np.nan, "b"]})
    r = recording(categorical=["city"])
    r.fit(train, np.arange(4.0))

    report = r.rank_features(evaluation, np.arange(4.0), scoring="r2", n_repeats=1)

    assert report["categorical"]["unseen_eval"] == {"city": 2}


def test_models_get_numeric_columns_as_given_and_codes_in_original_order():
    r = recording(categorical="auto")

    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    seen = r.trained_models["rec"].fit_X_
    codes = np.searchsorted(["a", "b", "c"], CITY[:N_TRAIN])
    np.testing.assert_array_equal(seen[:, 0], X_NUM[:N_TRAIN])
    np.testing.assert_array_equal(seen[:, 1], codes)
    np.testing.assert_array_equal(seen[:, 2], Z_NUM[:N_TRAIN])
    assert r.categorical_features_ == ["city"]


def test_shadow_columns_permute_the_codes():
    r = recording(categorical="auto", shadows=True, shadow_seed=0)

    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    seen = r.trained_models["rec"].fit_X_
    assert seen.shape == (N_TRAIN, 6)
    np.testing.assert_array_equal(np.sort(seen[:, 4]), np.sort(seen[:, 1]))
    assert set(np.unique(seen[:, 4])) <= {0.0, 1.0, 2.0}


def test_each_raw_column_is_one_feature():
    report = fit_rank(ranker())

    assert report["feature_groups"] == {"x": ["x"], "city": ["city"], "z": ["z"]}
    assert [row["feature_group"] for row in report["consensus"]].count("city") == 1


def test_categorical_column_can_join_a_user_group():
    report = fit_rank(ranker(), feature_groups={"geo": ["city", "z"]})

    assert report["feature_groups"]["geo"] == ["city", "z"]
    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


# -- date-valued data is not numeric (1.4) -------------------------------------

DATES = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
DELTAS = pd.to_timedelta([1, 2, 3, 4], unit="D")


def _frames():
    base = {"x": [1.0, 2.0, 3.0, 4.0]}
    return {
        "datetime column": pd.DataFrame({**base, "d": DATES}),
        "timedelta column": pd.DataFrame({**base, "d": DELTAS}),
        "tz-aware column": pd.DataFrame({**base, "d": DATES.tz_localize("UTC")}),
        "date objects": pd.DataFrame({**base, "d": [datetime.date(2020, 1, i) for i in range(1, 5)]}),
        "timestamp objects": pd.DataFrame({**base, "d": list(DATES.astype(object))}),
        "datetime categories": pd.DataFrame({**base, "d": pd.Categorical(DATES)}),
        "timedelta categories": pd.DataFrame({**base, "d": pd.Categorical(DELTAS)}),
    }


@pytest.mark.parametrize("label", list(_frames()))
@pytest.mark.parametrize("categorical", [None, "auto", ["d"]])
@pytest.mark.parametrize("missing", ["error", "impute"])
def test_date_valued_columns_raise_under_every_policy(label, categorical, missing):
    r = FeatureRanker("reg", "fast", verbose=False, categorical=categorical, missing=missing)

    with pytest.raises(ValueError, match=NUMERIC_ERROR):
        r.fit(_frames()[label], np.arange(4.0))


@pytest.mark.parametrize("kind", ["datetime64[ns]", "datetime64[D]", "timedelta64[s]"])
@pytest.mark.parametrize("categorical", [None, "auto", ["a"]])
@pytest.mark.parametrize("missing", ["error", "impute"])
def test_numpy_date_arrays_raise_under_every_policy(kind, categorical, missing):
    X = np.arange(8).reshape(4, 2).astype(kind)
    r = FeatureRanker("reg", "fast", verbose=False, categorical=categorical, missing=missing)

    with pytest.raises(ValueError, match=NUMERIC_ERROR):
        r.fit(X, np.arange(4.0), ["a", "b"])


def test_dates_also_raise_in_evaluation_resampling_and_auto_groups():
    r = ranker()
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])
    dated = FRAME.iloc[N_TRAIN:].assign(z=DATES[0])
    with pytest.raises(ValueError, match=NUMERIC_ERROR):
        r.rank_features(dated, Y_REG[N_TRAIN:], scoring="r2")
    with pytest.raises(ValueError, match=NUMERIC_ERROR):
        r.rank_resampled(FRAME.assign(city=DATES[0]), Y_REG, scoring="r2", cv=2)
    with pytest.raises(ValueError, match=NUMERIC_ERROR):
        auto_groups(FRAME.assign(city=DATES[0]), categorical="auto")


# -- built-in models (2.1) -----------------------------------------------------


def _steps(model):
    return [step for _, step in model.steps]


@pytest.mark.parametrize("missing", ["error", "impute"])
def test_linear_builtins_one_hot_categoricals_and_scale_numerics(missing):
    r = ranker("reg", "all", missing=missing)
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    linear = set(r.models["reg"]["linear"])
    assert linear
    for name in linear & set(r.trained_models):
        model = r.trained_models[name]
        transformer = _steps(model)[0]
        assert isinstance(transformer, ColumnTransformer), name
        assert not isinstance(_steps(model)[1], ColumnTransformer)
        assert isinstance(transformer.named_transformers_["cat"], OneHotEncoder)
        assert list(transformer.transformers[0][2]) == [1]
        remainder = _steps(transformer.named_transformers_["remainder"])
        assert isinstance(remainder[-1], StandardScaler)
        assert isinstance(remainder[0], SimpleImputer) == (missing == "impute")
        coded = FRAME.iloc[:5].assign(city=[0, 1, 2, 0, 1]).to_numpy(dtype=float)
        assert transformer.transform(coded).shape == (5, 5)


def test_tree_builtins_use_codes_unchanged():
    r = ranker("reg", "tree", missing="impute")
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    assert r.trained_models
    for model in r.trained_models.values():
        steps = _steps(model) if isinstance(model, Pipeline) else []
        assert not any(isinstance(step, (ColumnTransformer, OneHotEncoder)) for step in steps)


def test_registry_models_stay_as_in_1_8():
    r = ranker("reg", "all", missing="impute")
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    for model in r.models["reg"]["linear"].values():
        assert [type(step) for step in _steps(model)[:2]] == [SimpleImputer, StandardScaler]


@pytest.mark.parametrize(("task", "y", "scoring"), [("reg", Y_REG, "r2"), ("clf", Y_CLF, "accuracy")])
@pytest.mark.parametrize("missing", ["error", "impute"])
def test_fast_group_ranks_the_driving_categorical_first(task, y, scoring, missing):
    report = fit_rank(ranker(task, "fast", missing=missing), y=y, scoring=scoring)

    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]
    assert report["consensus"][0]["feature_group"] == "city"
    assert report["categorical"]["columns"] == ["city"]
    assert max(m["evaluation_score"] for m in report["models"].values()) > 0.7


def test_one_hot_lets_linear_models_fit_a_non_monotone_categorical():
    r = ranker("reg", "linear")
    report = fit_rank(r)

    assert report["models"]["linear_regression"]["evaluation_score"] > 0.9


def test_missing_values_in_both_kinds_of_column_with_impute():
    frame = FRAME.copy()
    frame.loc[frame.index[::7], "x"] = np.nan
    frame.loc[frame.index[::5], "city"] = None

    report = fit_rank(ranker("reg", "all", missing="impute"), X=frame)

    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]
    assert report["missing"]["train_counts"]["x"] > 0
    assert report["missing"]["train_counts"]["city"] == len(frame.iloc[:N_TRAIN][::5])
    assert report["categorical"]["train_levels"] == {"city": 4}


def test_shadows_with_categoricals_flag_above_shadow():
    report = fit_rank(ranker("reg", "all", shadows=True, shadow_seed=1))

    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]
    for model in report["models"].values():
        assert "above_shadow" in model["importance"]["city"]
    assert report["consensus"][0]["feature_group"] == "city"


def test_shadow_categorical_columns_are_one_hot_too():
    r = ranker("reg", "linear", shadows=True, shadow_seed=1)
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    transformer = _steps(r.trained_models["linear_regression"])[0]

    assert list(transformer.transformers[0][2]) == [1, 4]


def test_all_categorical_table_fits_linear_models():
    frame = pd.DataFrame({"u": CITY, "v": CITY[::-1]})

    report = fit_rank(ranker("reg", "linear"), X=frame)

    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


def test_custom_models_receive_codes_and_the_names_are_exposed():
    r = recording(categorical="auto")
    r.fit(FRAME.iloc[:N_TRAIN], Y_REG[:N_TRAIN])

    report = r.rank_features(FRAME.iloc[N_TRAIN:], Y_REG[N_TRAIN:], scoring="r2", n_repeats=1)

    assert r.categorical_features_ == ["city"]
    eval_codes = r.trained_models["rec"].eval_X_[:, 1]
    np.testing.assert_array_equal(eval_codes, np.searchsorted(["a", "b", "c"], CITY[N_TRAIN:]))
    assert report["categorical"]["unseen_eval"] == {"city": 0}


# -- evidence and record (2.2) -------------------------------------------------


def test_correlation_evidence_skips_categorical_columns():
    codes = np.tile([0.0, 1.0, 2.0], 20)
    X = np.column_stack([codes, codes * 2 + 1.0, np.linspace(0, 1, 60)])
    groups = {"c": [0], "n": [1], "m": [2]}

    assert correlation_evidence(X, groups)["pairs"][0]["groups"] == ["c", "n"]
    pairs = correlation_evidence(X, groups, categorical=[0])["pairs"]

    assert not [pair for pair in pairs if "c" in pair["groups"]]


def test_report_has_no_correlated_pair_with_a_categorical_column():
    x = np.repeat([0.0, 1.0, 2.0], N // 3)
    frame = pd.DataFrame({"x": x, "city": np.array(["a", "b", "c"])[x.astype(int)]})
    y = RNG.normal(size=N)

    report = fit_rank(ranker("reg", "fast", evidence=True), X=frame, y=y)

    assert report["evidence"]["correlation"]["pairs"] == []


def test_auto_groups_never_groups_categorical_columns():
    base = RNG.normal(size=60)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base * 2 + 0.01 * RNG.normal(size=60),
            "city": np.array(["p", "q", "r"])[np.argsort(np.argsort(base)) // 20],
        }
    )

    for option in ("auto", ["city"]):
        groups = auto_groups(frame, categorical=option)
        assert groups == {"a+b": ["a", "b"]}


def test_auto_groups_without_categorical_still_rejects_strings():
    with pytest.raises(ValueError, match="could not convert string"):
        auto_groups(FRAME)


def test_auto_groups_rejects_unknown_categorical_names_and_values():
    with pytest.raises(ValueError, match="nope"):
        auto_groups(FRAME, categorical=["nope"])
    with pytest.raises(ValueError, match="categorical"):
        auto_groups(FRAME, categorical=7)


def test_shap_explains_trees_on_codes_and_lists_one_hot_pipelines():
    pytest.importorskip("shap")
    r = ranker("reg", "all", evidence=True)

    report = fit_rank(r)

    unsupported = {entry["model"]: entry["reason"] for entry in report["evidence"]["shap"]["unsupported"]}
    linear = set(r.models["reg"]["linear"]) & set(report["models"])
    tree = set(r.models["reg"]["tree"]) & set(report["models"])
    assert linear and tree
    assert linear <= set(unsupported)
    assert all("one-hot" in unsupported[name] for name in linear)
    explained = {name for name, m in report["models"].items() if "shap" in m}
    assert explained and explained <= tree
    for name in explained & {"random_forest_regressor", "decision_tree_regressor"}:
        groups = report["models"][name]["shap"]["groups"]
        assert groups["city"]["mean_abs"] > groups["x"]["mean_abs"]
        assert groups["city"]["direction"] is None
        assert groups["x"]["direction"] is not None


def test_report_record():
    train = pd.DataFrame({"x": np.arange(12.0), "city": list("abc") * 4})
    evaluation = pd.DataFrame({"x": np.arange(6.0), "city": ["a", "d", "e", "b", "c", "a"]})
    r = ranker("reg", "fast")
    r.fit(train, np.arange(12.0))

    report = r.rank_features(evaluation, np.arange(6.0), scoring="r2", n_repeats=1)

    assert report["categorical"] == {
        "columns": ["city"], "train_levels": {"city": 3}, "unseen_eval": {"city": 2}
    }
    json.dumps(report, allow_nan=False)


def test_no_categorical_key_without_categorical_columns():
    report = fit_rank(FeatureRanker("reg", "fast", verbose=False), X=FRAME[["x", "z"]])

    assert "categorical" not in report


def test_resampled_split_reports_learn_their_own_levels():
    frame = pd.DataFrame({"x": np.arange(12.0), "city": list("aabbaabbaabb")})
    frame.loc[10, "city"] = "rare"
    y = np.arange(12.0)
    cv = [(np.arange(0, 10), np.arange(10, 12)), (np.arange(2, 12), np.arange(0, 2))]
    r = ranker("reg", "fast", models=None)

    report = r.rank_resampled(frame, y, scoring="neg_mean_absolute_error", cv=cv, n_repeats=1)

    records = [split["report"]["categorical"] for split in report["splits"]]
    assert records[0] == {
        "columns": ["city"], "train_levels": {"city": 2}, "unseen_eval": {"city": 1}
    }
    assert records[1]["train_levels"] == {"city": 3}
    assert records[1]["unseen_eval"] == {"city": 0}
    json.dumps(report, allow_nan=False)


def test_resampled_accepts_auto_on_a_frame_and_leaves_the_ranker_unfitted():
    r = ranker("reg", "fast")

    report = r.rank_resampled(FRAME, Y_REG, scoring="r2", cv=3, n_repeats=1, random_state=0)

    assert report["stability"][0]["feature_group"] == "city"
    assert all("categorical" in split["report"] for split in report["splits"])
    assert not r.is_fitted_


def test_resampled_with_an_object_array_and_a_name_list():
    X = np.array(FRAME, dtype=object)
    r = FeatureRanker("reg", "fast", verbose=False, categorical=["city"])

    report = r.rank_resampled(
        X, Y_REG, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1, random_state=0
    )

    assert report["stability"][0]["feature_group"] == "city"


# -- pass-through (3.1) --------------------------------------------------------


def _selector(**kwargs):
    return ConsensusSelector(
        "reg", "r2", k=1, categorical="auto", group="fast", cv=3, n_repeats=2,
        random_state=0, **kwargs,
    )


def test_selector_on_a_mixed_frame_selects_the_raw_column():
    selector = _selector()

    selector.fit(FRAME, Y_REG)

    assert selector.selected_groups_ == ["city"]
    transformed = selector.transform(FRAME)
    assert transformed.shape == (N, 1)
    assert transformed[:, 0].tolist() == CITY.tolist()
    assert selector.report_["splits"][0]["report"]["categorical"]["columns"] == ["city"]


def test_selector_transform_accepts_missing_categorical_values():
    frame = FRAME.copy()
    frame.loc[frame.index[::6], "city"] = None
    selector = _selector()

    selector.fit(frame, Y_REG)

    assert selector.selected_groups_ == ["city"]
    assert selector.transform(frame).shape == (N, 1)


def test_selector_stores_categorical_unchanged_and_clones():
    names = ["city"]
    selector = _selector()
    listed = ConsensusSelector("reg", "r2", k=1, categorical=names)

    assert ConsensusSelector("reg", "r2", k=1).get_params()["categorical"] is None
    assert clone(selector).get_params()["categorical"] == "auto"
    assert listed.categorical is names
    assert clone(listed).categorical == ["city"]


def test_selector_with_a_numpy_object_array_and_names():
    X = np.array(FRAME, dtype=object)
    selector = ConsensusSelector(
        "reg", "r2", k=1, categorical=["x1"], group="fast", cv=3, n_repeats=2, random_state=0
    )

    selector.fit(X, Y_REG)

    assert selector.selected_groups_ == ["x1"]
    assert selector.support_.tolist() == [False, True, False]


def test_selector_auto_groups_leave_categoricals_out():
    base = RNG.normal(size=N)
    frame = pd.DataFrame({"a": base, "b": base + 0.001 * RNG.normal(size=N), "city": CITY})
    selector = ConsensusSelector(
        "reg", "r2", k=1, categorical="auto", feature_groups="auto", group="fast", cv=3,
        n_repeats=1, random_state=0,
    )

    selector.fit(frame, Y_REG)

    assert selector.report_["feature_groups"] == {"a+b": ["a", "b"], "city": ["city"]}


def test_selector_unpickled_without_categorical_behaves_as_none():
    selector = ConsensusSelector("reg", "r2", k=1, group="fast", cv=3)
    state = pickle.loads(pickle.dumps(selector)).__dict__.copy()
    del state["categorical"]
    old = ConsensusSelector.__new__(ConsensusSelector)

    old.__setstate__(state)

    assert old.categorical is None
    with pytest.raises(ValueError, match="could not convert string"):
        old.fit(FRAME, Y_REG)


def test_selector_in_a_pipeline_under_cross_val_score():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(_selector(), OneHotEncoder(handle_unknown="ignore"), Ridge())

    scores = cross_val_score(pipeline, FRAME, Y_REG, cv=2)

    assert scores.shape == (2,) and scores.min() > 0.8


def _csv_files(tmp_path):
    def rows(start, stop):
        lines = ["x,city,z,target"]
        for index in range(start, stop):
            lines.append(f"{X_NUM[index]:.6f},{CITY[index]},{Z_NUM[index]:.6f},{Y_REG[index]:.6f}")
        return lines

    (tmp_path / "train.csv").write_text("\n".join(rows(0, N_TRAIN)), encoding="utf-8")
    (tmp_path / "eval.csv").write_text("\n".join(rows(N_TRAIN, N)), encoding="utf-8")
    return [
        "featranker", "--task", "reg", "--group", "fast", "--scoring", "r2",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--n-repeats", "2", "--random-state", "0", "--quiet",
    ]


@pytest.mark.parametrize("option", ["auto", "city", " city , z "])
def test_cli_categorical_reads_text_columns(monkeypatch, tmp_path, capsys, option):
    monkeypatch.setattr(sys, "argv", [*_csv_files(tmp_path), "--categorical", option])

    assert cli_module.main() == 0

    report = json.loads(capsys.readouterr().out)
    expected = ["city", "z"] if "z" in option else ["city"]
    assert report["categorical"]["columns"] == expected
    assert report["categorical"]["train_levels"]["city"] == 3
    if option != " city , z ":
        assert report["consensus"][0]["feature_group"] == "city"


def test_cli_without_categorical_rejects_text(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", _csv_files(tmp_path))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "numeric" in capsys.readouterr().err


def test_cli_rejects_unknown_categorical_name(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [*_csv_files(tmp_path), "--categorical", "nope"])

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "nope" in capsys.readouterr().err


def test_cli_empty_categorical_cell_is_the_missing_level(monkeypatch, tmp_path, capsys):
    argv = _csv_files(tmp_path)
    train = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()
    fields = train[1].split(",")
    fields[1] = ""
    train[1] = ",".join(fields)
    (tmp_path / "train.csv").write_text("\n".join(train), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*argv, "--categorical", "auto"])

    assert cli_module.main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["categorical"]["train_levels"] == {"city": 4}


def test_cli_auto_groups_leave_categoricals_out(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys, "argv", [*_csv_files(tmp_path), "--categorical", "auto", "--feature-groups", "auto"]
    )

    assert cli_module.main() == 0

    assert "city" in json.loads(capsys.readouterr().out)["feature_groups"]
