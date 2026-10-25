import json
import pickle
import re
import types
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from featranker import ConsensusSelector, FeatureRanker, summarize
from featranker import importance
from featranker.importance import _sample_whole_groups

RNG = np.random.default_rng(0)
N = 60
X_FAM = np.column_stack([np.arange(N, dtype=float), RNG.normal(size=N)])
Y_FAM = 3.0 * X_FAM[:, 0] + 0.1 * RNG.normal(size=N)
NAMES = ["a", "b"]


def three_ridges():
    return {
        "r1": Ridge(alpha=0.1), "r2": Ridge(alpha=0.2), "r3": Ridge(alpha=0.3),
        "tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    }


def rank(families=None, models=None, **kwargs):
    ranker = FeatureRanker(
        "reg", models=models or three_ridges(), families=families, verbose=False
    )
    ranker.fit(X_FAM[:40], Y_FAM[:40], NAMES)
    return ranker.rank_features(
        X_FAM[40:], Y_FAM[40:], scoring="r2", n_repeats=2, random_state=0, **kwargs
    )


# --- families -------------------------------------------------------------


@pytest.mark.parametrize(
    "families",
    ["ridge", ["r1"], {1: "ridge"}, {"": "ridge"}, {"r1": ""}, {"r1": 3}, {"r1": None}],
)
def test_invalid_families_raise_at_construction(families):
    with pytest.raises(ValueError, match="families"):
        FeatureRanker("reg", models=three_ridges(), families=families)


def test_unknown_model_name_raises_in_fit():
    ranker = FeatureRanker(
        "reg", models=three_ridges(), families={"nope": "ridge"}, verbose=False
    )
    with pytest.raises(ValueError, match="nope"):
        ranker.fit(X_FAM, Y_FAM, NAMES)


def test_builtin_name_outside_the_selected_group_raises_in_fit():
    ranker = FeatureRanker(
        "reg", group="linear", families={"random_forest_regression": "x"}, verbose=False
    )
    with pytest.raises(ValueError, match="random_forest_regression"):
        ranker.fit(X_FAM, Y_FAM, NAMES)


def test_unmapped_custom_models_are_their_own_families():
    report = rank()

    assert {m["family"] for m in report["models"].values()} == {"r1", "r2", "r3", "tree"}
    assert report["consensus"][0]["n_families"] == 4


def test_mapped_custom_models_share_one_vote():
    report = rank(families={"r1": "ridge", "r2": "ridge", "r3": "ridge"})

    assert report["models"]["r2"]["family"] == "ridge"
    assert report["models"]["tree"]["family"] == "tree"
    assert report["consensus"][0]["n_families"] == 2
    assert report["consensus"][0]["n_models"] == 4
    json.dumps(report, allow_nan=False)


def test_builtin_family_can_be_remapped():
    ranker = FeatureRanker(
        "reg", group="linear", families={"ridge_regression": "mine"}, verbose=False
    )
    ranker.fit(X_FAM, Y_FAM, NAMES)
    report = ranker.rank_features(X_FAM, Y_FAM, scoring="r2", n_repeats=1, random_state=0)

    assert report["models"]["ridge_regression"]["family"] == "mine"
    assert report["models"]["kernel_ridge_regression"]["family"] == "linear"


def test_families_mapping_is_copied():
    mapping = {"r1": "ridge"}
    ranker = FeatureRanker("reg", models=three_ridges(), families=mapping)
    mapping["r1"] = "changed"

    assert ranker.families == {"r1": "ridge"}


def test_ranker_pickled_before_families_loads_with_none():
    ranker = FeatureRanker("reg", models=three_ridges(), verbose=False)
    ranker.fit(X_FAM, Y_FAM, NAMES)
    del ranker.__dict__["families"]  # what a 1.17 pickle holds
    old = pickle.loads(pickle.dumps(ranker))
    report = old.rank_features(X_FAM, Y_FAM, scoring="r2", n_repeats=1, random_state=0)

    assert old.families is None
    assert report["models"]["r1"]["family"] == "r1"


def test_resampled_ranker_keeps_families():
    ranker = FeatureRanker(
        "reg", models=three_ridges(), verbose=False,
        families={"r1": "ridge", "r2": "ridge", "r3": "ridge"},
    )
    result = ranker.rank_resampled(
        X_FAM, Y_FAM, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1,
        random_state=0,
    )

    first = result["splits"][0]["report"]
    assert first["models"]["r3"]["family"] == "ridge"


def test_selector_passes_families_and_clones():
    s = ConsensusSelector(
        "reg", "r2", k=1, models=three_ridges(), cv=3, n_repeats=1, random_state=0,
        families={"r1": "ridge", "r2": "ridge", "r3": "ridge"},
    )

    assert clone(s).get_params()["families"] == s.families
    s.fit(X_FAM, Y_FAM)
    assert s.report_["splits"][0]["report"]["consensus"][0]["n_families"] == 2


def test_selector_pickled_before_families_loads():
    s = ConsensusSelector("reg", "r2", k=1, models=three_ridges(), cv=3, n_repeats=1)
    del s.__dict__["families"]
    old = pickle.loads(pickle.dumps(s))

    assert old.get_params()["families"] is None
    old.fit(X_FAM, Y_FAM)


def test_unknown_name_raises_before_any_split_in_rank_resampled():
    ranker = FeatureRanker(
        "reg", models=three_ridges(), families={"nope": "ridge"}, verbose=False
    )
    with pytest.raises(ValueError, match="nope"):
        ranker.rank_resampled(
            X_FAM, Y_FAM, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1
        )


def test_builtin_name_outside_the_group_raises_in_rank_resampled():
    ranker = FeatureRanker(
        "reg", group="linear", families={"random_forest_regression": "x"}, verbose=False
    )
    with pytest.raises(ValueError, match="random_forest_regression"):
        ranker.rank_resampled(
            X_FAM, Y_FAM, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1
        )


def test_selector_with_unknown_name_raises_value_error():
    s = ConsensusSelector(
        "reg", "r2", k=1, models=three_ridges(), cv=3, n_repeats=1, random_state=0,
        families={"nope": "ridge"},
    )
    with pytest.raises(ValueError, match="nope"):
        s.fit(X_FAM, Y_FAM)


@pytest.mark.filterwarnings("ignore:Skipping model")
def test_builtin_that_failed_to_initialise_is_still_a_known_name(monkeypatch):
    real = importance._model_params

    def failing(class_name, params):
        if class_name == "Ridge":
            raise RuntimeError("boom")
        return real(class_name, params)

    monkeypatch.setattr(importance, "_model_params", failing)
    with pytest.warns(UserWarning, match="Skipping model"):
        ranker = FeatureRanker(
            "reg", group="linear", families={"ridge_regression": "mine"}, verbose=False
        )
    assert "ridge_regression" in {f["model"] for f in ranker.initialization_failures_}

    ranker.fit(X_FAM, Y_FAM, NAMES)
    result = ranker.rank_resampled(
        X_FAM, Y_FAM, scoring="r2", cv=3, feature_names=NAMES, n_repeats=1,
        random_state=0,
    )

    assert "ridge_regression" not in result["splits"][0]["report"]["models"]


@pytest.mark.parametrize("mapping", [
    types.MappingProxyType({"r1": "ridge"}), OrderedDict(r1="ridge"),
])
def test_any_mapping_is_accepted_and_copied_to_a_dict(mapping):
    ranker = FeatureRanker("reg", models=three_ridges(), families=mapping)

    assert type(ranker.families) is dict
    assert ranker.families == {"r1": "ridge"}


def test_shadow_family_evidence_uses_the_mapped_families():
    ranker = FeatureRanker(
        "reg", models=three_ridges(), shadows=True, shadow_seed=0, verbose=False,
        families={"r1": "ridge", "r2": "ridge", "r3": "ridge"},
    )
    result = ranker.rank_resampled(
        X_FAM, Y_FAM, scoring="r2", cv=3, feature_names=NAMES, n_repeats=2,
        random_state=0,
    )

    for entry in result["splits"]:
        for row in entry["report"]["consensus"]:
            assert set(row["family_above_shadow"]) == {"ridge", "tree"}
    assert all(row["shadow_families"] <= 2 for row in result["stability"])
    assert all(set(row["family_above_shadow_share"]) <= {"ridge", "tree"}
               for row in result["stability"])


# --- constant within groups ----------------------------------------------

N_PATIENTS, ROWS_PER_PATIENT = 40, 4
PATIENT_RNG = np.random.default_rng(1)
PATIENTS = np.repeat(np.arange(N_PATIENTS), ROWS_PER_PATIENT)
X_PATIENT = np.repeat(PATIENT_RNG.normal(size=N_PATIENTS), ROWS_PER_PATIENT)
X_VARYING = PATIENT_RNG.normal(size=len(PATIENTS))
X_PAT = np.column_stack([X_PATIENT, X_VARYING])
Y_PAT = 3.0 * X_PATIENT + 0.1 * PATIENT_RNG.normal(size=len(PATIENTS))
HALF = len(PATIENTS) // 2


def patient_ranker(**kwargs):
    ranker = FeatureRanker(
        "reg", models={"lin": LinearRegression()}, verbose=False, **kwargs
    )
    within = kwargs.get("shuffle") == "within_groups"
    ranker.fit(
        X_PAT[:HALF], Y_PAT[:HALF], ["patient_x", "varying_x"],
        shuffle_groups=PATIENTS[:HALF] if within else None,
    )
    return ranker


def patient_report(ranker=None, **kwargs):
    return (ranker or patient_ranker(shuffle="within_groups")).rank_features(
        X_PAT[HALF:], Y_PAT[HALF:], scoring="r2", n_repeats=3, random_state=0,
        shuffle_groups=PATIENTS[HALF:], **kwargs
    )


def test_patient_level_predictor_is_listed_and_warned():
    with pytest.warns(UserWarning, match="patient_x"):
        report = patient_report()

    assert report["constant_within_groups"] == ["patient_x"]
    assert report["models"]["lin"]["importance"]["patient_x"]["mean"] == 0.0
    assert "varying_x" not in report["constant_within_groups"]
    assert list(report).index("constant_within_groups") == list(report).index("shuffle") + 1
    json.dumps(report, allow_nan=False)


def test_no_constant_group_gives_empty_list_and_no_warning():
    ranker = FeatureRanker("reg", models={"lin": LinearRegression()}, verbose=False,
                           shuffle="within_groups")
    ranker.fit(X_PAT[:HALF, 1:], Y_PAT[:HALF], ["varying_x"], shuffle_groups=PATIENTS[:HALF])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*constant within")
        report = ranker.rank_features(
            X_PAT[HALF:, 1:], Y_PAT[HALF:], scoring="r2", n_repeats=1,
            random_state=0, shuffle_groups=PATIENTS[HALF:],
        )

    assert report["constant_within_groups"] == []


def test_key_is_absent_in_other_modes():
    for mode in ("rows", "circular"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = patient_ranker(shuffle=mode).rank_features(
                X_PAT[HALF:], Y_PAT[HALF:], scoring="r2", n_repeats=1, random_state=0
            )
        assert "constant_within_groups" not in report


def test_feature_group_is_listed_only_if_every_column_is_constant():
    X = np.column_stack([X_PATIENT, 2 * X_PATIENT, X_VARYING])
    ranker = FeatureRanker("reg", models={"lin": LinearRegression()}, verbose=False,
                           shuffle="within_groups")
    ranker.fit(X[:HALF], Y_PAT[:HALF], ["p", "p2", "v"], shuffle_groups=PATIENTS[:HALF])
    with pytest.warns(UserWarning):
        report = ranker.rank_features(
            X[HALF:], Y_PAT[HALF:], scoring="r2", n_repeats=1, random_state=0,
            shuffle_groups=PATIENTS[HALF:],
            feature_groups={"mixed": ["p2", "v"], "alone": ["p"]},
        )

    assert report["constant_within_groups"] == ["alone"]


@pytest.mark.filterwarnings("ignore:No model beat")
def test_nan_counts_as_a_value():
    X = np.column_stack([X_PATIENT, X_VARYING]).copy()
    X[HALF + 0, 0] = np.nan  # one missing value inside patient 20 breaks constancy
    X[:HALF, 0] = 0.0
    ranker = FeatureRanker(
        "reg", models={"lin": make_pipeline(SimpleImputer(), LinearRegression())},
        verbose=False, shuffle="within_groups", missing="impute",
    )
    ranker.fit(X[:HALF], Y_PAT[:HALF], ["p", "v"], shuffle_groups=PATIENTS[:HALF])
    # Patient 20 now mixes NaN with a number; every other patient stays constant.
    report = ranker.rank_features(
        X[HALF:], Y_PAT[HALF:], scoring="r2", n_repeats=1, random_state=0,
        shuffle_groups=PATIENTS[HALF:],
    )

    assert report["constant_within_groups"] == []


def test_listing_uses_the_max_samples_subsample():
    labels = np.repeat(np.arange(6), 4)
    # Column z varies only inside group 5.
    z = np.zeros(24)
    z[20:] = [1.0, 2.0, 3.0, 4.0]
    X = np.column_stack([np.arange(24.0), z])
    y = 2.0 * X[:, 0]
    ranker = FeatureRanker("reg", models={"lin": LinearRegression()}, verbose=False,
                           shuffle="within_groups")
    ranker.fit(X, y, ["a", "z"], shuffle_groups=labels)
    outcomes = set()
    for seed in range(12):
        rows = _sample_whole_groups(np.random.default_rng([seed, 2]), labels, 20)
        expected = 5 not in labels[rows]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = ranker.rank_features(
                X, y, scoring="r2", n_repeats=1, random_state=seed,
                shuffle_groups=labels, max_samples=20,
            )
        assert ("z" in report["constant_within_groups"]) is expected
        outcomes.add(expected)

    assert outcomes == {True, False}


def test_resampled_split_reports_carry_the_list():
    ranker = patient_ranker(shuffle="within_groups")
    with pytest.warns(UserWarning, match="patient_x"):
        result = ranker.rank_resampled(
            X_PAT, Y_PAT, scoring="r2", cv=GroupKFold(3), groups=PATIENTS,
            feature_names=["patient_x", "varying_x"], n_repeats=1, random_state=0,
        )

    assert all(
        entry["report"]["constant_within_groups"] == ["patient_x"]
        for entry in result["splits"]
    )
    json.dumps(result, allow_nan=False)


def test_summary_warns_about_constant_groups():
    with pytest.warns(UserWarning):
        report = patient_report()
    findings = [f for f in summarize(report) if f["code"] == "constant_within_groups"]

    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"] == {"groups": ["patient_x"]}
    assert "patient_x" in findings[0]["message"]
    json.dumps(findings, allow_nan=False)


def test_summary_is_silent_without_constant_groups():
    report = rank()
    assert not [f for f in summarize(report) if f["code"] == "constant_within_groups"]
    report["constant_within_groups"] = []
    assert not [f for f in summarize(report) if f["code"] == "constant_within_groups"]


def test_default_report_has_no_new_keys():
    assert "constant_within_groups" not in rank()


@pytest.mark.filterwarnings("ignore:No model beat")
def test_a_patient_with_an_all_nan_column_is_still_constant():
    X = np.column_stack([X_PATIENT, X_VARYING]).copy()
    X[PATIENTS == 25, 0] = np.nan  # one patient has no value at all
    ranker = FeatureRanker(
        "reg", models={"lin": make_pipeline(SimpleImputer(), LinearRegression())},
        verbose=False, shuffle="within_groups", missing="impute",
    )
    ranker.fit(X[:HALF], Y_PAT[:HALF], ["p", "v"], shuffle_groups=PATIENTS[:HALF])
    with pytest.warns(UserWarning, match="constant within"):
        report = ranker.rank_features(
            X[HALF:], Y_PAT[HALF:], scoring="r2", n_repeats=1, random_state=0,
            shuffle_groups=PATIENTS[HALF:],
        )

    assert report["constant_within_groups"] == ["p"]


def test_resampled_report_lists_the_union_of_split_lists():
    ranker = patient_ranker(shuffle="within_groups")
    with pytest.warns(UserWarning):
        result = ranker.rank_resampled(
            X_PAT, Y_PAT, scoring="r2", cv=GroupKFold(3), groups=PATIENTS,
            feature_names=["patient_x", "varying_x"], n_repeats=1, random_state=0,
        )

    assert result["constant_within_groups"] == ["patient_x"]
    assert list(result).index("constant_within_groups") == list(result).index("shuffle") + 1
    json.dumps(result, allow_nan=False)


@pytest.mark.filterwarnings("ignore:No model beat")
def test_resampled_union_is_an_empty_list_without_constant_groups():
    ranker = FeatureRanker("reg", models={"lin": LinearRegression()}, verbose=False,
                           shuffle="within_groups")
    result = ranker.rank_resampled(
        X_PAT[:, 1:], Y_PAT, scoring="r2", cv=GroupKFold(3), groups=PATIENTS,
        feature_names=["varying_x"], n_repeats=1, random_state=0,
    )

    assert result["constant_within_groups"] == []
    assert not [f for f in summarize(result) if f["code"] == "constant_within_groups"]


def test_resampled_key_is_absent_in_other_modes():
    result = FeatureRanker(
        "reg", models={"lin": LinearRegression()}, verbose=False
    ).rank_resampled(
        X_PAT, Y_PAT, scoring="r2", cv=3, feature_names=["a", "b"], n_repeats=1,
        random_state=0,
    )

    assert "constant_within_groups" not in result


def test_summary_of_a_resampled_report_warns_about_constant_groups():
    ranker = patient_ranker(shuffle="within_groups")
    with pytest.warns(UserWarning):
        result = ranker.rank_resampled(
            X_PAT, Y_PAT, scoring="r2", cv=GroupKFold(3), groups=PATIENTS,
            feature_names=["patient_x", "varying_x"], n_repeats=1, random_state=0,
        )
    findings = [f for f in summarize(result) if f["code"] == "constant_within_groups"]

    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"] == {
        "groups": ["patient_x"], "splits": {"patient_x": [0, 1, 2]},
    }


# --- documented ablation example -------------------------------------------

USAGE = Path(__file__).parents[1] / "docs" / "usage.md"
NUMBER = re.compile(r"-?\d+\.\d+")


def test_documented_ablation_example_runs_and_matches_its_output(capsys):
    # The example lives in docs/usage.md (section "Ablation curve"); this runs it as shown.
    section = USAGE.read_text(encoding="utf-8").split("## ▍ Ablation curve", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    shown = re.search(r"```text\n(.*?)```", section, re.S).group(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "docs/usage.md", "exec"), {})
    printed = capsys.readouterr().out.strip().splitlines()
    expected = shown.strip().splitlines()

    assert len(printed) == len(expected)
    for got, want in zip(printed, expected):
        assert NUMBER.sub("#", got) == NUMBER.sub("#", want)
        for a, b in zip(NUMBER.findall(got), NUMBER.findall(want)):
            assert float(a) == pytest.approx(float(b), abs=0.011)
