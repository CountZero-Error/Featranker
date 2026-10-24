import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker

RNG = np.random.default_rng(0)
N = 90
X = RNG.normal(size=(N, 3))
Y = 2 * X[:, 0] + 0.1 * RNG.normal(size=N)
NAMES = ["a", "b", "c"]
TRAIN, EVAL = slice(0, 60), slice(60, N)
HEX = r"^[0-9a-f]{64}$"
AUTO = {"method": "spearman-complete-linkage"}


def ranker(**kwargs):
    return FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False, **kwargs)


def single(X_train=X[TRAIN], X_eval=X[EVAL], y_eval=Y[EVAL], names=NAMES, **kwargs):
    r = ranker(**kwargs)
    r.fit(X_train, Y[TRAIN], feature_names=names)
    return r.rank_features(X_eval, y_eval, scoring="r2", n_repeats=2, random_state=0)


def single_weighted(train_weight, eval_weight):
    r = ranker()
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES, sample_weight=train_weight)
    return r.rank_features(
        X[EVAL], Y[EVAL], scoring="r2", n_repeats=2, random_state=0,
        sample_weight=eval_weight,
    )


def resampled(cv=3, data=X, sample_weight=None, **kwargs):
    return ranker(**kwargs).rank_resampled(
        data, Y, feature_names=NAMES, scoring="r2", cv=cv, n_repeats=2, random_state=0,
        sample_weight=sample_weight,
    )


# -- run block ----------------------------------------------------------------

def test_single_split_run_block():
    run = single()["run"]

    assert run["task"] == "reg" and run["group"] == "all"
    assert list(run) == ["task", "group", "fingerprint_scheme", "data_sha256", "train_data_sha256"]
    assert run["data_sha256"] != run["train_data_sha256"]
    for key in ("data_sha256", "train_data_sha256"):
        assert len(run[key]) == 64 and set(run[key]) <= set("0123456789abcdef")
    json.dumps(run, allow_nan=False)


def test_resampled_run_block_and_per_split_run():
    report = resampled()

    run = report["run"]
    assert list(run) == ["task", "group", "fingerprint_scheme", "data_sha256", "cv", "splits_sha256"]
    assert run["cv"] == 3
    assert "train_data_sha256" not in run
    assert len(run["splits_sha256"]) == 64
    json.dumps(run, allow_nan=False)
    for entry in report["splits"]:
        assert set(entry["report"]["run"]) >= {"data_sha256", "train_data_sha256"}
    assert report["splits"][0]["report"]["run"] != report["splits"][1]["report"]["run"]


def test_cv_is_the_splitter_repr():
    cv = KFold(3, shuffle=True, random_state=1)

    assert resampled(cv=cv)["run"]["cv"] == repr(cv)


def test_same_inputs_give_equal_run_blocks():
    assert single()["run"] == single()["run"]
    assert resampled()["run"] == resampled()["run"]


def test_one_evaluation_value_changed():
    changed = Y[EVAL].copy()
    changed[0] += 1e-9

    first, second = single()["run"], single(y_eval=changed)["run"]

    assert first["data_sha256"] != second["data_sha256"]
    assert first["train_data_sha256"] == second["train_data_sha256"]


def test_row_order_and_names_change_the_fingerprint():
    base = single()["run"]

    reordered = single(X_eval=X[EVAL][::-1], y_eval=Y[EVAL][::-1])["run"]
    renamed = single(names=["a", "b", "z"])["run"]

    assert reordered["data_sha256"] != base["data_sha256"]
    assert reordered["train_data_sha256"] == base["train_data_sha256"]
    assert renamed["data_sha256"] != base["data_sha256"]
    assert renamed["train_data_sha256"] != base["train_data_sha256"]


def test_different_splits_change_only_the_split_fingerprint():
    first = resampled(cv=KFold(3, shuffle=True, random_state=0))["run"]
    second = resampled(cv=KFold(3, shuffle=True, random_state=1))["run"]

    assert first["data_sha256"] == second["data_sha256"]
    assert first["splits_sha256"] != second["splits_sha256"]
    assert first["cv"] != second["cv"]


def test_preset_is_recorded():
    run = FeatureRanker("clf", "fast", verbose=False).rank_resampled(
        X, (Y > 0).astype(int), feature_names=NAMES, scoring="accuracy", cv=2, n_repeats=1
    )["run"]

    assert run["task"] == "clf" and run["group"] == "fast"


def test_categorical_and_missing_data_hash_stably():
    frame = pd.DataFrame(X[:, :2].copy(), columns=["a", "b"])
    frame["city"] = np.where(RNG.random(N) < 0.5, "x", "y")
    frame.loc[3, "a"] = np.nan
    frame.loc[5, "city"] = None

    def run(data):
        built_in = FeatureRanker("reg", "linear", verbose=False, missing="impute",
                                 categorical=["city"])
        return built_in.rank_resampled(
            data, Y, scoring="r2", cv=3, n_repeats=1, random_state=0
        )["run"]

    changed = frame.copy()
    changed.loc[0, "city"] = "z"

    assert run(frame) == run(frame.copy())
    assert run(changed)["data_sha256"] != run(frame)["data_sha256"]
    nan_moved = frame.copy()
    nan_moved.loc[3, "a"], nan_moved.loc[4, "a"] = frame.loc[4, "a"], np.nan
    assert run(nan_moved)["data_sha256"] != run(frame)["data_sha256"]


def test_cv_repr_has_no_memory_addresses():
    class Plain:  # no __repr__: the default one holds an address
        def split(self, X, y=None, groups=None):
            return KFold(3).split(X)

    first = [KFold(3, shuffle=True, random_state=np.random.RandomState(0)), Plain()]
    second = [KFold(3, shuffle=True, random_state=np.random.RandomState(0)), Plain()]

    for a, b in zip(first, second):
        run_a, run_b = resampled(cv=a)["run"], resampled(cv=b)["run"]
        assert run_a == run_b and " at 0x" not in run_a["cv"]


def test_cv_iterable_of_splits_is_recorded_as_such():
    splits = list(KFold(3).split(X))

    assert resampled(cv=splits)["run"]["cv"] == "iterable of splits"


def test_weights_are_fingerprinted_only_when_given():
    weights = np.linspace(1, 2, N)
    plain, weighted = single()["run"], single_weighted(weights[TRAIN], weights[EVAL])["run"]

    assert "weights_sha256" not in plain and "train_weights_sha256" not in plain
    assert len(weighted["weights_sha256"]) == 64
    assert len(weighted["train_weights_sha256"]) == 64
    assert weighted["data_sha256"] == plain["data_sha256"]
    json.dumps(weighted, allow_nan=False)


def test_changing_only_the_weights_changes_the_run_block():
    weights = np.linspace(1, 2, N)
    changed = weights.copy()
    changed[0] += 1e-6
    changed_eval = weights.copy()
    changed_eval[-1] += 1e-6

    base = single_weighted(weights[TRAIN], weights[EVAL])["run"]
    train_changed = single_weighted(changed[TRAIN], weights[EVAL])["run"]
    eval_changed = single_weighted(weights[TRAIN], changed_eval[EVAL])["run"]

    assert train_changed["train_weights_sha256"] != base["train_weights_sha256"]
    assert train_changed["weights_sha256"] == base["weights_sha256"]
    assert eval_changed["weights_sha256"] != base["weights_sha256"]
    assert eval_changed["train_weights_sha256"] == base["train_weights_sha256"]

    first = resampled(sample_weight=weights)["run"]
    assert first == resampled(sample_weight=weights)["run"]
    assert first != resampled(sample_weight=changed)["run"]
    assert "weights_sha256" not in resampled()["run"]


def test_numpy_and_pandas_numeric_data_hash_the_same():
    frame = pd.DataFrame(X, columns=NAMES)

    assert resampled(data=frame)["run"] == resampled(data=X)["run"]


def test_object_array_hashing_is_fast_and_deterministic():
    from featranker.importance import _fingerprint

    rng = np.random.default_rng(1)
    values = np.empty((20000, 5), dtype=object)
    values[:, :3] = rng.normal(size=(20000, 3))
    values[:, 3:] = rng.choice(["x", "y", None], size=(20000, 2))
    target = rng.normal(size=20000)

    assert _fingerprint(NAMES + ["d", "e"], values, target) == _fingerprint(
        NAMES + ["d", "e"], values.copy(), target
    )


def test_old_pickle_without_train_fingerprint_reports_null():
    r = ranker()
    r.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)
    del r.train_data_sha256_

    old = pickle.loads(pickle.dumps(r))
    report = old.rank_features(X[EVAL], Y[EVAL], scoring="r2", n_repeats=1)

    assert report["run"]["train_data_sha256"] is None
    json.dumps(report["run"], allow_nan=False)


# -- auto_groups threshold and record ------------------------------------------

E = RNG.normal(size=200)
PAIR = np.column_stack([E, 0.65 * E + 0.76 * RNG.normal(size=200), RNG.normal(size=200)])
PAIR_Y = E + 0.1 * RNG.normal(size=200)


def pair_selector(**kwargs):
    return ConsensusSelector(
        "reg", "r2", k=1, models={"ridge": Ridge()}, cv=2, n_repeats=1, random_state=0,
        feature_groups="auto", **kwargs,
    )


def test_looser_threshold_groups_a_moderate_pair_and_is_recorded():
    default = pair_selector().fit(PAIR, PAIR_Y)
    loose = pair_selector(auto_groups_threshold=0.5).fit(PAIR, PAIR_Y)

    assert "x0+x1" not in default.report_["feature_groups"]
    assert default.report_["run"]["auto_groups"] == {"threshold": 0.7, **AUTO}
    assert loose.report_["feature_groups"]["x0+x1"] == ["x0", "x1"]
    assert loose.report_["run"]["auto_groups"] == {"threshold": 0.5, **AUTO}


def test_auto_groups_threshold_is_a_clonable_parameter():
    s = ConsensusSelector("reg", "r2", k=1, auto_groups_threshold=0.5)

    assert s.get_params()["auto_groups_threshold"] == 0.5
    assert clone(s).auto_groups_threshold == 0.5
    assert ConsensusSelector("reg", "r2", k=1).auto_groups_threshold == 0.7


@pytest.mark.parametrize("threshold", [0, 1.5, True, "high"])
def test_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        pair_selector(auto_groups_threshold=threshold).fit(PAIR, PAIR_Y)


@pytest.mark.parametrize("feature_groups", [None, {"pair": ["x0", "x1"]}])
def test_no_auto_groups_record_without_automatic_grouping(feature_groups):
    s = pair_selector(auto_groups_threshold=0.5)
    s.set_params(feature_groups=feature_groups)

    s.fit(PAIR, PAIR_Y)

    assert "auto_groups" not in s.report_["run"]


def test_auto_groups_threshold_numpy_scalar_stays_json_serialisable():
    from featranker.evidence import auto_groups_record

    record = auto_groups_record(np.float32(0.5))

    assert json.dumps(record) and type(record["threshold"]) is float


def test_old_selector_pickle_gets_the_default_threshold():
    s = pair_selector()
    state = {k: v for k, v in s.__dict__.items() if k != "auto_groups_threshold"}
    old = ConsensusSelector.__new__(ConsensusSelector)

    old.__setstate__(state)

    assert old.auto_groups_threshold == 0.7


def run_cli(monkeypatch, tmp_path, *extra):
    def write_csv(path, n_rows):
        e = np.random.default_rng(n_rows).normal(size=n_rows)
        noise = np.random.default_rng(n_rows + 1).normal(size=(n_rows, 2))
        lines = ["e,f,g,target"]
        lines += [
            ",".join(map(str, [a, a + 0.05 * b, c, 2 * a]))
            for a, (b, c) in zip(e, noise)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(tmp_path / "train.csv", 60)
    write_csv(tmp_path / "eval.csv", 30)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "featranker", "--task", "reg", "--group", "linear", "--quiet",
        "--train-data", "train.csv", "--eval-data", "eval.csv", "--target", "target",
        "--scoring", "r2", "--n-repeats", "2", "--random-state", "0",
        "--output", str(tmp_path / "report.json"), *extra,
    ])
    assert cli_module.main() == 0
    return json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_cli_run_block_and_default_threshold(monkeypatch, tmp_path):
    report = run_cli(monkeypatch, tmp_path, "--feature-groups", "auto")

    assert report["feature_groups"]["e+f"] == ["e", "f"]
    assert report["run"]["auto_groups"] == {"threshold": 0.7, **AUTO}
    assert report["run"]["task"] == "reg" and len(report["run"]["data_sha256"]) == 64


def test_cli_threshold_option(monkeypatch, tmp_path):
    report = run_cli(
        monkeypatch, tmp_path, "--feature-groups", "auto", "--auto-groups-threshold", "1"
    )

    assert "e+f" not in report["feature_groups"]
    assert report["run"]["auto_groups"]["threshold"] == 1.0


def test_cli_without_auto_has_no_auto_groups_record(monkeypatch, tmp_path):
    assert "auto_groups" not in run_cli(monkeypatch, tmp_path)["run"]


def test_cli_invalid_threshold_is_an_error(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path, "--feature-groups", "auto",
                "--auto-groups-threshold", "0")


# -- warning when splits did not vote ------------------------------------------

class FailsWithoutRowZero(Ridge):
    """Raises in fit() when row 0 (value 0.0 in column 0) is not in the training rows."""

    def fit(self, X, y, **kwargs):
        if X[:, 0].min() > 0:
            raise RuntimeError("no row zero")
        return super().fit(X, y, **kwargs)


class WrongOnRowZero(Ridge):
    """Predicts nonsense for any evaluation set holding row 0: that split has no consensus."""

    def predict(self, X):
        if X[:, 0].min() == 0:
            return np.full(len(X), 1e6)
        return super().predict(X)


SEQ_X = np.column_stack([np.arange(60.0), RNG.normal(size=60), RNG.normal(size=60)])
SEQ_Y = 2.0 * SEQ_X[:, 0] + 0.1 * RNG.normal(size=60)


def vote_warnings(model):
    s = ConsensusSelector("reg", "r2", k=1, models={"m": model}, cv=KFold(5), n_repeats=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s.fit(SEQ_X, SEQ_Y)
    return s, [str(w.message) for w in caught if "did not vote" in str(w.message)]


def test_failed_split_warns_and_still_selects():
    s, messages = vote_warnings(FailsWithoutRowZero())

    assert len(messages) == 1
    assert "1 of 5 splits did not vote" in messages[0] and "misses" in messages[0]
    assert s.selected_groups_ == ["x0"]


def test_empty_consensus_split_warns():
    s, messages = vote_warnings(WrongOnRowZero())

    assert len(messages) == 1 and "1 of 5 splits did not vote" in messages[0]


def test_no_split_warning_when_every_split_votes():
    _, messages = vote_warnings(Ridge())

    assert messages == []
