import json
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, get_scorer, make_scorer, mean_absolute_error, r2_score, recall_score,
)

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker

N = 300
N_TRAIN = 100
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, 3))
Y = 2 * X[:, 0] + X[:, 1] + 0.1 * RNG.normal(size=N)
W = RNG.uniform(0.5, 2.0, size=N)
NAMES = ["a", "b", "c"]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, N)
BOTH = ["r2", "neg_mean_absolute_error"]


def fitted(**kwargs):
    ranker = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False, **kwargs)
    return ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)


def rank(ranker=None, **kwargs):
    kwargs = {"n_repeats": 3, "random_state": 0, "scoring": BOTH, **kwargs}
    return (ranker or fitted()).rank_features(X[EVAL], Y[EVAL], **kwargs)


def broken(model, X_, y_):
    raise ValueError("secondary boom")


@pytest.mark.parametrize("scoring", [[], (), ["r2", 3], ["r2", ["r2"]], 3, ["r2", "nope"]])
def test_invalid_scoring_raises(scoring):
    with pytest.raises((ValueError, TypeError, KeyError)):
        rank(scoring=scoring)


def test_invalid_list_raises_before_any_scoring():
    calls = []

    def recorder(model, X_, y_):
        calls.append(1)
        return 0.0

    with pytest.raises(ValueError, match="unique"):
        rank(scoring=[recorder, "r2", "r2"])
    assert calls == []


def test_make_scorer_objects_are_named_by_their_score_function():
    rng = np.random.default_rng(1)
    Xc = rng.normal(size=(200, 2))
    yc = (Xc[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    ranker = FeatureRanker("clf", models={"lr": LogisticRegression()}, verbose=False)
    ranker.fit(Xc[:100], yc[:100], feature_names=["a", "b"])
    report = ranker.rank_features(
        Xc[100:], yc[100:], n_repeats=2, random_state=0,
        scoring=[
            "accuracy",
            make_scorer(f1_score, average="macro"),
            make_scorer(recall_score, average="macro"),
        ],
    )
    assert report["secondary_scoring"] == ["f1_score", "recall_score"]
    assert report["failures"] == []


def test_colliding_scorer_names_error_says_how_to_fix_it():
    scoring = [make_scorer(f1_score, average="macro"), make_scorer(f1_score, average="micro")]
    with pytest.raises(ValueError, match="unique.*__name__"):
        rank(scoring=scoring)


def test_duplicate_names_raise_value_error():
    with pytest.raises(ValueError, match="unique"):
        rank(scoring=["r2", "r2"])


def test_one_element_list_and_tuple_equal_a_plain_string():
    plain = rank(scoring="r2")
    for scoring in (["r2"], ("r2",)):
        report = rank(scoring=scoring)
        assert report["models"] == plain["models"]
        assert "secondary_scoring" not in report and "secondary" not in report["models"]["ridge"]
        assert set(report) == set(plain)


def test_primary_is_unchanged_by_secondary_scorers():
    plain = rank(scoring="r2", ranker=fitted(shadows=True, shadow_seed=1))
    both = rank(scoring=BOTH, ranker=fitted(shadows=True, shadow_seed=1))
    assert both["scoring"] == "r2"
    assert both["consensus"] == plain["consensus"]
    assert both["baseline_score"] == plain["baseline_score"]
    model_plain, model_both = plain["models"]["ridge"], both["models"]["ridge"]
    for key in model_plain:
        assert model_both[key] == model_plain[key]
    assert set(model_both) - set(model_plain) == {"secondary"}


def test_secondary_values_match_manual_scoring():
    ranker = fitted()
    report = rank(ranker)
    model = ranker.trained_models["ridge"]
    entry = report["models"]["ridge"]["secondary"]["neg_mean_absolute_error"]
    expected = -mean_absolute_error(Y[EVAL], model.predict(X[EVAL]))
    assert entry["evaluation_score"] == pytest.approx(expected)
    assert set(entry["importance"]) == set(NAMES)
    for group in entry["importance"].values():
        assert set(group) == {"mean", "std"}
    # The dominant feature hurts MAE the most when permuted.
    means = {g: v["mean"] for g, v in entry["importance"].items()}
    assert max(means, key=means.get) == "a"
    assert report["secondary_scoring"] == ["neg_mean_absolute_error"]
    assert list(report["secondary_baseline_scores"]) == ["neg_mean_absolute_error"]


def test_secondary_importance_is_the_mean_of_per_repeat_drops():
    seen = []

    def recorder(model, X_, y_):
        seen.append(np.array(X_))
        return -mean_absolute_error(y_, model.predict(X_))

    ranker = fitted()
    report = rank(ranker, scoring=["r2", recorder], n_repeats=2)
    model = ranker.trained_models["ridge"]
    baseline = -mean_absolute_error(Y[EVAL], model.predict(X[EVAL]))
    # seen[0] is the dummy, seen[1] the model baseline, then 3 groups x 2 repeats
    drops = [baseline + mean_absolute_error(Y[EVAL], model.predict(Xp)) for Xp in seen[2:]]
    entry = report["models"]["ridge"]["secondary"]["recorder"]["importance"]
    assert entry["a"]["mean"] == pytest.approx(np.mean(drops[0:2]))
    assert entry["c"]["mean"] == pytest.approx(np.mean(drops[4:6]))


def test_every_scorer_sees_the_same_permuted_arrays():
    seen = {"p": [], "s": []}

    def make(key):
        def scorer(model, X_, y_):
            seen[key].append(np.array(X_))
            return r2_score(y_, model.predict(X_))

        scorer.__name__ = key
        return scorer

    rank(scoring=[make("p"), make("s")], n_repeats=2)
    assert len(seen["p"]) == len(seen["s"]) > 6
    for a, b in zip(seen["p"], seen["s"]):
        assert np.array_equal(a, b)


def test_secondary_failure_keeps_the_model_ranked_with_null_entry():
    report = rank(scoring=["r2", broken, "neg_mean_absolute_error"])
    model = report["models"]["ridge"]
    assert model["secondary"]["broken"] is None
    assert model["secondary"]["neg_mean_absolute_error"] is not None
    assert report["secondary_baseline_scores"]["broken"] is None
    failures = [f for f in report["failures"] if f["stage"] == "evidence"]
    assert [f["model"] for f in failures] == ["ridge"]
    assert "secondary boom" in failures[0]["message"] and "broken" in failures[0]["message"]
    assert report["consensus"][0]["n_models"] == 1
    json.dumps(dict(report), allow_nan=False)


def test_secondary_failing_only_when_permuted_is_null_too():
    state = {"calls": 0}

    def late(model, X_, y_):
        state["calls"] += 1
        if state["calls"] > 3:
            raise RuntimeError("late boom")
        return 0.0

    report = rank(scoring=["r2", late])
    assert report["models"]["ridge"]["secondary"]["late"] is None
    assert [f["stage"] for f in report["failures"]] == ["evidence"]


def test_non_finite_secondary_is_a_failure():
    def nan_scorer(model, X_, y_):
        return float("nan")

    report = rank(scoring=["r2", nan_scorer])
    assert report["models"]["ridge"]["secondary"]["nan_scorer"] is None
    assert [f["stage"] for f in report["failures"]] == ["evidence"]
    json.dumps(dict(report), allow_nan=False)


def test_dummy_secondary_scores_and_unavailable_dummy_scores():
    report = rank()
    dummy = DummyRegressor().fit(X[TRAIN], Y[TRAIN])
    expected = get_scorer("neg_mean_absolute_error")(dummy, X[EVAL], Y[EVAL])
    assert report["secondary_baseline_scores"]["neg_mean_absolute_error"] == pytest.approx(expected)
    # A dummy failure is null, not a failure entry.
    report = rank(scoring=["r2", broken])
    assert report["secondary_baseline_scores"] == {"broken": None}
    assert not any(f["model"] == "baseline" for f in report["failures"])


def test_strict_json_and_secondary_with_weights_subsampling_shadows_dataframe():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=NAMES)
    ranker = FeatureRanker(
        "reg", models={"ridge": Ridge()}, verbose=False, shadows=True, shadow_seed=0,
        model_input="dataframe",
    ).fit(frame.iloc[TRAIN], Y[TRAIN], sample_weight=W[TRAIN])
    kwargs = {"n_repeats": 2, "random_state": 0, "sample_weight": W[EVAL], "max_samples": 50}
    both = ranker.rank_features(frame.iloc[EVAL], Y[EVAL], scoring=BOTH, **kwargs)
    plain = ranker.rank_features(frame.iloc[EVAL], Y[EVAL], scoring="r2", **kwargs)
    assert both["models"]["ridge"]["importance"] == plain["models"]["ridge"]["importance"]
    assert both["failures"] == [] and both["n_eval_rows"] == 50
    entry = both["models"]["ridge"]["secondary"]["neg_mean_absolute_error"]
    array_ranker = FeatureRanker(
        "reg", models={"ridge": Ridge()}, verbose=False, shadows=True, shadow_seed=0,
    ).fit(X[TRAIN], Y[TRAIN], sample_weight=W[TRAIN], feature_names=NAMES)
    array = array_ranker.rank_features(X[EVAL], Y[EVAL], scoring=BOTH, **kwargs)
    expected = array["models"]["ridge"]["secondary"]["neg_mean_absolute_error"]
    assert entry["evaluation_score"] == pytest.approx(expected["evaluation_score"])
    assert entry["importance"] == expected["importance"]
    json.dumps(dict(both), allow_nan=False)


def test_secondary_scorers_are_weighted_and_see_the_subsample():
    seen = []

    def weighted_mae(model, X_, y_, sample_weight=None):
        seen.append((len(y_), sample_weight))
        return -mean_absolute_error(y_, model.predict(X_), sample_weight=sample_weight)

    rank(scoring=["r2", weighted_mae], sample_weight=W[EVAL], max_samples=40, n_repeats=2)
    assert seen and all(n == 40 and w is not None and len(w) == 40 for n, w in seen)


def test_summary_and_prompt_still_work_with_secondary_scorers():
    from featranker import to_prompt

    report = rank()
    assert report.summary()
    with pytest.warns(FutureWarning):
        prompt = to_prompt(report, anonymize=True)
    assert "FINDINGS" in prompt
    assert "neg_mean_absolute_error" not in prompt


def broken_too(model, X_, y_):
    raise ValueError("second boom")


def test_summary_lists_a_model_once_when_several_secondary_scorers_fail():
    report = rank(scoring=["r2", broken, broken_too])
    (finding,) = [f for f in report.summary() if f["code"] == "failures"]
    assert finding["evidence"]["models"] == ["ridge"]
    assert "1 model(s)" in finding["message"] and "ridge, ridge" not in finding["message"]


def test_resampled_summary_lists_a_split_once_per_model():
    resampled = fitted().rank_resampled(
        X, Y, scoring=["r2", broken, broken_too], cv=2, n_repeats=2, random_state=0,
        feature_names=NAMES,
    )
    (finding,) = [f for f in resampled.summary() if f["code"] == "failures"]
    splits = finding["evidence"]["models"]["ridge"]
    assert splits == sorted(set(splits))
    assert "0, 0" not in finding["message"]


def test_secondary_failure_with_shadows_keeps_shadow_evidence():
    ranker = fitted(shadows=True, shadow_seed=0)
    report = rank(ranker, scoring=["r2", broken])
    model = report["models"]["ridge"]
    assert model["secondary"]["broken"] is None
    assert "shadow_max" in model
    assert [f["stage"] for f in report["failures"]] == ["evidence"]
    json.dumps(dict(report), allow_nan=False)


def test_overflowing_secondary_importance_is_that_scorers_failure():
    def huge(model, X_, y_):
        # Each drop is finite, but their mean overflows.
        return 1.7e308 if np.array_equal(X_, X[EVAL]) else 0.0

    report = rank(scoring=["r2", huge])
    assert report["models"]["ridge"]["secondary"]["huge"] is None
    assert [f["stage"] for f in report["failures"]] == ["evidence"]
    json.dumps(dict(report), allow_nan=False)


def test_rank_resampled_split_reports_carry_secondary_and_stability_is_unchanged():
    kwargs = {"cv": 3, "n_repeats": 2, "random_state": 0, "feature_names": NAMES}
    both = fitted().rank_resampled(X, Y, scoring=BOTH, **kwargs)
    plain = fitted().rank_resampled(X, Y, scoring="r2", **kwargs)
    assert both["scoring"] == "r2" and both["secondary_scoring"] == ["neg_mean_absolute_error"]
    assert both["stability"] == plain["stability"]
    assert "secondary_scoring" not in plain
    for split in both["splits"]:
        assert split["report"]["models"]["ridge"]["secondary"]["neg_mean_absolute_error"]
    json.dumps(dict(both), allow_nan=False)


def test_rank_resampled_rejects_a_bad_list_early():
    with pytest.raises(ValueError, match="unique"):
        fitted().rank_resampled(X, Y, scoring=["r2", "r2"], cv=2, feature_names=NAMES)


def test_consensus_selector_accepts_a_list():
    selector = ConsensusSelector(
        "reg", BOTH, k=1, models={"ridge": Ridge()}, cv=3, n_repeats=2, random_state=0
    )
    assert clone(selector).get_params()["scoring"] == BOTH
    selector.fit(X, Y)
    assert selector.report_["secondary_scoring"] == ["neg_mean_absolute_error"]
    single = ConsensusSelector(
        "reg", "r2", k=1, models={"ridge": Ridge()}, cv=3, n_repeats=2, random_state=0
    ).fit(X, Y)
    assert selector.report_["stability"] == single.report_["stability"]
    assert selector.get_support().tolist() == single.get_support().tolist()


def write_csv(path, n, rng):
    lines = ["signal,noise,target"]
    for _ in range(n):
        s, z = rng.normal(size=2)
        lines.append(f"{s},{z},{2 * s + 0.1 * rng.normal()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli_argv(tmp_path, scoring):
    return [
        "featranker", "--task", "reg", "--group", "linear",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--scoring", scoring, "--n-repeats", "2",
        "--random-state", "0", "--quiet", "--output", str(tmp_path / "out.json"),
    ]


def test_cli_comma_separated_scoring(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    write_csv(tmp_path / "train.csv", 60, rng)
    write_csv(tmp_path / "eval.csv", 40, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "r2,neg_mean_absolute_error"))

    assert cli_module.main() == 0

    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["scoring"] == "r2"
    assert report["secondary_scoring"] == ["neg_mean_absolute_error"]


def test_cli_strips_whitespace_around_scorer_names(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    write_csv(tmp_path / "train.csv", 60, rng)
    write_csv(tmp_path / "eval.csv", 40, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "r2, neg_mean_absolute_error"))

    assert cli_module.main() == 0

    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["secondary_scoring"] == ["neg_mean_absolute_error"]


@pytest.mark.parametrize("scoring", ["r2,r2", "r2,nope", "r2,", ",", "r2, ,r2"])
def test_cli_rejects_bad_scoring_list(monkeypatch, tmp_path, scoring):
    rng = np.random.default_rng(0)
    write_csv(tmp_path / "train.csv", 60, rng)
    write_csv(tmp_path / "eval.csv", 40, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, scoring))

    with pytest.raises((SystemExit, ValueError, KeyError)):
        cli_module.main()
    assert not (tmp_path / "out.json").exists()


def test_classification_list_with_probability_scorer():
    rng = np.random.default_rng(1)
    Xc = rng.normal(size=(200, 2))
    yc = (Xc[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    ranker = FeatureRanker("clf", models={"lr": LogisticRegression()}, verbose=False)
    ranker.fit(Xc[:100], yc[:100], feature_names=["a", "b"])
    report = ranker.rank_features(
        Xc[100:], yc[100:], scoring=["accuracy", "neg_log_loss"], n_repeats=2, random_state=0
    )
    assert report["scoring"] == "accuracy"
    assert report["models"]["lr"]["secondary"]["neg_log_loss"] is not None
    assert report["failures"] == []
