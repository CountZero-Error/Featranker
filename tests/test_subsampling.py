import json
import sys

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import featranker.cli as cli_module
from featranker import ConsensusSelector, FeatureRanker

N = 300
N_TRAIN = 100
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, 3))
Y = 2 * X[:, 0] + X[:, 1] + 0.1 * RNG.normal(size=N)
W = RNG.uniform(0.5, 2.0, size=N)
NAMES = ["a", "b", "c"]
TRAIN, EVAL = slice(0, N_TRAIN), slice(N_TRAIN, N)  # 200 evaluation rows


def fitted(**kwargs):
    ranker = FeatureRanker("reg", models={"ridge": Ridge()}, verbose=False, **kwargs)
    return ranker.fit(X[TRAIN], Y[TRAIN], feature_names=NAMES)


def rank(ranker=None, **kwargs):
    kwargs = {"n_repeats": 2, "random_state": 0, "scoring": "r2", **kwargs}
    return (ranker or fitted()).rank_features(X[EVAL], Y[EVAL], **kwargs)


@pytest.mark.parametrize("value", [0, -1, 1.5, 0.0, -0.5, True, "5", float("nan"), [3]])
def test_invalid_max_samples_raises_before_scoring(value):
    calls = []

    def scorer(model, X_, y_, **kwargs):
        calls.append(1)
        return 0.0

    with pytest.raises(ValueError, match="max_samples"):
        rank(scoring=scorer, max_samples=value)
    assert calls == []


def test_invalid_max_samples_in_rank_resampled_raises():
    with pytest.raises(ValueError, match="max_samples"):
        fitted().rank_resampled(X, Y, scoring="r2", cv=2, feature_names=NAMES, max_samples=0)


def test_default_report_has_no_subsample_keys_and_equals_none():
    plain, explicit = rank(), rank(max_samples=None)
    assert "max_samples" not in plain and "n_eval_rows" not in plain
    assert plain["models"] == explicit["models"]


def test_every_score_sees_the_same_rows():
    seen = []

    def scorer(model, X_, y_, sample_weight=None):
        seen.append((np.array(X_), np.array(y_), sample_weight))
        return r2_score(y_, model.predict(X_), sample_weight=sample_weight)

    fitted(shadows=True, shadow_seed=0).rank_features(
        X[EVAL], Y[EVAL], scoring=scorer, n_repeats=2, random_state=0,
        sample_weight=W[EVAL], max_samples=50,
    )

    # dummy, baseline, permuted and shadow-permuted scores
    assert len(seen) > 6 and {len(y_) for _, y_, _ in seen} == {50}
    row_of = {float(y_): i for i, y_ in enumerate(Y[EVAL])}
    first_rows = [row_of[float(y_)] for y_ in seen[0][1]]
    for X_, y_, w in seen:
        rows = [row_of[float(v)] for v in y_]
        assert rows == first_rows
        assert np.array_equal(w, W[EVAL][rows])
        # a call permutes at most one of the three real columns; the rest match their rows
        intact = [np.array_equal(X_[:, c], X[EVAL][rows, c]) for c in range(3)]
        assert sum(intact) >= 2


def test_record_has_max_samples_and_n_eval_rows_and_strict_json():
    report = rank(max_samples=0.5)
    assert report["max_samples"] == 0.5 and report["n_eval_rows"] == 100
    json.dumps(dict(report), allow_nan=False)
    assert rank(max_samples=50)["n_eval_rows"] == 50


@pytest.mark.parametrize("value, plain", [(np.int64(50), 50), (np.float32(0.5), 0.5)])
def test_numpy_scalar_max_samples_is_stored_as_plain_number(value, plain):
    report = rank(max_samples=value)
    assert type(report["max_samples"]) is type(plain) and report["max_samples"] == plain
    json.dumps(dict(report), allow_nan=False)
    resampled = fitted().rank_resampled(
        X, Y, scoring="r2", cv=2, n_repeats=2, random_state=0, feature_names=NAMES,
        max_samples=value,
    )
    assert type(resampled["max_samples"]) is type(plain)
    json.dumps(dict(resampled), allow_nan=False)


@pytest.mark.parametrize(
    "value, rows", [(1000, 200), (200, 200), (1, 1), (1.0, 200), (0.001, 1), (0.333, 66)]
)
def test_size_rules(value, rows):
    assert rank(max_samples=value, scoring="neg_mean_squared_error")["n_eval_rows"] == rows


def test_reproducible_and_seed_dependent():
    first, again = rank(max_samples=0.5), rank(max_samples=0.5)
    assert first["models"] == again["models"]
    assert first["baseline_score"] == again["baseline_score"]
    other = rank(max_samples=0.5, random_state=1)
    assert other["baseline_score"] != first["baseline_score"]


def test_run_record_fingerprints_full_eval_data():
    assert rank(max_samples=50)["run"] == rank()["run"]


def test_works_with_shadows_evidence_and_dataframe_input():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=NAMES)
    ranker = FeatureRanker(
        "reg", models={"ridge": Ridge()}, verbose=False, shadows=True, shadow_seed=0,
        model_input="dataframe",
    ).fit(frame.iloc[TRAIN], Y[TRAIN])
    report = ranker.rank_features(
        frame.iloc[EVAL], Y[EVAL], scoring="r2", n_repeats=2, random_state=0, max_samples=40
    )
    assert report["n_eval_rows"] == 40 and report["shadows"] is True
    assert report["failures"] == []


def test_shap_and_evidence_unchanged_but_overfit_gap_uses_subsampled_baseline():
    pytest.importorskip("shap")
    ranker = fitted(evidence=True)
    full, sub = rank(ranker), rank(ranker, max_samples=10)
    assert sub["evidence"] == full["evidence"]
    assert sub["models"]["ridge"]["shap"] == full["models"]["ridge"]["shap"]
    model = sub["models"]["ridge"]
    assert model["overfit_gap"] == pytest.approx(model["train_score"] - model["evaluation_score"])


def test_rank_resampled_subsamples_each_split():
    report = fitted().rank_resampled(
        X, Y, scoring="r2", cv=3, n_repeats=2, random_state=0, feature_names=NAMES,
        max_samples=0.5
    )
    assert report["max_samples"] == 0.5
    for split in report["splits"]:
        assert split["report"]["n_eval_rows"] == split["n_eval"] // 2
    assert "max_samples" not in fitted().rank_resampled(
        X, Y, scoring="r2", cv=3, n_repeats=2, random_state=0, feature_names=NAMES
    )


def test_consensus_selector_passes_max_samples():
    from sklearn.base import clone

    selector = ConsensusSelector(
        "reg", "r2", k=1, models={"ridge": Ridge()}, cv=3, n_repeats=2,
        random_state=0, max_samples=0.5,
    )
    assert clone(selector).get_params()["max_samples"] == 0.5
    selector.fit(X, Y)
    for split in selector.report_["splits"]:
        assert split["report"]["n_eval_rows"] == split["n_eval"] // 2


def write_csv(path, n, rng):
    lines = ["signal,noise,target"]
    for _ in range(n):
        s, z = rng.normal(size=2)
        lines.append(f"{s},{z},{2 * s + 0.1 * rng.normal()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli_argv(tmp_path, *extra):
    return [
        "featranker", "--task", "reg", "--group", "linear",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--scoring", "r2", "--n-repeats", "2",
        "--random-state", "0", "--quiet", "--output", str(tmp_path / "out.json"), *extra,
    ]


@pytest.mark.parametrize("text, value, rows", [("20", 20, 20), ("0.5", 0.5, 20)])
def test_cli_max_samples(monkeypatch, tmp_path, text, value, rows):
    rng = np.random.default_rng(0)
    write_csv(tmp_path / "train.csv", 60, rng)
    write_csv(tmp_path / "eval.csv", 40, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--max-samples", text))

    assert cli_module.main() == 0

    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["max_samples"] == value and report["n_eval_rows"] == rows


@pytest.mark.parametrize("text", ["0", "1.5", "abc"])
def test_cli_rejects_bad_max_samples(monkeypatch, tmp_path, capsys, text):
    rng = np.random.default_rng(0)
    write_csv(tmp_path / "train.csv", 60, rng)
    write_csv(tmp_path / "eval.csv", 40, rng)
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, "--max-samples", text))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "max" in capsys.readouterr().err.lower()


def test_subsample_uses_its_own_seed_stream():
    seen = []

    def scorer(model, X_, y_):
        seen.append(np.array(y_))
        return r2_score(y_, model.predict(X_))

    rank(scoring=scorer, random_state=3, max_samples=50)
    expected = np.sort(np.random.default_rng([3, 2]).choice(200, 50, replace=False))
    assert np.array_equal(seen[0], Y[EVAL][expected])


def test_selector_pickled_before_max_samples_loads_with_none():
    state = ConsensusSelector("reg", "r2", k=1).__getstate__()
    del state["max_samples"]
    loaded = ConsensusSelector.__new__(ConsensusSelector)
    loaded.__setstate__(state)
    assert loaded.max_samples is None


def test_subsample_missing_a_class_fails_with_a_clear_error():
    from sklearn.linear_model import LogisticRegression

    y_train = np.arange(N_TRAIN) % 2
    y_eval = np.zeros(N - N_TRAIN, dtype=int)
    y_eval[0] = 1  # a single positive evaluation row
    seed = next(
        s for s in range(100)
        if 0 not in np.random.default_rng([s, 2]).choice(200, 10, replace=False)
    )
    ranker = FeatureRanker("clf", models={"lr": LogisticRegression()}, verbose=False)
    ranker.fit(X[TRAIN], y_train, feature_names=NAMES)

    with pytest.raises(RuntimeError, match="No models ranked.*Only one class present"):
        ranker.rank_features(
            X[EVAL], y_eval, scoring="roc_auc", n_repeats=2, random_state=seed,
            max_samples=10,
        )
