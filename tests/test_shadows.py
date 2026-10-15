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
