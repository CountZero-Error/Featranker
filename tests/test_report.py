import builtins
import json
import pickle

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from featranker import FeatureRanker, Report, summarize

RNG = np.random.default_rng(0)
X = RNG.normal(size=(60, 3))
Y = 3.0 * X[:, 0] + 1.0 * X[:, 1] + 0.1 * RNG.normal(size=60)
NAMES = ["a", "b", "c"]
MODELS = {"lin": LinearRegression(), "ridge": Ridge()}


@pytest.fixture(scope="module")
def single():
    ranker = FeatureRanker(task="reg", models=MODELS, verbose=False).fit(X[:40], Y[:40], NAMES)
    return ranker.rank_features(X[40:], Y[40:], scoring="r2", n_repeats=2, random_state=0)


@pytest.fixture(scope="module")
def resampled():
    return FeatureRanker(task="reg", models=MODELS, verbose=False).rank_resampled(
        X, Y, feature_names=NAMES, scoring="r2", cv=3, n_repeats=1, random_state=0
    )


def test_reports_are_dict_compatible(single, resampled):
    for report in (single, resampled):
        assert isinstance(report, Report) and isinstance(report, dict)
        plain = dict(report)
        assert report == plain
        assert json.dumps(report, allow_nan=False) == json.dumps(plain, allow_nan=False)
        loaded = pickle.loads(pickle.dumps(report))
        assert isinstance(loaded, Report) and loaded == report
    assert all(isinstance(s["report"], Report) for s in resampled["splits"])


def test_summary(single, resampled):
    assert single.summary() == summarize(single)
    assert resampled.summary() == summarize(resampled)


def test_to_frame(single, resampled):
    pytest.importorskip("pandas")
    frame = single.to_frame()
    assert list(frame["feature_group"]) == [r["feature_group"] for r in single["consensus"]]
    assert "median_rank" in frame and "family_ranks" not in frame

    stability = resampled.to_frame()
    assert list(stability["feature_group"]) == [r["feature_group"] for r in resampled["stability"]]
    assert "top_k_share" in stability and "split_ranks" not in stability


def test_plot_single_and_resampled(single, resampled):
    for report, rows in ((single, "consensus"), (resampled, "stability")):
        ax = report.plot()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == [r["feature_group"] for r in report[rows]]


def test_plot_empty_consensus(single):
    empty = Report({**single, "consensus": []})

    ax = empty.plot()

    assert "no consensus" in " ".join(t.get_text() for t in ax.texts).lower()


@pytest.mark.parametrize("method, module, extra", [
    ("to_frame", "pandas", "pandas"), ("plot", "matplotlib", "featranker[plot]"),
])
def test_missing_optional_dependency(single, monkeypatch, method, module, extra):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == module or name.startswith(module + "."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=extra.replace("[", r"\[").replace("]", r"\]")):
        getattr(single, method)()


def test_plot_order_is_stable_on_reused_and_preinverted_axes(single):
    import matplotlib.pyplot as plt

    expected = [r["feature_group"] for r in single["consensus"]]
    ax = single.plot()
    single.plot(ax=ax)  # reusing the axes must not flip the order
    assert ax.yaxis_inverted()
    _, fresh = plt.subplots()
    fresh.invert_yaxis()
    single.plot(ax=fresh)  # already inverted axes stay top-down
    assert fresh.yaxis_inverted()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == expected
    plt.close("all")
