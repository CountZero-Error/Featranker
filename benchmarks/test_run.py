"""Benchmark tests (fake selectors, no heavy fits). Run: python -m pytest benchmarks"""

import numpy as np
import pytest

import run


def test_data_is_deterministic_per_seed_and_permutes_roles():
    first, again = run.make_data("reg", 0, False), run.make_data("reg", 0, False)
    assert np.array_equal(first.X, again.X) and np.array_equal(first.y, again.y)
    assert first.roles == again.roles
    assert first.X.shape == (run.N_ROWS, 10)
    layouts = {tuple(run.make_data("reg", s, False).roles["signal"]) for s in range(5)}
    assert len(layouts) > 1
    roles = first.roles
    assert len(roles["signal"]) == 5 and len(roles["neutral"]) == 1 and len(roles["noise"]) == 4
    assert set(roles["signal"] + roles["neutral"] + roles["noise"]) == set(first.names)


def test_correlated_copy_tracks_a_signal_column():
    data = run.make_data("reg", 2, False)
    copy = data.names.index(data.roles["neutral"][0])
    best = max(abs(np.corrcoef(data.X[:, copy], data.X[:, data.names.index(n)])[0, 1])
               for n in data.roles["signal"])
    assert best > 0.95


def test_classification_labels_are_binary_and_hard_variant_is_noisier():
    clf = run.make_data("clf", 1, False)
    assert set(np.unique(clf.y)) == {0, 1}
    easy, hard = run.make_data("reg", 1, False), run.make_data("reg", 1, True)
    assert np.array_equal(easy.X, hard.X)
    assert np.std(hard.y - easy.y) > 0.5  # extra noise on the same signal


def test_scoring_counts_recall_and_noise_and_ignores_copy():
    data = run.make_data("reg", 0, False)
    roles = data.roles
    idx = lambda names: [data.names.index(n) for n in names]
    perfect = run.score(data, idx(roles["signal"]))
    assert perfect["recall"] == 1.0 and perfect["noise_selected"] == 0
    with_copy = run.score(data, idx(roles["signal"] + roles["neutral"]))
    assert with_copy["recall"] == 1.0 and with_copy["noise_selected"] == 0
    assert with_copy["neutral_selected"] == 1
    mixed = run.score(data, idx(roles["signal"][:2] + roles["noise"][:3]))
    assert mixed["recall"] == pytest.approx(0.4) and mixed["noise_selected"] == 3
    assert run.score(data, [])["recall"] == 0.0


def fakes(data_for):
    def oracle(task, X, y, seed):
        data = data_for(task, seed)
        return [data.names.index(n) for n in data.roles["signal"]]

    def missing(task, X, y, seed):
        import not_a_real_module_for_benchmark  # noqa: F401

    def broken(task, X, y, seed):
        raise RuntimeError("boom")

    return {"oracle": oracle, "everything": lambda t, X, y, s: list(range(X.shape[1])),
            "missing": missing, "broken": broken}


def test_run_records_missing_and_errors_and_continues():
    results = run.run_benchmark(
        tasks=["reg"], variants=["standard"], seeds=[0, 1], selectors=fakes(
            lambda task, seed: run.make_data(task, seed, False)))
    by_name = {}
    for r in results["runs"]:
        by_name.setdefault(r["selector"], []).append(r)
    assert [r["status"] for r in by_name["missing"]] == ["missing", "missing"]
    assert "not_a_real_module_for_benchmark" in by_name["missing"][0]["error"]
    assert by_name["broken"][0]["status"] == "error" and "boom" in by_name["broken"][0]["error"]
    assert all(r["status"] == "ok" and r["recall"] == 1.0 for r in by_name["oracle"])
    assert all(r["noise_selected"] == 4 for r in by_name["everything"])  # runs after the failures
    assert all("seconds" in r for r in by_name["oracle"])


def test_summary_and_table_skip_unavailable_selectors():
    results = run.run_benchmark(
        tasks=["reg"], variants=["standard"], seeds=[0, 1, 2], selectors=fakes(
            lambda task, seed: run.make_data(task, seed, False)))
    rows = {r["selector"]: r for r in run.summarize(results["runs"])}
    assert rows["oracle"]["recall_mean"] == 1.0 and rows["oracle"]["n_runs"] == 3
    assert rows["everything"]["noise_mean"] == 4.0 and rows["everything"]["noise_std"] == 0.0
    assert rows["missing"]["status"] == "missing" and rows["broken"]["status"] == "error"
    table = run.render_table(results)
    assert "oracle" in table and "1.00 ± 0.00" in table and "missing" in table


def test_role_table_reports_per_role_hits():
    oracle = lambda task, X, y, seed: [run.make_data(task, seed, False).names.index(n)
                                      for n in run.make_data(task, seed, False).roles["signal"]]
    results = run.run_benchmark(["reg"], ["standard"], [0, 1], {"oracle": oracle})
    row = [line for line in run.role_table(results).splitlines() if line.startswith("| oracle")][0]
    assert row.split("|")[2:7] == [" 1.0 "] * 5 and row.split("|")[7] == " 0.0 "


def test_planted_columns_drive_the_target():
    for seed in range(3):
        data = run.make_data("reg", seed, False)
        col = lambda name: data.X[:, data.names.index(name)]
        a, b, c, d, e = (col(n) for n in data.roles["signal"])
        corr = lambda u: abs(np.corrcoef(u, data.y)[0, 1])
        assert corr(a) > 0.3 and corr(e) > 0.2 and corr((b > 0).astype(float)) > 0.2 and corr(c * d) > 0.2
        assert max(corr(col(n)) for n in data.roles["noise"]) < 0.15
        assert corr(c) < 0.15 and corr(d) < 0.15  # the pair matters only together
    clf = run.make_data("clf", 0, False)
    a_col = clf.X[:, clf.names.index(clf.roles["signal"][0])]
    assert np.corrcoef(a_col, clf.y)[0, 1] > 0.2


REAL_SELECTORS = {
    "featranker_share_fast": [],
    "permutation_rule": [],
    "boruta": ["boruta"],
    "select_by_shuffling": ["feature_engine"],
}


@pytest.mark.parametrize("name", REAL_SELECTORS)
def test_real_selector_wrappers_return_valid_column_indices(name):
    for module in REAL_SELECTORS[name]:
        pytest.importorskip(module)
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, run.N_FEATURES))
    y = 10 * X[:, 7] + 0.1 * rng.standard_normal(200)  # one overwhelming column
    out = run.SELECTORS[name]("reg", X, y, 0)
    selected = out[0] if isinstance(out, tuple) else out
    assert 7 in selected  # index mapping is right
    assert len(set(selected)) == len(selected) and all(0 <= int(i) < run.N_FEATURES for i in selected)
    if isinstance(out, tuple):
        assert set(out[1]) == {"above_shadow_share", "family_shares"}


def test_diagnostics_are_labelled_by_role_and_stored_with_roles():
    data = run.make_data("reg", 0, False, 200)
    labels = run.role_labels(data)
    assert sorted(labels.values()) == list(run.ROLE_KEYS) and len(labels) == 6
    fake = lambda task, X, y, seed: ([0], {"above_shadow_share": {"x0": 0.5, "x1": 0.1},
                                           "family_shares": {"tree": {"x0": 1.0}}})
    row = run.run_benchmark(["reg"], ["standard"], [0], {"fake": fake}, n_rows=200)["runs"][0]
    assert row["roles"] == data.roles
    assert row["diagnostics"]["above_shadow_share"][labels["f0"]] == 0.5
    assert row["diagnostics"]["family_shares"] == {"tree": {labels["f0"]: 1.0}}
    assert "tree" in run.family_table({"runs": [row]})


def test_default_run_uses_the_preregistered_selectors_only(monkeypatch):
    seen = []
    monkeypatch.setattr(run, "SELECTORS", {n: (lambda t, X, y, s, n=n: seen.append(n) or [0])
                                           for n in (*run.PREREGISTERED, "featranker_share_tree")})
    run.run_benchmark(["reg"], ["standard"], [0], n_rows=50)
    assert seen == list(run.PREREGISTERED)


def _model(family, in_consensus, flags):
    return {"family": family, "in_consensus": in_consensus,
            "importance": {g: {"above_shadow": f} for g, f in flags.items()}}


def test_shadow_diagnostics_exact_values_from_a_hand_built_report():
    split0 = {"models": {
        "m1": _model("tree", True, {"x0": True, "x1": False}),
        "m2": _model("tree", True, {"x0": False, "x1": False}),
        "m3": _model("linear", True, {"x0": True, "x1": False}),
        "weak": _model("linear", False, {"x0": False, "x1": True}),  # failed the dummy baseline: no vote
    }}
    split1 = {"models": {
        "m1": _model("tree", True, {"x0": True, "x1": True}),
        "m2": _model("tree", True, {"x0": True, "x1": False}),
        "m3": _model("linear", True, {"x0": False, "x1": False}),
    }}
    report = {"splits": [{"report": split0}, {"report": split1}],
              "stability": [{"feature_group": "x0", "above_shadow_share": 0.75},
                            {"feature_group": "x1", "above_shadow_share": 0.125}]}
    out = run._shadow_diagnostics(report)
    assert out["above_shadow_share"] == {"x0": 0.75, "x1": 0.125}
    # tree: x0 -> (0.5 + 1.0) / 2 = 0.75, x1 -> (0 + 0.5) / 2 = 0.25; linear: x0 -> (1 + 0) / 2, x1 -> 0
    assert out["family_shares"] == {"tree": {"x0": 0.75, "x1": 0.25}, "linear": {"x0": 0.5, "x1": 0.0}}
