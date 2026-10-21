"""Calibration study tests (fake rankers, no heavy fits). Run: python -m pytest benchmarks"""

import numpy as np
import pytest

import calibration as cal
import run


# Hand-made stability rows

def row(name, share=0.0, p=1.0, q=1.0, families=0):
    return {"feature_group": name, "above_shadow_share": share, "shadow_p_value": p,
            "shadow_q_value": q, "shadow_families": families}


def test_null_data_is_deterministic_and_independent_of_the_target():
    reg, again, clf = (cal.make_null(t, 100, 1000) for t in ("reg", "reg", "clf"))
    assert np.array_equal(reg.X, again.X) and np.array_equal(reg.y, again.y)
    assert np.array_equal(reg.X, clf.X)  # the features do not depend on the task or on y
    assert reg.X.shape == (1000, 10)
    assert reg.roles == {"signal": [], "neutral": [], "noise": reg.names}
    assert set(np.unique(clf.y)) == {0, 1} and 0.4 < clf.y.mean() < 0.6
    for data in (reg, clf):
        corr = [abs(np.corrcoef(data.X[:, i], data.y)[0, 1]) for i in range(10)]
        assert max(corr) < 0.15
    assert not np.array_equal(cal.make_null("reg", 101, 1000).X, reg.X)


def test_rules_use_the_selectors_thresholds_on_one_stability_table():
    stability = [
        row("f0", share=0.9, p=0.001, q=0.01, families=3),
        row("f1", share=0.5, p=0.04, q=0.06, families=2),
        row("f2", share=0.49, p=0.2, q=0.3, families=1),
        row("f3", share=0.0, p=1.0, q=1.0, families=0),
    ]
    picked = cal.apply_rules(stability)
    assert picked["share"] == ["f0", "f1"]
    assert picked["q"] == ["f0"]
    assert picked["fam1"] == ["f0", "f1", "f2"]
    assert picked["fam2"] == ["f0", "f1"]
    assert picked["fam3"] == ["f0"]
    assert list(picked) == list(cal.RULES)


def test_each_configuration_calls_the_ranker_once_and_all_rules_share_that_report():
    calls = []

    def ranker(task, X, y, names, group, mode, seed):
        calls.append((task, group, mode, seed))
        return {"stability": [row(n, share=1.0, p=0.0, q=0.0, families=3) for n in names[:2]]}

    runs = cal.run_study("null", ["reg", "clf"], ["fast"], ["joint", "independent"], [100, 101],
                         ranker=ranker, n_rows=50)
    assert len(calls) == len(set(calls)) == 8 and len(runs) == 8
    assert all(r["status"] == "ok" for r in runs)
    for r in runs:
        assert all(r["selected"][rule] == ["f0", "f1"] for rule in cal.RULES)
        assert [s["feature_group"] for s in r["stability"]] == ["f0", "f1"]


def test_run_records_ranker_errors_and_continues():
    def ranker(task, X, y, names, group, mode, seed):
        if seed == 100:
            raise RuntimeError("boom")
        return {"stability": [row("f0")]}

    runs = cal.run_study("null", ["reg"], ["fast"], ["joint"], [100, 101], ranker=ranker, n_rows=50)
    assert [r["status"] for r in runs] == ["error", "ok"] and "boom" in runs[0]["error"]


def test_signal_study_uses_the_benchmark_generator_and_scores_roles():
    seen = {}

    def oracle(task, X, y, names, group, mode, seed):
        data = run.make_data(task, seed, False, len(X))
        seen["same_data"] = np.array_equal(X, data.X) and np.array_equal(y, data.y)
        return {"stability": [row(n, share=1.0, p=0.0, q=0.0, families=3) for n in data.roles["signal"]]}

    runs = cal.run_study("signal", ["reg"], ["fast"], ["joint"], [100], ranker=oracle, n_rows=200)
    assert seen["same_data"]
    rows = cal.summarize_signal(runs)
    by_rule = {r["rule"]: r for r in rows}
    assert by_rule["q"]["recall"] == 1.0 and by_rule["q"]["noise"] == 0.0
    assert by_rule["q"]["interaction_recall"] == 1.0 and by_rule["q"]["copy"] == 0.0


def test_null_summary_counts_hits_and_selected_groups():
    def make(seed, ps, selected):
        return {"study": "null", "task": "reg", "group": "fast", "mode": "joint", "seed": seed, "status": "ok",
                "stability": [row(f"f{i}", p=p) for i, p in enumerate(ps)],
                "selected": {rule: selected.get(rule, []) for rule in cal.RULES}}

    runs = [make(100, [0.01, 0.5, 1.0, 1.0], {"fam1": ["f0"], "q": ["f0"]}),
            make(101, [1.0, 1.0, 1.0, 1.0], {})]
    (summary,) = cal.summarize_null(runs)
    assert (summary["n_runs"], summary["n_groups"], summary["hits"]) == (2, 8, 1)
    assert summary["fraction"] == pytest.approx(1 / 8)
    assert summary["ci_low"] < 1 / 8 < summary["ci_high"]
    assert summary["rules"]["fam1"] == {"frac_groups": 1 / 8, "frac_runs_any": 0.5}
    assert summary["rules"]["fam2"] == {"frac_groups": 0.0, "frac_runs_any": 0.0}


def test_rules_match_the_selectors_own_filters(monkeypatch):
    """Parity with ConsensusSelector.fit, using a stub ranker that returns a hand-made stability table."""
    from featranker import selection

    stability = [row(f"x{i}", share=sh, p=p, q=q, families=f) for i, (sh, p, q, f) in enumerate(
        [(0.9, 0.001, 0.01, 3), (0.5, 0.04, 0.05, 2), (0.49, 0.2, 0.051, 1), (0.0, 1.0, 1.0, 0), (0.7, 0.5, 0.04, 4)])]
    report = {"stability": stability, "feature_groups": {r["feature_group"]: [r["feature_group"]] for r in stability},
              "n_splits": 10, "shadow_test": {"min_p_value": 0.001}}

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def rank_resampled(self, *args, **kwargs):
            return report

    monkeypatch.setattr(selection, "FeatureRanker", Stub)
    X, y = np.zeros((6, 5)), np.zeros(6)
    settings = {"share": {"min_shadow_share": 0.5}, "q": {"max_shadow_q": 0.05},
                "fam1": {"min_shadow_families": 1}, "fam2": {"min_shadow_families": 2},
                "fam3": {"min_shadow_families": 3}}
    picked = cal.apply_rules(stability)
    assert set(settings) == set(cal.RULES)
    for rule, kwargs in settings.items():
        chosen = selection.ConsensusSelector("reg", "r2", **kwargs).fit(X, y).selected_groups_
        assert picked[rule] == chosen, rule


def test_null_summary_with_an_empty_stability_table_counts_the_run_but_no_groups():
    empty = {"study": "null", "task": "reg", "group": "fast", "mode": "joint", "seed": 100, "status": "ok",
             "stability": [], "selected": {rule: [] for rule in cal.RULES}}
    full = {**empty, "seed": 101, "stability": [row(f"f{i}", p=1.0) for i in range(10)]}
    (summary,) = cal.summarize_null([empty, full])
    assert (summary["n_runs"], summary["n_groups"], summary["hits"]) == (2, 10, 0)
    assert summary["rules"]["share"] == {"frac_groups": 0.0, "frac_runs_any": 0.0}
    with pytest.raises(ValueError):  # known limit of the frozen script: a cell with no groups at all cannot be summarised
        cal.summarize_null([empty])


# Decisions

def null_row(task="reg", group="fast", mode="joint", hits=10, n_groups=200, fam=(0.0, 0.0, 0.0)):
    from scipy.stats import binomtest

    ci = binomtest(hits, n_groups).proportion_ci(confidence_level=0.95, method="exact")
    rules = {rule: {"frac_groups": 0.0, "frac_runs_any": 0.0} for rule in cal.RULES}
    for n, value in zip((1, 2, 3), fam):
        rules[f"fam{n}"] = {"frac_groups": value, "frac_runs_any": value}
    return {"task": task, "group": group, "mode": mode, "n_runs": 20, "n_groups": n_groups, "hits": hits,
            "fraction": hits / n_groups, "ci_low": ci.low, "ci_high": ci.high, "rules": rules}


def test_decision_q_needs_both_the_point_estimate_and_the_upper_bound():
    allowed = lambda hits, n=200: cal.decide_q([null_row(hits=hits, n_groups=n)])[0]["allowed"]
    assert allowed(4)  # 2 %, upper bound about 5 %
    assert allowed(7)  # 3.5 %, upper bound just under 7.5 %
    assert not allowed(8)  # 4 %: upper bound above 7.5 %
    assert not allowed(10)  # 5 %: point estimate fine, upper bound above 7.5 %
    assert not allowed(11)  # point estimate above 5 %
    assert allowed(50, n=2000) is True  # 2.5 %, narrow interval
    out = cal.decide_q([null_row(hits=4), null_row(group="tree", hits=20)])
    assert "conservative for built-in `fast` models" in out[0]["text"]
    assert "excess" in out[1]["text"] and "10.0%" in out[1]["text"]
    assert "no conservativeness claim" in cal.decide_q([null_row(hits=8)])[0]["text"]


def sig_row(task, mode, rule, interaction, group="fast", e=1.0, copy=0.0):
    return {"task": task, "group": group, "mode": mode, "rule": rule, "n_runs": 10,
            "interaction_recall": interaction, "e_recall": e, "copy": copy, "recall": 1.0, "noise": 0.0}


def signal_for(task, recalls):
    return [sig_row(task, mode, f"fam{n}", rec) for n, rec in recalls.items() for mode in ("joint", "independent")]


def test_decision_f_picks_the_smallest_n_meeting_both_limits():
    nulls = [null_row(mode=m, group=g, fam=(0.20, 0.04, 0.0)) for m in ("joint", "independent") for g in ("fast", "tree")]
    signal = signal_for("reg", {1: 1.0, 2: 0.9, 3: 0.5})
    out = cal.decide_f(nulls, signal)["reg"]
    assert out["recommended"] == 2
    assert out["table"][1]["null_max"] == 0.20 and not out["table"][1]["passes"]
    assert out["table"][3]["interaction_recall_min"] == 0.5 and not out["table"][3]["passes"]


def test_decision_f_borderline_values_pass_and_one_bad_configuration_fails():
    nulls = [null_row(mode=m, fam=(0.05, 0.05, 0.05)) for m in ("joint", "independent")]
    assert cal.decide_f(nulls, signal_for("reg", {1: 0.8, 2: 0.8, 3: 0.8}))["reg"]["recommended"] == 1
    one_bad = nulls + [null_row(group="tree", mode="joint", fam=(0.051, 0.05, 0.05))]
    assert cal.decide_f(one_bad, signal_for("reg", {1: 1.0, 2: 0.8, 3: 0.8}))["reg"]["recommended"] == 2
    low = signal_for("reg", {1: 0.79, 2: 0.7, 3: 0.6})
    assert cal.decide_f(nulls, low)["reg"]["recommended"] is None
    both = [sig_row("reg", "joint", "fam1", 0.9), sig_row("reg", "independent", "fam1", 0.7)]
    assert cal.decide_f(nulls, both)["reg"]["table"][1]["interaction_recall_min"] == 0.7  # every fast mode must reach 0.8
    tree_only = [sig_row("reg", "joint", "fam1", 0.0, group="tree")]
    assert cal.decide_f(nulls, tree_only)["reg"]["recommended"] is None  # tree signal rows do not count


def s_inputs(e_joint, e_ind, copy_joint=0.0, copy_ind=0.0, null_joint=0.0, null_ind=0.0):
    signal = [sig_row("reg", "joint", "q", 1.0, e=e_joint, copy=copy_joint),
              sig_row("reg", "independent", "q", 1.0, e=e_ind, copy=copy_ind)]
    nulls = []
    for mode, value in (("joint", null_joint), ("independent", null_ind)):
        row_ = null_row(mode=mode)
        row_["rules"]["q"]["frac_groups"] = value
        nulls.append(row_)
    return nulls, signal


def test_decision_s_requires_a_gain_and_no_new_null_inflation():
    assert cal.decide_s(*s_inputs(0.5, 0.8))["recommend"]  # raises e recall
    assert cal.decide_s(*s_inputs(0.8, 0.8, copy_joint=0.2, copy_ind=0.6))["recommend"]  # keeps the copy more often
    assert not cal.decide_s(*s_inputs(0.8, 0.8))["recommend"]  # no gain
    assert not cal.decide_s(*s_inputs(0.5, 0.9, null_joint=0.04, null_ind=0.06))["recommend"]  # raised above 0.05
    assert cal.decide_s(*s_inputs(0.5, 0.9, null_joint=0.08, null_ind=0.06))["recommend"]  # not raised
    assert cal.decide_s(*s_inputs(0.5, 0.9, null_joint=0.04, null_ind=0.05))["recommend"]  # exactly 0.05 is not above
    assert not cal.decide_s(*s_inputs(0.9, 0.5))["recommend"]  # worse


def test_table_rendering_and_hash_are_stable():
    assert len(cal.script_sha256()) == 64
    results = {"runs": [], "decisions": {}, "null_summary": [null_row()], "signal_summary": signal_for("reg", {1: 1.0})}
    text = cal.render_markdown(results)
    assert "reg" in text and "fam1" in text
