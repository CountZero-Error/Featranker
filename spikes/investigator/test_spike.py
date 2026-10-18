"""Spike tests (scripted fake LLM). Run: python -m pytest spikes/investigator"""

import json

import numpy as np
import pytest

import spike


def fake_llm(replies):
    """An llm(prompt) -> str that returns scripted replies and records prompts."""
    queue = [r if isinstance(r, str) else json.dumps(r) for r in replies]
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return queue.pop(0)

    llm.prompts = prompts
    return llm


@pytest.fixture(scope="module")
def lab():
    return spike.build_lab(0)


def roles(lab):
    return lab["bench"]["roles"]


def finding(features, mechanism, model=None):
    item = {"features": features, "mechanism": mechanism, "reason": "test"}
    if model is not None:
        item["model"] = model
    return item


def step(claim, path, op, value, action):
    return {
        "hypothesis": "test hypothesis",
        "claim": claim,
        "prediction": {"path": path, "op": op, "value": value},
        "action": action,
    }


# 2.1 benchmark


def test_benchmark_is_deterministic_and_permutes_roles():
    first, again, other = spike.make_benchmark(0), spike.make_benchmark(0), spike.make_benchmark(1)
    assert np.array_equal(first["X_train"], again["X_train"])
    assert first["roles"] == again["roles"]
    assert first["roles"] != other["roles"]
    assert first["X_train"].shape == (700, 10) and first["X_eval"].shape == (300, 10)
    assert first["names"] == [f"f{i}" for i in range(10)]


def test_ground_truth_lists_mechanisms_and_noise():
    bench = spike.make_benchmark(3)
    r = bench["roles"]
    planted = [r["linear"], r["nonlinear"], *r["interaction"], *r["correlated"]]
    assert len(set(planted)) == 6 and len(r["noise"]) == 4
    assert set(planted) | set(r["noise"]) == set(bench["names"])
    assert len(spike.scored_items(bench)) == 5
    e, f = (bench["names"].index(name) for name in r["correlated"])
    assert np.corrcoef(bench["X_eval"][:, e], bench["X_eval"][:, f])[0, 1] > 0.95


# 2.3 parsing and scoring


def test_scoring_matches_order_free_and_ignores_duplicates(lab):
    r = roles(lab)
    findings = [
        finding([r["linear"]], "linear"),
        finding([r["linear"]], "linear"),
        finding(list(reversed(r["interaction"])), "interaction"),
        finding(list(reversed(r["correlated"])), "correlated"),
        finding([r["nonlinear"]], "nonlinear"),
        finding([], "overfit", model="decision_tree"),
    ]
    result = spike.score(findings, lab["bench"], lab["gaps"])
    assert result["recall"] == 1.0 and result["false_findings"] == 0
    assert len(result["matched"]) == 5


def test_scoring_counts_each_false_kind(lab):
    r, noise = roles(lab), roles(lab)["noise"]
    false = [
        finding([noise[0]], "linear"),
        finding([r["linear"]], "noise"),
        finding([r["nonlinear"]], "linear"),
        finding([r["linear"]], "nonlinear"),
        finding([r["linear"], r["nonlinear"]], "interaction"),
        finding([r["linear"], r["nonlinear"]], "correlated"),
        finding([], "overfit", model="ridge"),
    ]
    result = spike.score(false, lab["bench"], lab["gaps"])
    assert result["false_findings"] == len(false) and result["recall"] == 0.0


def test_scoring_neutral_findings_are_neither(lab):
    r = roles(lab)
    neutral = [finding([r["correlated"][0]], "linear"), finding([r["interaction"][0]], "nonlinear")]
    result = spike.score(neutral, lab["bench"], lab["gaps"])
    assert result == {"recall": 0.0, "matched": [], "false_findings": 0}


def test_wrong_arity_is_false(lab):
    # D8: grouped or short findings no longer pass as neutral.
    r = roles(lab)
    wrong = [
        finding([r["linear"], r["correlated"][1]], "linear"),
        finding([r["nonlinear"], r["interaction"][0]], "nonlinear"),
        finding(r["noise"], "noise"),
        finding([r["interaction"][0]], "interaction"),
        finding(r["correlated"] + [r["linear"]], "correlated"),
        finding([r["linear"]], "overfit", model="decision_tree"),
    ]
    result = spike.score(wrong, lab["bench"], lab["gaps"])
    assert result == {"recall": 0.0, "matched": [], "false_findings": len(wrong)}
    claim = {"verdict": "confirmed", "claim": wrong[0]}
    assert spike.confirmed_false([claim], lab["bench"], lab["gaps"]) == 1


def test_parse_findings_handles_fences_bad_entries_and_garbage():
    good = {"findings": [finding(["f1"], "linear"), {"mechanism": "bogus"}, finding([], "overfit")]}
    parsed = spike.parse_findings("```json\n" + json.dumps(good) + "\n```")
    assert parsed == [finding(["f1"], "linear")]  # bad mechanism and model-less overfit dropped
    assert spike.parse_findings("not json") is None
    assert spike.parse_findings(json.dumps({"claims": []})) is None


# 3.1 evidence battery


def test_battery_contents(lab):
    battery = lab["battery"]
    assert set(battery["scores"]) == set(spike.MODEL_NAMES)
    for scores in battery["scores"].values():
        assert set(scores) == {"train_r2", "eval_r2"}
    for model in spike.MODEL_NAMES:
        assert set(battery["shap"][model]) == set(lab["bench"]["names"])
        for entry in battery["shap"][model].values():
            assert set(entry) == {"mean_abs", "direction"}
    pairs = battery["correlated_pairs"]
    assert all(abs(p["r"]) >= 0.5 for p in pairs)
    assert sorted(roles(lab)["correlated"]) in [sorted(p["features"]) for p in pairs]
    linear = roles(lab)["linear"]
    assert battery["shap"]["ridge"][linear]["direction"] > 0.9


# 3.3 actions


def test_actions_return_documented_keys(lab):
    r = roles(lab)
    dep = spike.run_action({"name": "shap_dependence", "model": "hist_gb", "feature": r["nonlinear"]}, lab)
    assert set(dep) == {"bins", "linearity_r2"} and len(dep["bins"]) == 10
    inter = spike.run_action(
        {"name": "shap_interaction", "model": "hist_gb", "features": r["interaction"]}, lab
    )
    assert set(inter) == {"strength", "share"}
    refit = spike.run_action({"name": "refit", "model": "decision_tree", "params": {"max_depth": 3}}, lab)
    assert set(refit) == {"train_r2", "eval_r2"}
    merged = spike.run_action({"name": "merge_rank", "features": r["correlated"]}, lab)
    assert set(merged) == {"merged_median_rank", "merged_position", "member_median_ranks"}


def test_planted_signals_show_in_actions(lab):
    r = roles(lab)
    act = lambda **a: spike.run_action(a, lab)  # noqa: E731
    threshold = act(name="shap_dependence", model="hist_gb", feature=r["nonlinear"])
    linear = act(name="shap_dependence", model="hist_gb", feature=r["linear"])
    assert linear["linearity_r2"] > threshold["linearity_r2"]
    pair = act(name="shap_interaction", model="hist_gb", features=r["interaction"])
    other = act(name="shap_interaction", model="hist_gb", features=[r["linear"], r["noise"][0]])
    assert pair["share"] > other["share"]


@pytest.mark.parametrize(
    "action",
    [
        {"name": "refit", "model": "decision_tree", "params": {"criterion": "absolute_error"}},
        {"name": "refit", "model": "decision_tree", "params": {"max_depth": 0}},
        {"name": "refit", "model": "ridge", "params": {"max_depth": 3}},
        {"name": "shap_interaction", "model": "ridge", "features": ["f0", "f1"]},
        {"name": "shap_dependence", "model": "nope", "feature": "f0"},
        {"name": "shap_dependence", "model": "ridge", "feature": "f99"},
        {"name": "merge_rank", "features": ["f0"]},
        {"name": "rm_rf"},
    ],
)
def test_bad_actions_are_rejected(lab, action):
    with pytest.raises(ValueError):
        spike.run_action(action, lab)


# 4.1 arms and loop


def test_narrator_and_evidence_make_one_call(lab):
    for arm in ("narrator", "evidence"):
        llm = fake_llm([{"findings": [finding(["f1"], "linear")]}])
        run = spike.run_arm(arm, lab, llm)
        assert run["calls"] == 1 and run["findings"] == [finding(["f1"], "linear")]
    narrator = fake_llm([{"findings": []}])
    spike.run_arm("narrator", lab, narrator)
    evidence = fake_llm([{"findings": []}])
    spike.run_arm("evidence", lab, evidence)
    assert "EVIDENCE" not in narrator.prompts[0] and "EVIDENCE" in evidence.prompts[0]
    assert "ACTIONS" not in evidence.prompts[0]


def test_investigator_verdicts_and_budget(lab):
    r = roles(lab)
    claim = finding([r["nonlinear"]], "nonlinear")
    dep = {"name": "shap_dependence", "model": "hist_gb", "feature": r["nonlinear"]}
    replies = [
        step(claim, "linearity_r2", "<", 0.99, dep),   # confirmed
        step(claim, "linearity_r2", ">", 0.99, dep),   # refuted
        step(claim, "no.such.path", ">", 0, dep),      # invalid path
        {"findings": [claim]},
    ]
    # an invalid step forces the next reply to be final, so turn 4 is final
    llm = fake_llm(replies)
    run = spike.run_arm("investigator", lab, llm)
    assert [e["verdict"] for e in run["trail"]] == ["confirmed", "refuted", "invalid"]
    assert run["calls"] == 4 and run["findings"] == [claim]
    assert "ACTIONS" in llm.prompts[0] and "final" in llm.prompts[3].lower()


def test_investigator_stops_after_three_followups(lab):
    r = roles(lab)
    claim = finding([r["nonlinear"]], "nonlinear")
    dep = {"name": "shap_dependence", "model": "hist_gb", "feature": r["nonlinear"]}
    replies = [step(claim, "linearity_r2", "<", 0.99, dep)] * 4
    run = spike.run_arm("investigator", lab, fake_llm(replies))
    assert run["calls"] == 4 and len(run["trail"]) == 3
    assert run["findings"] is None  # a step instead of the forced final reply


def test_invalid_step_forces_final(lab):
    replies = [{"hypothesis": "h"}, {"findings": []}]
    run = spike.run_arm("investigator", lab, fake_llm(replies))
    assert run["trail"][0]["verdict"] == "invalid" and run["calls"] == 2
    assert run["findings"] == []


def test_bad_action_is_invalid_not_a_crash(lab):
    claim = finding(["f0"], "linear")
    replies = [step(claim, "eval_r2", ">", 0, {"name": "rm_rf"}), {"findings": []}]
    run = spike.run_arm("investigator", lab, fake_llm(replies))
    assert run["trail"][0]["verdict"] == "invalid" and "error" in run["trail"][0]["result"]


# 4.3 replay


def test_replay_reproduces_trail(lab):
    r = roles(lab)
    claim = finding(r["interaction"], "interaction")
    replies = [
        step(claim, "share", ">", 0.01,
             {"name": "shap_interaction", "model": "hist_gb", "features": r["interaction"]}),
        step(finding([], "overfit", model="decision_tree"), "eval_r2", ">", 0.0,
             {"name": "refit", "model": "decision_tree", "params": {"max_depth": 4}}),
        {"findings": []},
    ]
    run = spike.run_arm("investigator", lab, fake_llm(replies))
    fresh = spike.build_lab(0)
    assert spike.replay(run["trail"], fresh) == [(e["result"], e["verdict"]) for e in run["trail"]]


# 4.5 confirmed-false


def test_confirmed_false_counts_only_confirmed_false_claims(lab):
    r = roles(lab)
    dep = {"name": "shap_dependence", "model": "hist_gb", "feature": r["linear"]}
    trail = [
        {"claim": finding([r["linear"]], "nonlinear"), "verdict": "confirmed"},  # false + confirmed
        {"claim": finding([r["linear"]], "nonlinear"), "verdict": "refuted"},
        {"claim": finding([r["linear"]], "linear"), "verdict": "confirmed"},     # true
        {"claim": None, "verdict": "invalid", "action": dep},
    ]
    assert spike.confirmed_false(trail, lab["bench"], lab["gaps"]) == 1


# 4.7 decision rule


def seeds(nar, evi, inv, false=(0, 0, 0), confirmed_false=0):
    rows = []
    for i in range(5):
        rows.append({
            "narrator": {"recall": nar[i], "false_findings": false[0]},
            "evidence": {"recall": evi[i], "false_findings": false[1]},
            "investigator": {"recall": inv[i], "false_findings": false[2],
                             "confirmed_false": confirmed_false if i == 0 else 0},
        })
    return rows


def test_decide_go_both():
    d = spike.decide(seeds([0.2] * 5, [0.4] * 5, [0.8, 0.8, 0.8, 0.8, 0.4]))
    assert d["outcome"] == "go_1.1_1.2"


def test_decide_needs_four_of_five_seeds():
    # mean gap 0.2 but investigator beats evidence in only 3 seeds
    d = spike.decide(seeds([0.2] * 5, [0.4] * 5, [0.8, 0.8, 0.4, 0.4, 0.6]))
    assert d["outcome"] == "go_1.1"


def test_decide_confirmed_false_blocks_loop():
    d = spike.decide(seeds([0.2] * 5, [0.4] * 5, [0.8] * 5, confirmed_false=2))
    assert d["outcome"] == "go_1.1"


def test_decide_no_go_and_borderline():
    d = spike.decide(seeds([0.4] * 5, [0.4] * 5, [0.4] * 5))
    assert d["outcome"] == "no_go" and not d["borderline"]
    near = spike.decide(seeds([0.4] * 5, [0.6] * 5, [0.6] * 5, false=(0, 0.2, 0.2)))
    assert near["outcome"] == "go_1.1" and near["borderline"]  # gap 0.2 sits on the threshold
