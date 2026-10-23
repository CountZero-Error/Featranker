import copy
import json

import pytest

from featranker import summarize


def _model(family, groups, in_consensus=True):
    """groups: name -> (mean, std, rank)."""
    return {
        "family": family,
        "in_consensus": in_consensus,
        "evaluation_score": 0.5,
        "importance": {
            name: {"values": [mean], "mean": mean, "std": std, "rank": rank}
            for name, (mean, std, rank) in groups.items()
        },
    }


def _row(group, median_rank, n_families=3):
    return {
        "feature_group": group,
        "median_rank": median_rank,
        "mean_rank": median_rank,
        "rank_std": 0.0,
        "n_models": 3,
        "n_families": n_families,
        "family_ranks": {},
    }


def clean_report():
    groups = {"a": (0.5, 0.1, 1), "b": (0.2, 0.05, 2)}
    return {
        "baseline_score": 0.1,
        "models": {
            "m1": _model("F1", groups),
            "m2": _model("F2", groups),
            "m3": _model("F3", groups),
        },
        "consensus": [_row("a", 1.0), _row("b", 2.0)],
        "failures": [],
    }


def codes(findings):
    return [finding["code"] for finding in findings]


def test_clean_report_has_no_warnings():
    findings = summarize(clean_report())

    assert not [f for f in findings if f["severity"] == "warning"]
    assert findings == []


def test_summary_is_pure_deterministic_and_strict_json():
    report = clean_report()
    report["baseline_score"] = None
    report["consensus"][1]["median_rank"] = 1.0
    before = copy.deepcopy(report)

    first = summarize(report)

    assert first == summarize(report)
    assert report == before
    json.dumps(first, allow_nan=False)
    for finding in first:
        assert set(finding) == {"code", "severity", "message", "evidence"}
        assert finding["severity"] in {"info", "warning"}
        assert isinstance(finding["message"], str) and finding["message"]


def test_gate_off_is_first_warning():
    report = clean_report()
    report["baseline_score"] = None
    report["failures"] = [
        {"model": "dummy_baseline", "stage": "fit", "error_type": "ValueError",
         "message": "boom"}
    ]

    findings = summarize(report)

    assert findings[0]["code"] == "gate_off"
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"]["dummy_failures"] == report["failures"]


def test_empty_consensus_warns():
    report = clean_report()
    report["consensus"] = []

    findings = summarize(report)

    assert codes(findings) == ["empty_consensus"]
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"] == {"n_ranked_models": 3}


def _failure(model, stage):
    return {"model": model, "stage": stage, "error_type": "E", "message": "m"}


def test_initialization_only_failures_are_info():
    report = clean_report()
    report["failures"] = [_failure("xgboost", "initialization")]

    findings = summarize(report)

    assert codes(findings) == ["failures"]
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"stage": "initialization", "models": ["xgboost"]}


def test_failures_one_finding_per_stage_in_fixed_order():
    report = clean_report()
    report["failures"] = [
        _failure("svr", "ranking"),
        _failure("xgboost", "initialization"),
        _failure("qda", "fit"),
        _failure("lda", "fit"),
    ]

    findings = summarize(report)

    assert [(f["evidence"]["stage"], f["severity"]) for f in findings] == [
        ("initialization", "info"),
        ("fit", "warning"),
        ("ranking", "warning"),
    ]
    assert findings[1]["evidence"]["models"] == ["qda", "lda"]


def test_few_families_warns():
    report = clean_report()
    report["consensus"] = [_row("a", 1.0, n_families=2), _row("b", 2.0, n_families=2)]

    findings = summarize(report)

    assert codes(findings) == ["few_families"]
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"] == {"n_families": 2, "n_models": 3}


def test_report_without_family_fields_skips_few_families():
    report = clean_report()
    for row in report["consensus"]:
        del row["n_families"], row["family_ranks"]

    assert "few_families" not in codes(summarize(report))


def test_tied_groups_one_finding_per_tie_set():
    report = clean_report()
    groups = {"a": (0.5, 0.1, 1), "b": (0.5, 0.1, 1), "c": (0.2, 0.0, 3),
              "d": (0.2, 0.0, 3), "e": (0.1, 0.0, 5)}
    report["models"] = {m: _model(f, groups) for m, f in
                        [("m1", "F1"), ("m2", "F2"), ("m3", "F3")]}
    report["consensus"] = [_row("a", 1.0), _row("b", 1.0), _row("c", 3.0),
                           _row("d", 3.0), _row("e", 5.0)]

    findings = [f for f in summarize(report) if f["code"] == "tied_groups"]

    assert [f["evidence"] for f in findings] == [
        {"median_rank": 1.0, "groups": ["a", "b"]},
        {"median_rank": 3.0, "groups": ["c", "d"]},
    ]
    assert all(f["severity"] == "info" for f in findings)


def test_within_noise_needs_a_majority_of_voting_models():
    report = clean_report()
    noisy = {"a": (0.5, 0.1, 1), "b": (0.0, 0.05, 2)}
    report["models"] = {
        "m1": _model("F1", noisy),
        "m2": _model("F2", noisy),
        "m3": _model("F3", {"a": (0.5, 0.1, 1), "b": (0.2, 0.05, 2)}),
        "gated": _model("F4", noisy, in_consensus=False),
    }

    findings = [f for f in summarize(report) if f["code"] == "within_noise"]

    assert len(findings) == 1
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {
        "groups": {"b": {"within_noise": 2, "voting_models": 3}}
    }


def test_within_noise_not_emitted_at_exactly_half():
    report = clean_report()
    report["models"]["m3"] = _model("F3", {"a": (0.5, 0.1, 1), "b": (0.0, 0.0, 2)})
    report["models"]["m4"] = _model("F4", {"a": (0.5, 0.1, 1), "b": (0.0, 0.0, 2)})

    assert "within_noise" not in codes(summarize(report))


def test_rule_order():
    report = clean_report()
    report["baseline_score"] = None
    report["failures"] = [_failure("svr", "ranking")]
    report["consensus"] = [_row("a", 1.0, 2), _row("b", 1.0, 2)]
    report["models"]["m1"]["importance"]["b"].update(mean=0.0, std=0.1)
    report["models"]["m2"]["importance"]["b"].update(mean=0.0, std=0.1)

    assert codes(summarize(report)) == [
        "gate_off", "failures", "few_families", "tied_groups", "within_noise",
    ]


def test_rejects_non_dict():
    with pytest.raises(TypeError, match="dict"):
        summarize([])


def test_missing_key_is_named():
    with pytest.raises(ValueError, match="baseline_score"):
        summarize({"models": {}})


def test_resampled_report_missing_key_is_named():
    with pytest.raises(ValueError, match="stability"):
        summarize({"evaluation_mode": "resampled", "splits": [], "failures": []})


# -- resampled reports ----------------------------------------------------------

def _split(index, baseline=0.1, consensus=None):
    consensus = [_row("a", 1.0)] if consensus is None else consensus
    return {"split": index, "n_train": 8, "n_eval": 2,
            "report": {"baseline_score": baseline, "consensus": consensus}}


def resampled_report(n_splits=5):
    return {
        "evaluation_mode": "resampled",
        "n_splits": n_splits,
        "splits": [_split(i) for i in range(n_splits)],
        "stability": [{"feature_group": "a", "median_rank": 1.0}],
        "failures": [],
    }


def split_failure(index):
    return {"split": index, "model": None, "stage": "split",
            "error_type": "RuntimeError", "message": "boom"}


def test_insufficient_splits_without_n_splits_key():
    report = resampled_report(3)
    del report["n_splits"]
    report["shadow_test"] = {"n_trials": 3, "min_p_value": 0.125}

    findings = summarize(report)

    assert codes(findings) == ["insufficient_splits"]
    assert findings[0]["evidence"]["n_splits"] == 3


def test_clean_resampled_report_has_no_warnings():
    assert summarize(resampled_report()) == []


def test_resampled_summary_is_pure_and_strict_json():
    report = resampled_report()
    report["failures"] = [split_failure(1), {**_failure("m1", "fit"), "split": 0}]
    snapshot = copy.deepcopy(report)

    first, second = summarize(report), summarize(report)

    assert first == second and report == snapshot
    json.dumps(first, allow_nan=False)


def test_split_failures_list_the_split_and_error_type():
    report = resampled_report()
    del report["splits"][2]
    report["failures"] = [split_failure(2)]

    findings = summarize(report)

    assert codes(findings) == ["split_failures"]
    assert findings[0]["severity"] == "warning"
    assert findings[0]["evidence"] == {"splits": [{"split": 2, "error_type": "RuntimeError"}]}
    assert "2" in findings[0]["message"]


def test_resampled_failures_aggregate_models_over_splits_per_stage():
    report = resampled_report()
    report["failures"] = [
        {**_failure("m1", "fit"), "split": 0},
        {**_failure("m1", "fit"), "split": 3},
        {**_failure("m2", "fit"), "split": 3},
        {**_failure("m1", "initialization"), "split": 0},
        {**_failure("m1", "initialization"), "split": 1},
    ]

    findings = summarize(report)

    assert codes(findings) == ["failures", "failures"]
    assert [f["severity"] for f in findings] == ["info", "warning"]
    assert findings[0]["evidence"] == {"stage": "initialization", "models": {"m1": [0, 1]}}
    assert findings[1]["evidence"] == {"stage": "fit", "models": {"m1": [0, 3], "m2": [3]}}


def test_gate_off_and_empty_consensus_name_the_splits():
    report = resampled_report()
    report["splits"][1] = _split(1, baseline=None)
    report["splits"][3] = _split(3, consensus=[])

    findings = summarize(report)

    assert codes(findings) == ["gate_off", "empty_consensus"]
    assert [f["severity"] for f in findings] == ["warning", "warning"]
    assert findings[0]["evidence"] == {"splits": [1]}
    assert findings[1]["evidence"] == {"splits": [3]}


def test_resampled_below_shadow_reads_the_stability_table():
    report = resampled_report()
    report["stability"] = [
        {"feature_group": "a", "above_shadow_share": 0.9},
        {"feature_group": "b", "above_shadow_share": 0.2},
        {"feature_group": "c"},
    ]

    findings = summarize(report)

    assert codes(findings) == ["below_shadow"]
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"groups": {"b": 0.2}}


def test_insufficient_splits_when_shadow_test_cannot_reach_005():
    report = resampled_report(3)
    report["shadow_test"] = {"n_trials": 3, "min_p_value": 0.5 ** 3}

    findings = summarize(report)

    assert codes(findings) == ["insufficient_splits"]
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"n_splits": 3, "min_p_value": 0.125}
    report["shadow_test"] = {"n_trials": 5, "min_p_value": 0.5 ** 5}
    assert summarize(report) == []


def test_resampled_findings_come_in_spec_order():
    report = resampled_report(3)
    report["splits"][0] = _split(0, baseline=None)
    report["splits"][1] = _split(1, consensus=[])
    del report["splits"][2]
    report["failures"] = [split_failure(2), {**_failure("m1", "fit"), "split": 0}]
    report["stability"] = [{"feature_group": "b", "above_shadow_share": 0.0}]
    report["shadow_test"] = {"n_trials": 3, "min_p_value": 0.125}

    assert codes(summarize(report)) == [
        "split_failures", "failures", "gate_off", "empty_consensus",
        "below_shadow", "insufficient_splits",
    ]


def test_report_summary_works_on_resampled_reports():
    import numpy as np
    from sklearn.linear_model import Ridge

    from featranker import FeatureRanker

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    report = FeatureRanker("reg", models={"r": Ridge()}, verbose=False, shadows=True,
                           shadow_seed=0).rank_resampled(
        X, X[:, 0], feature_names=["a", "b"], scoring="r2", cv=3, n_repeats=2, random_state=0
    )

    assert report.summary() == summarize(report)
    assert "insufficient_splits" in codes(report.summary())


def test_below_shadow_lists_groups_under_half():
    report = clean_report()
    report["consensus"][0]["above_shadow_share"] = 1.0
    report["consensus"][1]["above_shadow_share"] = 0.25

    findings = [f for f in summarize(report) if f["code"] == "below_shadow"]

    assert len(findings) == 1
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"groups": {"b": 0.25}}


def test_below_shadow_comes_after_within_noise():
    report = clean_report()
    report["models"]["m1"]["importance"]["b"].update(mean=0.0, std=0.1)
    report["models"]["m2"]["importance"]["b"].update(mean=0.0, std=0.1)
    for row in report["consensus"]:
        row["above_shadow_share"] = 0.0

    assert codes(summarize(report))[-2:] == ["within_noise", "below_shadow"]


def test_no_below_shadow_without_shares():
    assert "below_shadow" not in codes(summarize(clean_report()))


def _with_train(report, **train_scores):
    """Give models a train_score; the model's evaluation_score stays 0.5."""
    for name, score in train_scores.items():
        report["models"][name]["train_score"] = score
    return report


def test_overfit_lists_models_that_lost_half_their_training_advantage():
    report = clean_report()
    report["baseline_score"] = 0.0
    report["models"]["m1"]["evaluation_score"] = 0.2
    report["models"]["m2"].update(train_score=0.56, evaluation_score=0.64)
    _with_train(report, m1=1.0)

    findings = [f for f in summarize(report) if f["code"] == "overfit"]

    assert len(findings) == 1 and findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {
        "models": {"m1": {"train_score": 1.0, "evaluation_score": 0.2, "share": 0.8}}
    }
    json.dumps(findings, allow_nan=False)


def test_overfit_skips_null_train_score_and_models_without_an_advantage():
    report = clean_report()
    report["baseline_score"] = 0.0
    report["models"]["m1"].update(train_score=None, evaluation_score=-1.0)
    report["models"]["m2"].update(train_score=-0.5, evaluation_score=-2.0)
    report["models"]["m3"].update(train_score=0.0, evaluation_score=-2.0)

    assert "overfit" not in codes(summarize(report))


def test_overfit_threshold_is_half():
    report = clean_report()
    report["baseline_score"] = 0.0
    report["models"]["m1"].update(train_score=1.0, evaluation_score=0.5)
    report["models"]["m2"].update(train_score=1.0, evaluation_score=0.51)

    finding = next(f for f in summarize(report) if f["code"] == "overfit")

    assert list(finding["evidence"]["models"]) == ["m1"]


def test_no_overfit_without_train_score_or_baseline():
    assert "overfit" not in codes(summarize(clean_report()))
    report = _with_train(clean_report(), m1=1.0, m2=1.0, m3=1.0)
    report["baseline_score"] = None
    assert "overfit" not in codes(summarize(report))


def _pairs_report(pairs):
    report = clean_report()
    report["evidence"] = {"correlation": {"pairs": pairs}}
    return report


def test_correlated_groups_lists_pairs():
    pairs = [{"groups": ["e", "f"], "rho": 0.99}]

    findings = [f for f in summarize(_pairs_report(pairs)) if f["code"] == "correlated_groups"]

    assert len(findings) == 1 and findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"pairs": pairs}
    assert "e" in findings[0]["message"] and "f" in findings[0]["message"]
    assert "split" in findings[0]["message"]


def test_no_correlated_groups_finding_without_pairs():
    assert "correlated_groups" not in codes(summarize(_pairs_report([])))
    assert "correlated_groups" not in codes(summarize(clean_report()))


def test_evidence_findings_come_after_below_shadow_in_order():
    report = _pairs_report([{"groups": ["a", "b"], "rho": 0.9}])
    report["baseline_score"] = 0.0
    report["models"]["m1"].update(train_score=1.0, evaluation_score=0.1)
    for row in report["consensus"]:
        row["above_shadow_share"] = 0.0

    assert codes(summarize(report))[-3:] == ["below_shadow", "overfit", "correlated_groups"]


def test_evidence_stage_failures_are_info():
    report = clean_report()
    report["failures"] = [_failure("m1", "evidence")]

    findings = summarize(report)

    assert codes(findings) == ["failures"]
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {"stage": "evidence", "models": ["m1"]}


def _ablation_report(medians):
    """A clean report plus an ablation block whose curve has these median scores."""
    report = clean_report()
    report["ablation"] = {
        "scoring": "r2",
        "models": ["m1", "m2", "m3"],
        "baseline_score": 0.1,
        "n_groups": 2,
        "curve": [
            {"k": k, "feature_groups": [], "n_features": k, "scores": {}, "median_score": m}
            for k, m in medians.items()
        ],
        "failures": [],
    }
    return report


def test_ablation_curve_finding_names_the_best_k_and_the_largest_k():
    findings = summarize(_ablation_report({1: 0.5, 2: 0.9, 3: 0.8}))

    assert codes(findings) == ["ablation_curve"]
    assert findings[0]["severity"] == "info"
    assert findings[0]["evidence"] == {
        "best_k": 2, "best_median_score": 0.9, "max_k": 3, "max_k_median_score": 0.8,
    }
    assert "k=2" in findings[0]["message"]
    assert "0.8 at the largest scored k, 3" in findings[0]["message"]
    json.dumps(findings, allow_nan=False)


def test_ablation_curve_finding_prefers_the_smallest_k_on_ties():
    findings = summarize(_ablation_report({1: 0.9, 2: 0.9, 3: 0.9}))

    assert findings[0]["evidence"]["best_k"] == 1


def test_ablation_curve_skips_null_medians():
    findings = summarize(_ablation_report({1: 0.5, 2: None, 3: 0.4}))

    assert findings[0]["evidence"]["best_k"] == 1
    assert findings[0]["evidence"]["max_k"] == 3
    assert "the largest scored k" in findings[0]["message"]


def test_no_ablation_curve_finding_without_a_usable_block():
    assert "ablation_curve" not in codes(summarize(clean_report()))
    assert "ablation_curve" not in codes(summarize(_ablation_report({1: None, 2: None})))
