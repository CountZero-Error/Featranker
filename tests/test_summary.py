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
