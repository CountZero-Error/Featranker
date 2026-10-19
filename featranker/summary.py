"""
Deterministic findings for a FeatRanker report (the reading guide as code).
"""

from typing import Any, Dict, List

# A consensus resting on fewer families than this is flagged.
MIN_FAMILIES = 3
# Groups whose family-weighted above-shadow share is below this are listed.
MIN_SHADOW_SHARE = 0.5
# Models that lost at least this share of their training advantage over the
# dummy on the evaluation data are listed as overfit.
MIN_OVERFIT_SHARE = 0.5

REQUIRED_KEYS = ("baseline_score", "models", "consensus", "failures")
FAILURE_STAGES = ("initialization", "fit", "ranking", "evidence")


def _finding(code: str, severity: str, message: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "evidence": evidence}


def _gate_off(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    if report["baseline_score"] is not None:
        return []
    dummy_failures = [f for f in report["failures"] if f.get("model") == "dummy_baseline"]
    return [_finding(
        "gate_off", "warning",
        "The dummy baseline could not be scored, so every ranked model voted "
        "whether or not it beat the dummy.",
        {"dummy_failures": dummy_failures},
    )]


def _empty_consensus(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    if report["consensus"]:
        return []
    return [_finding(
        "empty_consensus", "warning",
        "No model beat the dummy baseline, so there is no consensus ranking.",
        {"n_ranked_models": len(report["models"])},
    )]


def _failures(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for stage in FAILURE_STAGES:
        models = [f["model"] for f in report["failures"] if f.get("stage") == stage]
        if not models:
            continue
        severity = "info" if stage in ("initialization", "evidence") else "warning"
        findings.append(_finding(
            "failures", severity,
            f"{len(models)} model(s) failed at {stage}: {', '.join(models)}.",
            {"stage": stage, "models": models},
        ))
    return findings


def _few_families(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    consensus = report["consensus"]
    # Reports from before 0.5 have no n_families; skip rather than guess.
    if not consensus or "n_families" not in consensus[0]:
        return []
    n_families = consensus[0]["n_families"]
    if n_families >= MIN_FAMILIES:
        return []
    return [_finding(
        "few_families", "warning",
        f"The consensus rests on {n_families} model "
        f"famil{'y' if n_families == 1 else 'ies'}; treat it as weak evidence.",
        {"n_families": n_families, "n_models": consensus[0]["n_models"]},
    )]


def _tied_groups(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_rank: Dict[float, List[str]] = {}
    for row in report["consensus"]:
        by_rank.setdefault(row["median_rank"], []).append(row["feature_group"])
    return [
        _finding(
            "tied_groups", "info",
            f"{', '.join(groups)} share median rank {rank:g}; mean_rank breaks any "
            "remaining ordering tie. This finding does not assess ranking uncertainty.",
            {"median_rank": rank, "groups": groups},
        )
        for rank, groups in sorted(by_rank.items())
        if len(groups) > 1
    ]


def _within_noise(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    voters = [m for m in report["models"].values() if m.get("in_consensus")]
    noisy: Dict[str, Dict[str, int]] = {}
    for row in report["consensus"]:
        group = row["feature_group"]
        importances = [m["importance"][group] for m in voters if group in m["importance"]]
        count = sum(1 for imp in importances if imp["mean"] <= imp["std"])
        if importances and 2 * count > len(importances):
            noisy[group] = {"within_noise": count, "voting_models": len(importances)}
    if not noisy:
        return []
    return [_finding(
        "within_noise", "info",
        f"For {', '.join(noisy)}, mean importance is no greater than the shuffle "
        "standard deviation in most voting models. This flags large relative spread "
        "or non-positive importance; it does not establish absence of a score drop.",
        {"groups": noisy},
    )]


def _below_shadow(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    below = {
        row["feature_group"]: row["above_shadow_share"]
        for row in report["consensus"]
        if "above_shadow_share" in row and row["above_shadow_share"] < MIN_SHADOW_SHARE
    }
    if not below:
        return []
    return [_finding(
        "below_shadow", "info",
        f"{', '.join(below)} have a family-weighted above-shadow share below 0.5. "
        "Each family contributes its fraction of voting members beating that "
        "model's strongest shadow group. This is descriptive, not a significance test.",
        {"groups": below},
    )]


def _overfit(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    baseline = report["baseline_score"]
    if baseline is None:
        return []
    overfit = {}
    for name, model in report["models"].items():
        train = model.get("train_score")
        if train is None or train - baseline <= 0:
            continue
        share = (train - model["evaluation_score"]) / (train - baseline)
        if share >= MIN_OVERFIT_SHARE:
            overfit[name] = {
                "train_score": train,
                "evaluation_score": model["evaluation_score"],
                "share": share,
            }
    if not overfit:
        return []
    return [_finding(
        "overfit", "info",
        f"{', '.join(overfit)} lost at least half of the training advantage over the "
        "dummy baseline on the evaluation data, so may be overfit; read their "
        "importances with that in mind.",
        {"models": overfit},
    )]


def _correlated_groups(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = report.get("evidence", {}).get("correlation", {}).get("pairs")
    if not pairs:
        return []
    listed = "; ".join(f"{a} and {b} (rho {pair['rho']:.2f})"
                       for pair in pairs for a, b in [pair["groups"]])
    return [_finding(
        "correlated_groups", "info",
        f"Correlated on the training data: {listed}. Permutation importance may be "
        "split between correlated groups, so each can look less important than "
        "the pair together.",
        {"pairs": pairs},
    )]


RULES = (
    _gate_off, _empty_consensus, _failures, _few_families, _tied_groups,
    _within_noise, _below_shadow, _overfit, _correlated_groups,
)


def summarize(report: Any) -> List[Dict[str, Any]]:
    """
    Return structured findings for a report from FeatureRanker.rank_features().

    Each finding is a dict with code, severity ("info" or "warning"), message,
    and JSON-safe evidence. The report is not modified.
    """
    if not isinstance(report, dict):
        raise TypeError("[!] summarize() expects a report dict.")
    if report.get("evaluation_mode") == "resampled":
        raise ValueError(
            "[!] summarize() reads single-split reports; for a resampled report, "
            "summarize each report['splits'][i]['report'] instead."
        )
    for key in REQUIRED_KEYS:
        if key not in report:
            raise ValueError(f"[!] Report is missing required key '{key}'.")
    return [finding for rule in RULES for finding in rule(report)]
