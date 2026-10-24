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

# A resampled report with a smallest shadow p-value above this cannot reach q <= 0.05.
SPLIT_Q_LEVEL = 0.05

REQUIRED_KEYS = ("baseline_score", "models", "consensus", "failures")
RESAMPLED_KEYS = ("splits", "stability", "failures")
FAILURE_STAGES = ("initialization", "fit", "ranking", "evidence", "shadows")


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
        models = list(dict.fromkeys(
            f["model"] for f in report["failures"] if f.get("stage") == stage
        ))
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


def _ablation_curve(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    block = report.get("ablation", {})
    # The paired median compares one set of models at every k.
    paired = bool(block.get("paired_models"))
    # Blocks from 1.16 have no paired_models key at all.
    predates = "paired_models" not in block
    key = "paired_median_score" if paired else "median_score"
    rows = [row for row in block.get("curve", []) if row.get(key) is not None]
    if not rows:
        return []
    # max keeps the first of equal scores, and the curve is in ascending k.
    best = max(rows, key=lambda row: row[key])
    last = rows[-1]
    at_last = (
        "the largest scored k" if best is last
        else f"{last[key]:.3g} at the largest scored k, {last['k']}"
    )
    if paired:
        basis = f"median over the {len(block['paired_models'])} model(s) scored at every k (paired)"
    elif predates:
        basis = "median score (this ablation block predates the paired curve)"
    else:
        basis = "unpaired median over the models scored at each k (no model scored at every k)"
    return [_finding(
        "ablation_curve", "info",
        f"Refit on the top-k groups, the {basis} was highest at k={best['k']} "
        f"({best[key]:.3g}), {at_last}. "
        "The curve is descriptive; it is scored on the evaluation data unless another "
        "set was passed to ablate().",
        {
            "best_k": best["k"], "best_median_score": best[key],
            "max_k": last["k"], "max_k_median_score": last[key],
        },
    )]


def _ablation_cohort_changed(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {
            "k": row["k"],
            "n_models_scored": sum(score is not None for score in row["scores"].values()),
            "models": [name for name, score in row["scores"].items() if score is not None],
        }
        for row in report.get("ablation", {}).get("curve", [])
        if "scores" in row
    ]
    if len({frozenset(row["models"]) for row in rows}) < 2:
        return []
    return [_finding(
        "ablation_cohort_changed", "warning",
        "The set of scored models changes across k in the ablation curve, so the "
        "unpaired median_score compares different model sets; read paired_median_score.",
        {"rows": rows},
    )]


RULES = (
    _gate_off, _empty_consensus, _failures, _few_families, _tied_groups,
    _within_noise, _below_shadow, _overfit, _correlated_groups, _ablation_curve,
    _ablation_cohort_changed,
)


def _split_failures(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    failed = [
        {"split": f["split"], "error_type": f.get("error_type")}
        for f in report["failures"]
        if f.get("stage") == "split"
    ]
    if not failed:
        return []
    return [_finding(
        "split_failures", "warning",
        f"{len(failed)} split(s) failed outright: "
        + ", ".join(f"{f['split']} ({f['error_type']})" for f in failed)
        + ". Their shares count them as misses.",
        {"splits": failed},
    )]


def _resampled_failures(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    for stage in FAILURE_STAGES:
        models: Dict[str, List[int]] = {}
        for f in report["failures"]:
            if f.get("stage") == stage:
                at = models.setdefault(f["model"], [])
                if f["split"] not in at:
                    at.append(f["split"])
        if not models:
            continue
        severity = "info" if stage in ("initialization", "evidence") else "warning"
        listed = "; ".join(f"{name} in split(s) {', '.join(map(str, at))}"
                           for name, at in models.items())
        findings.append(_finding(
            "failures", severity,
            f"{len(models)} model(s) failed at {stage}: {listed}.",
            {"stage": stage, "models": models},
        ))
    return findings


def _split_numbers(report: Dict[str, Any], is_listed: Any) -> List[int]:
    return [entry["split"] for entry in report["splits"] if is_listed(entry["report"])]


def _resampled_gate_off(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    splits = _split_numbers(report, lambda r: r["baseline_score"] is None)
    if not splits:
        return []
    return [_finding(
        "gate_off", "warning",
        f"The dummy baseline could not be scored in split(s) {', '.join(map(str, splits))}, "
        "so every ranked model voted there whether or not it beat the dummy.",
        {"splits": splits},
    )]


def _resampled_empty_consensus(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    splits = _split_numbers(report, lambda r: not r["consensus"])
    if not splits:
        return []
    return [_finding(
        "empty_consensus", "warning",
        f"No model beat the dummy baseline in split(s) {', '.join(map(str, splits))}, "
        "so those splits have no consensus ranking.",
        {"splits": splits},
    )]


def _resampled_below_shadow(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _below_shadow({"consensus": report["stability"]})


def _insufficient_splits(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    test = report.get("shadow_test")
    if test is None or test["min_p_value"] <= SPLIT_Q_LEVEL:
        return []
    n_splits = test["n_trials"]
    return [_finding(
        "insufficient_splits", "info",
        f"With {n_splits} splits the smallest possible shadow p-value is "
        f"{test['min_p_value']:.3f}, so no group can reach a q-value of {SPLIT_Q_LEVEL}; "
        "use more splits, e.g. RepeatedKFold.",
        {"n_splits": n_splits, "min_p_value": test["min_p_value"]},
    )]


RESAMPLED_RULES = (
    _split_failures, _resampled_failures, _resampled_gate_off,
    _resampled_empty_consensus, _resampled_below_shadow, _insufficient_splits,
)


def summarize(report: Any) -> List[Dict[str, Any]]:
    """
    Return structured findings for a report from FeatureRanker.rank_features()
    or rank_resampled().

    Each finding is a dict with code, severity ("info" or "warning"), message,
    and JSON-safe evidence. The report is not modified.
    """
    if not isinstance(report, dict):
        raise TypeError("[!] summarize() expects a report dict.")
    resampled = report.get("evaluation_mode") == "resampled"
    for key in RESAMPLED_KEYS if resampled else REQUIRED_KEYS:
        if key not in report:
            raise ValueError(f"[!] Report is missing required key '{key}'.")
    return [
        finding
        for rule in (RESAMPLED_RULES if resampled else RULES)
        for finding in rule(report)
    ]
