"""
Ablation curve: refit the voting models on the top-k consensus groups.
"""

import copy
import numbers
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .importance import (
    FINGERPRINT_SCHEME,
    FeatureRanker,
    _fingerprint,
    _labels_fingerprint,
    _resolve_scorer,
    _validate_X,
    _validate_weights,
    _validate_y,
    _weights_fingerprint,
)
from .report import Report

FAILURE_STAGES = ("fit", "ranking")
# The decades of the default ks: 10, 20, 50, 100, 200, 500, ...
DECADE_MARKS = (1, 2, 5)
LOW_MARKS = (1, 2, 3, 5)
KEPT_GROUP = "kept"


def _default_ks(n_groups: int) -> List[int]:
    """1, 2, 3, 5, 10, 20, 50, 100, ... up to n_groups, plus n_groups."""
    marks = set(LOW_MARKS)
    decade = 10
    while decade <= n_groups:
        marks.update(mark * decade for mark in DECADE_MARKS)
        decade *= 10
    return sorted({k for k in marks if k <= n_groups} | {n_groups})


def _validate_ks(ks: Any, n_groups: int) -> List[int]:
    """The sorted, de-duplicated ks; the default ks for None."""
    if ks is None:
        return _default_ks(n_groups)
    valid = (
        isinstance(ks, (list, tuple, np.ndarray))
        and len(ks) > 0
        and all(
            isinstance(k, numbers.Integral) and not isinstance(k, bool) and 1 <= int(k) <= n_groups
            for k in ks
        )
    )
    if not valid:
        raise ValueError(
            f"[!] ks must be a non-empty list of ints from 1 to {n_groups} "
            "(the number of consensus groups)."
        )
    return sorted({int(k) for k in ks})


def _voters(ranker: FeatureRanker, report: Dict[str, Any]) -> List[str]:
    """Names of the voting models; raises when the report cannot be ablated."""
    if not ranker.is_fitted_:
        raise ValueError("[!] Call fit() before ablate().")
    if report.get("evaluation_mode") == "resampled":
        raise ValueError("[!] ablate() needs a held-out report, not a resampled one.")
    voters = [name for name, model in report["models"].items() if model["in_consensus"]]
    if not report["consensus"] or not voters:
        raise ValueError("[!] The report has no consensus ranking, so there is nothing to ablate.")
    known = (
        set(ranker.user_models)
        if ranker.user_models is not None
        else {name for group in ranker.models[ranker.task].values() for name in group}
    )
    unknown = [name for name in voters if name not in known]
    if unknown:
        raise ValueError(
            f"[!] The report's model '{unknown[0]}' is not one of the ranker's models."
        )
    return voters


def _weights_match(weights: Any, n_rows: int, recorded: Optional[str]) -> bool:
    """Whether the weights fingerprint to the report's record (None: no weights)."""
    return _weights_fingerprint(_validate_weights(weights, n_rows)) == recorded


def _check_weights(weights: Any, n_rows: int, recorded: Optional[str], name: str) -> None:
    """Raise when the weights do not fingerprint to the report's record."""
    if not _weights_match(weights, n_rows, recorded):
        raise ValueError(
            f"[!] {name} is not what this report was ranked with "
            "(pass the same weights, or none if the ranking used none)."
        )


def _prepare(
    ranker: FeatureRanker,
    data: tuple[Any, Any, Any, Any],
    report: Dict[str, Any],
    weights: tuple[Any, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Validated training and evaluation arrays, with the inputs checked against the report.

    The last value is whether the evaluation data is the report's ranking holdout.
    """
    X, y, X_eval, y_eval = data
    names = ranker.feature_names_
    categorical = [i for i, name in enumerate(names) if name in ranker.categorical_features_]
    options: Dict[str, Any] = {
        "expected_names": names,
        "allow_nan": ranker.missing == "impute",
        "categorical": categorical,
    }
    X_values, _ = _validate_X(X, **options)
    y_values = _validate_y(y, len(X_values))
    run = report.get("run", {})
    # Reports from before scheme 2 carry no key and were fingerprinted with scheme 1.
    scheme = run.get("fingerprint_scheme", 1)
    expected = run.get("train_data_sha256")
    if expected is not None:
        if _fingerprint(names, X_values, y_values, scheme) != expected:
            raise ValueError("[!] X and y are not the training data this report was ranked from.")
        # The run block leaves the weights key out when there were none.
        _check_weights(weights[0], len(X_values), run.get("train_weights_sha256"), "sample_weight")
    # No version before 1.21 could subsample training rows, so a missing hash means every row;
    # the ranker must have fit on the same rows (None being every row) in both directions.
    expected_rows = run.get("train_rows_sha256")
    actual_rows = None if ranker.train_rows_ is None else _labels_fingerprint(ranker.train_rows_)
    if actual_rows != expected_rows:
        raise ValueError(
            "[!] This ranker was not fit on the training rows this report was ranked from "
            "(fit it with the same max_samples and random_state)."
        )
    eval_values, _ = _validate_X(X_eval, **options)
    y_eval_values = _validate_y(y_eval, len(eval_values))
    same_data = _fingerprint(names, eval_values, y_eval_values, scheme) == run.get("data_sha256")
    if same_data:  # the report's rows with other weights are refused, not recorded
        _check_weights(
            weights[1], len(eval_values), run.get("weights_sha256"), "eval_sample_weight"
        )
    same_holdout = same_data and _weights_match(
        weights[1], len(eval_values), run.get("weights_sha256")
    )
    return X_values, y_values, eval_values, y_eval_values, same_holdout


def _kept_columns(
    ranker: FeatureRanker, report: Dict[str, Any], k: int
) -> tuple[List[str], List[int]]:
    """The top-k consensus groups, and the positions of their columns in training order."""
    top = [row["feature_group"] for row in report["consensus"][:k]]
    kept = {column for group in top for column in report["feature_groups"][group]}
    unknown = sorted(kept - set(ranker.feature_names_))
    if unknown:
        raise ValueError(f"[!] The report's column '{unknown[0]}' is not a column of the ranker.")
    return top, [i for i, name in enumerate(ranker.feature_names_) if name in kept]


def _sub_ranker(ranker: FeatureRanker, voters: List[str], kept: List[str]) -> FeatureRanker:
    """An unfitted ranker with the same preprocessing and only the voting models."""
    options: Dict[str, Any] = {
        "task": ranker.task,
        "n_jobs": ranker.n_jobs,
        "missing": ranker.missing,
        "model_input": ranker.model_input,
        "categorical": [n for n in ranker.categorical_features_ if n in kept] or None,
        "max_categories": ranker.max_categories,
        "verbose": False,
    }
    if ranker.user_models is not None:
        return FeatureRanker(models={n: ranker.user_models[n] for n in voters}, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the original run already warned about init failures
        sub = FeatureRanker(group=ranker.group, **options)
    sub.models = {
        task: {
            group: {name: model for name, model in models.items() if name in voters}
            for group, models in groups.items()
        }
        for task, groups in ranker.models.items()
    }
    return sub


def _failure_rows(
    failures: Sequence[Dict[str, Any]], voters: List[str], k: int
) -> List[Dict[str, Any]]:
    return [
        {"model": f["model"], "stage": f["stage"], "message": f["message"], "k": k}
        for f in failures
        if f.get("model") in voters and f.get("stage") in FAILURE_STAGES
    ]


def _all_failed(
    sub: FeatureRanker, voters: List[str], message: str, k: int
) -> List[Dict[str, Any]]:
    """Fit failures as recorded, and a ranking failure for each voter that did fit."""
    rows = _failure_rows(sub.fit_failures_, voters, k)
    failed = {row["model"] for row in rows}
    return rows + [
        {"model": name, "stage": "ranking", "message": message, "k": k}
        for name in voters
        if name not in failed
    ]


def _refit_scores(
    sub: FeatureRanker,
    voters: List[str],
    k: int,
    kept: List[str],
    data: tuple[np.ndarray, np.ndarray, np.ndarray, Any],
    settings: Dict[str, Any],
) -> tuple[Dict[str, Optional[float]], List[Dict[str, Any]], Optional[float]]:
    """
    Fit the sub-ranker and score each voter on every evaluation row; None when it failed.

    The last value is the dummy's score from the same fit, weights, rows and scorer.
    """
    X_train, y_train, X_eval, y_eval = data
    nothing: Dict[str, Optional[float]] = {name: None for name in voters}
    try:
        sub.fit(
            X_train, y_train, feature_names=kept, sample_weight=settings["sample_weight"]
        )
    except RuntimeError:  # no model fit
        return nothing, _failure_rows(sub.fit_failures_, voters, k), None
    try:
        with warnings.catch_warnings():
            # A small k can lose to the dummy; that is the curve, not a warning.
            warnings.filterwarnings("ignore", message="No model beat the dummy")
            ranked = sub.rank_features(
                X_eval,
                y_eval,
                scoring=settings["scoring"],
                feature_groups={KEPT_GROUP: kept},
                n_repeats=1,
                random_state=settings["random_state"],
                sample_weight=settings["eval_sample_weight"],
            )
    except RuntimeError as error:  # no model ranked
        return nothing, _all_failed(sub, voters, str(error), k), None
    scores: Dict[str, Optional[float]] = {
        name: ranked["models"][name]["evaluation_score"] if name in ranked["models"] else None
        for name in voters
    }
    return scores, _failure_rows(ranked["failures"], voters, k), ranked["baseline_score"]


def _median(scores: Dict[str, Optional[float]]) -> Optional[float]:
    present = [score for score in scores.values() if score is not None]
    return float(np.median(present)) if present else None


def _paired_models(voters: List[str], curve: List[Dict[str, Any]]) -> List[str]:
    """The voters with a score at every k, in report order."""
    return [name for name in voters if all(row["scores"][name] is not None for row in curve)]


def ablate(
    ranker: FeatureRanker,
    X: Any,
    y: Any,
    X_eval: Any,
    y_eval: Any,
    report: Dict[str, Any],
    *,
    ks: Optional[Sequence[int]] = None,
    scoring: Any = None,
    sample_weight: Any = None,
    eval_sample_weight: Any = None,
) -> Report:
    """A copy of report with an "ablation" block; see FeatureRanker.ablate()."""
    voters = _voters(ranker, report)
    n_groups = len(report["consensus"])
    ks = _validate_ks(ks, n_groups)
    scoring = report["scoring"] if scoring is None else scoring
    scoring_name = _resolve_scorer(scoring)[1]
    X_values, y_values, eval_values, y_eval_values, same_holdout = _prepare(
        ranker, (X, y, X_eval, y_eval), report, (sample_weight, eval_sample_weight)
    )
    train_weight = sample_weight
    if ranker.train_rows_ is not None:
        # The ranker fit on a training subsample (checked above against all rows):
        # refit on the same rows, so the curve compares like with like.
        rows = ranker.train_rows_
        train_weight = _validate_weights(sample_weight, len(X_values))
        train_weight = None if train_weight is None else train_weight[rows]
        X_values, y_values = X_values[rows], y_values[rows]
    settings = {
        "scoring": scoring,
        "random_state": report.get("random_state"),
        "sample_weight": train_weight,
        "eval_sample_weight": eval_sample_weight,
    }
    curve: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    baseline_score: Optional[float] = None
    for k in ks:
        top, positions = _kept_columns(ranker, report, k)
        kept = [ranker.feature_names_[i] for i in positions]
        data = (X_values[:, positions], y_values, eval_values[:, positions], y_eval_values)
        scores, k_failures, k_baseline = _refit_scores(
            _sub_ranker(ranker, voters, kept), voters, k, kept, data, settings
        )
        curve.append({
            "k": k,
            "feature_groups": top,
            "n_features": len(positions),
            "scores": scores,
            "median_score": _median(scores),
            "n_models_scored": sum(score is not None for score in scores.values()),
        })
        failures.extend(k_failures)
        if baseline_score is None:  # the dummy ignores the features, so any k will do
            baseline_score = k_baseline
    paired = _paired_models(voters, curve)
    for row in curve:
        row["paired_median_score"] = _median({name: row["scores"][name] for name in paired})
    result = copy.deepcopy(report)
    result["ablation"] = {
        "scoring": scoring_name,
        "models": voters,
        "baseline_score": baseline_score,
        "n_groups": n_groups,
        "fingerprint_scheme": FINGERPRINT_SCHEME,
        "data_sha256": _fingerprint(ranker.feature_names_, eval_values, y_eval_values),
        "weights_sha256": _weights_fingerprint(
            _validate_weights(eval_sample_weight, len(eval_values))
        ),
        "n_eval_rows": len(eval_values),
        "same_holdout": same_holdout,
        "paired_models": paired,
        "curve": curve,
        "failures": failures,
    }
    return Report(result)
