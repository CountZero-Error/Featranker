"""
Mechanism evidence for a report: correlated feature groups and a SHAP summary.
"""

import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# Group pairs whose strongest member-column |Spearman rho| reaches this are listed.
CORRELATION_THRESHOLD = 0.5
# At most this many evaluation rows are explained, and as many training rows
# serve as the linear background.
SHAP_ROWS = 1000
# Longest error message kept in an "unsupported" reason.
REASON_CHARS = 200


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Column-by-column Spearman rho; NaN where a column is constant."""
    # Not scipy's spearmanr: with a constant column it returns one NaN for all.
    centered = rankdata(X, axis=0) - (X.shape[0] + 1) / 2
    norm = np.sqrt((centered**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(centered.T @ centered / np.outer(norm, norm), -1.0, 1.0)


def correlation_evidence(X_train: np.ndarray, groups: Dict[str, List[int]]) -> Dict[str, Any]:
    """Group pairs correlated on the training data (real columns only)."""
    rho = _spearman_matrix(X_train)
    pairs: List[Dict[str, Any]] = []
    for first, second in itertools.combinations(sorted(groups), 2):
        block = rho[np.ix_(groups[first], groups[second])]
        strength = np.where(np.isnan(block), -1.0, np.abs(block))
        index = np.unravel_index(np.argmax(strength), block.shape)
        if strength[index] >= CORRELATION_THRESHOLD:
            pairs.append({"groups": [first, second], "rho": float(block[index])})
    pairs.sort(key=lambda pair: (-abs(pair["rho"]), pair["groups"]))
    return {
        "method": "spearman",
        "data": "train",
        "threshold": CORRELATION_THRESHOLD,
        "pairs": pairs,
    }


def evenly_spaced_rows(n_rows: int, cap: int = SHAP_ROWS) -> np.ndarray:
    """At most cap row indices, evenly spaced, so no seed is involved."""
    return np.linspace(0, n_rows - 1, min(n_rows, cap)).astype(int)


# Per-column affine steps: a scaled column's SHAP value is the original column's.
COLUMN_SCALERS = (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)


def _unwrap(model: Any, *arrays: np.ndarray) -> Tuple[Any, List[np.ndarray]]:
    """The final estimator and the arrays passed through leading column scalers."""
    if not isinstance(model, Pipeline):
        return model, list(arrays)
    *earlier, (_, final) = model.steps
    for _, step in earlier:
        if not isinstance(step, COLUMN_SCALERS):
            raise ValueError(f"pipeline step {type(step).__name__} is not a per-column scaler")
    if not earlier:
        return final, list(arrays)
    return final, [model[:-1].transform(array) for array in arrays]


def _shap_values(
    model: Any, X: np.ndarray, background: np.ndarray, shap: Any
) -> Tuple[str, np.ndarray]:
    """(explainer name, SHAP values shaped (k, n_rows, n_columns))."""
    final, (X_final, background_final) = _unwrap(model, X, background)
    if hasattr(final, "coef_") and hasattr(final, "intercept_"):
        masker = shap.maskers.Independent(background_final, max_samples=len(background_final))
        name, values = "linear", shap.LinearExplainer(final, masker).shap_values(X_final)
    else:
        name, values = "tree", shap.TreeExplainer(final).shap_values(X_final)
    # (n, p), (n, p, k), or a list of k arrays, depending on model and version.
    stacked = np.stack(values) if isinstance(values, list) else np.asarray(values)
    if stacked.ndim == 2:
        stacked = stacked[np.newaxis]
    elif stacked.ndim == 3 and not isinstance(values, list):
        stacked = np.moveaxis(stacked, 2, 0)
    if stacked.ndim != 3 or not np.isfinite(stacked).all():
        raise ValueError("unexpected SHAP output")
    return name, stacked


def _group_summary(
    stacked: np.ndarray, X: np.ndarray, groups: Dict[str, List[int]]
) -> Dict[str, Dict[str, Optional[float]]]:
    # One class: regression or single output; two: the positive class;
    # more: mean_abs averaged over classes and no direction.
    classes = stacked[-1:] if len(stacked) <= 2 else stacked
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for group, columns in groups.items():
        per_row = classes[:, :, columns].sum(axis=2)  # sum over member columns
        direction = None
        if len(columns) == 1 and len(classes) == 1:
            pair = np.column_stack([X[:, columns[0]], per_row[0]])
            rho = _spearman_matrix(pair)[0, 1]
            direction = None if np.isnan(rho) else float(rho)
        summary[group] = {
            "mean_abs": float(np.abs(per_row).mean(axis=1).mean()),
            "direction": direction,
        }
    return summary


def shap_evidence(
    models: Dict[str, Any],
    X_eval: np.ndarray,
    X_train: np.ndarray,
    groups: Dict[str, List[int]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    SHAP summary per model on evenly spaced evaluation rows.

    Linear models use evenly spaced training rows as the background. Returns
    (the report's evidence.shap block, model name -> its shap entry). X_eval
    and X_train have every column the models were fit on; groups name real
    columns only.
    """
    try:
        import shap
    except Exception as error:  # a broken shap/numba install is not an ImportError
        problem = (
            "is not installed"
            if isinstance(error, ImportError)
            else f"failed to import ({type(error).__name__}: {str(error)[:REASON_CHARS]})"
        )
        warnings.warn(
            f"shap {problem}, so the SHAP evidence is skipped: "
            "pip install 'featranker[shap]'",
            UserWarning,
            # caller -> rank_features -> shap_evidence
            stacklevel=3,
        )
        return {"available": False}, {}
    rows = X_eval[evenly_spaced_rows(len(X_eval))]
    background = X_train[evenly_spaced_rows(len(X_train))]
    unsupported: List[Dict[str, str]] = []
    per_model: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        try:
            explainer, stacked = _shap_values(model, rows, background, shap)
            per_model[name] = {
                "explainer": explainer,
                "groups": _group_summary(stacked, rows, groups),
            }
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error)[:REASON_CHARS]}"
            unsupported.append({"model": name, "reason": reason})
    block = {"available": True, "n_rows": len(rows), "unsupported": unsupported}
    return block, per_model
