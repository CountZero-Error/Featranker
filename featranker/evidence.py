"""
Mechanism evidence for a report: correlated feature groups and a SHAP summary.
"""

import itertools
import numbers
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

# Group pairs whose strongest member-column |Spearman rho| reaches this are listed.
CORRELATION_THRESHOLD = 0.5
# Floating-point slack on the auto_groups cut.
CUT_TOLERANCE = 1e-12
# At most this many evaluation rows are explained, and as many training rows
# serve as the linear background.
SHAP_ROWS = 1000
# Cells (rows x columns) ranked and squared at a time by _spearman_matrix.
SPEARMAN_BLOCK_CELLS = 1 << 18
# Longest error message kept in an "unsupported" reason.
REASON_CHARS = 200
# A column pair needs at least this many rows present in both to be correlated;
# with fewer, independent columns look strongly correlated by chance.
MIN_PAIR_ROWS = 30


def _column_blocks(width: int, n_rows: int) -> List[Tuple[int, int]]:
    """Column ranges holding at most SPEARMAN_BLOCK_CELLS cells (one column at least)."""
    step = max(1, SPEARMAN_BLOCK_CELLS // max(1, n_rows))
    return [(start, min(start + step, width)) for start in range(0, width, step)]


def _spearman_matrix(X: np.ndarray) -> np.ndarray:
    """
    Column-by-column Spearman rho; NaN where a column is constant.

    With missing values, a pair involving a column with NaN uses the rows
    where both are present, and is NaN with fewer than MIN_PAIR_ROWS such
    rows; pairs of complete columns are computed as without NaN.
    """
    if np.isnan(X).any():
        return _pairwise_spearman_matrix(X)
    # Not scipy's spearmanr: with a constant column it returns one NaN for all.
    # Ranked and squared a block of columns at a time, so beyond the ranks
    # themselves only a block is extra. rankdata(axis=0) returns Fortran order,
    # and so does the preallocated array: the column sums then run in the same
    # order as on a whole matrix, which keeps the values identical to ranking at once.
    blocks = _column_blocks(X.shape[1], X.shape[0]) or [(0, 0)]
    first = rankdata(X[:, blocks[0][0] : blocks[0][1]], axis=0)
    # The buffers take the dtype rankdata returns (SciPy keeps float32 ranks only
    # from some versions on), so the values equal ranking the whole matrix at once.
    centered = first if len(blocks) == 1 else np.empty(X.shape, dtype=first.dtype, order="F")
    if len(blocks) > 1:
        centered[:, : blocks[0][1]] = first
        for start, stop in blocks[1:]:
            centered[:, start:stop] = rankdata(X[:, start:stop], axis=0)
    centered -= (X.shape[0] + 1) / 2
    norm = np.empty(X.shape[1], dtype=centered.dtype)
    for start, stop in blocks:
        norm[start:stop] = (centered[:, start:stop] ** 2).sum(axis=0)
    norm = np.sqrt(norm)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(centered.T @ centered / np.outer(norm, norm), -1.0, 1.0)


def _pairwise_spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Complete columns together; other pairs re-rank on their shared rows."""
    present = ~np.isnan(X)
    complete = present.all(axis=0)
    rho = np.full((X.shape[1], X.shape[1]), np.nan)
    if complete.any():
        rho[np.ix_(complete, complete)] = _spearman_matrix(X[:, complete])
    for first in range(X.shape[1]):
        for second in range(first, X.shape[1]):
            if complete[first] and complete[second]:
                continue
            rows = present[:, first] & present[:, second]
            if rows.sum() >= MIN_PAIR_ROWS:
                pair = X[rows][:, [first, second]]
                rho[first, second] = rho[second, first] = _spearman_matrix(pair)[0, 1]
    return rho


def auto_groups_record(threshold: float) -> Dict[str, Any]:
    """The run["auto_groups"] entry for groups proposed by auto_groups()."""
    return {"threshold": float(threshold), "method": "spearman-complete-linkage"}


def auto_groups(
    X: Any,
    *,
    threshold: float = 0.7,
    feature_names: Optional[List[str]] = None,
    categorical: Any = None,
) -> Dict[str, List[str]]:
    """
    Propose feature_groups from X alone: columns clustered by Spearman correlation.

    Complete linkage on 1 - |rho|, cut so that every pair inside a group has
    |rho| >= threshold. A constant column is uncorrelated with every other.
    Only groups of two or more columns are returned; a group is named by its
    members joined with "+". Uses no target. NaN is allowed: each pair is
    correlated on the rows where both values are present. Categorical columns
    (categorical as in FeatureRanker: "auto" or a list of names) are never
    grouped.
    """
    # Local import: importance imports this module.
    from .importance import _categorical_indices, _validate_X

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, numbers.Real)
        or not 0 < threshold <= 1
    ):
        raise ValueError("[!] threshold must be a number with 0 < threshold <= 1.")
    if feature_names is None and not hasattr(X, "columns"):
        # Same default names as ConsensusSelector.
        width = np.shape(X)[1] if np.ndim(X) == 2 else 0
        feature_names = [f"x{index}" for index in range(width)]
    skipped = _categorical_indices(X, categorical, feature_names)
    values, names = _validate_X(
        X, feature_names=feature_names, allow_nan=True, categorical=skipped
    )
    if values.shape[0] < 2:
        raise ValueError("[!] X must contain at least two rows.")
    members = names
    if skipped:
        kept = [index for index in range(len(names)) if index not in skipped]
        values, members = values[:, kept].astype(float, copy=False), [names[index] for index in kept]
    if values.shape[1] < 2:
        return {}

    # NaN (constant column) -> 2, beyond any cut, so it is never grouped.
    distance = np.nan_to_num(np.clip(1.0 - np.abs(_spearman_matrix(values)), 0.0, 1.0), nan=2.0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="complete"),
        # Exact copies can compute |rho| = 1 - 2e-16, so the cut allows a tiny slack.
        t=1.0 - threshold + CUT_TOLERANCE,
        criterion="distance",
    )
    clusters: Dict[int, List[str]] = {}
    for label, name in zip(labels, members):  # insertion order = first member's position
        clusters.setdefault(int(label), []).append(name)
    groups: Dict[str, List[str]] = {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        group_name = "+".join(members)
        if group_name in names or group_name in groups:
            raise ValueError(
                f"[!] Group name '{group_name}' collides with a column or another group."
            )
        groups[group_name] = members
    return groups


def correlation_evidence(
    X_train: np.ndarray, groups: Dict[str, List[int]], categorical: Sequence[int] = ()
) -> Dict[str, Any]:
    """Group pairs correlated on the training data (real columns only, codes left out)."""
    rho = _spearman_matrix(X_train)
    rho[list(categorical), :] = rho[:, list(categorical)] = np.nan
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


# Per-column steps: a transformed column's SHAP value is the original column's.
COLUMN_SCALERS = (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)


def _is_column_step(step: Any) -> bool:
    # An imputer with add_indicator=True would append columns.
    if isinstance(step, SimpleImputer):
        return not step.add_indicator
    return isinstance(step, COLUMN_SCALERS)


def _unwrap(model: Any, *arrays: np.ndarray) -> Tuple[Any, List[np.ndarray]]:
    """The final estimator and the arrays passed through leading column scalers."""
    if not isinstance(model, Pipeline):
        return model, list(arrays)
    *earlier, (_, final) = model.steps
    for _, step in earlier:
        if isinstance(step, ColumnTransformer):
            raise ValueError(
                "pipeline contains a ColumnTransformer; SHAP needs the model's input "
                "columns to match the features"
            )
        # "passthrough" and None are scikit-learn's identity step.
        identity = step is None or (isinstance(step, str) and step == "passthrough")
        if not (identity or _is_column_step(step)):
            raise ValueError(f"pipeline step {type(step).__name__} is not a per-column scaler")
    if not earlier:
        return final, list(arrays)
    transformed = [model[:-1].transform(array) for array in arrays]
    if any(new.shape[1] != old.shape[1] for new, old in zip(transformed, arrays)):
        # e.g. SimpleImputer drops an all-missing column unless keep_empty_features.
        raise ValueError("pipeline steps change the number of columns")
    return final, transformed


def _shap_values(
    model: Any, X: Any, background: Any, shap: Any
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
    stacked: np.ndarray,
    X: np.ndarray,
    groups: Dict[str, List[int]],
    categorical: Sequence[int] = (),
) -> Dict[str, Dict[str, Optional[float]]]:
    # One class: regression or single output; two: the positive class;
    # more: mean_abs averaged over classes and no direction.
    classes = stacked[-1:] if len(stacked) <= 2 else stacked
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for group, columns in groups.items():
        per_row = classes[:, :, columns].sum(axis=2)  # sum over member columns
        direction = None
        # Codes are labels, so a correlation with the SHAP values means nothing.
        if len(columns) == 1 and len(classes) == 1 and columns[0] not in categorical:
            pair = np.column_stack([X[:, columns[0]], per_row[0]])
            rho = _spearman_matrix(pair)[0, 1]
            direction = None if np.isnan(rho) else float(rho)
        summary[group] = {
            "mean_abs": float(np.abs(per_row).mean(axis=1).mean()),
            "direction": direction,
        }
    return summary


_MaybeArray = TypeVar("_MaybeArray", bound=Optional[np.ndarray])


def _read_only(values: _MaybeArray) -> _MaybeArray:
    """
    A read-only array over the same memory (no copy); None stays None.

    A plain view with write=False is not enough: scikit-learn's check_array
    sets the flag back on a view of a writable array. A view of a read-only
    buffer cannot be flipped back. Dtypes without a buffer format are copied.
    """
    if values is None:
        return values
    try:
        buffer: Any = memoryview(cast(Any, values)).toreadonly()
        return cast(_MaybeArray, np.asarray(buffer))
    except (TypeError, ValueError, NotImplementedError, BufferError):
        frozen = cast(np.ndarray, values).copy()
        frozen.setflags(write=False)
        return cast(_MaybeArray, frozen)


def _model_X(X: np.ndarray, names: Optional[List[str]]) -> Any:
    """What models see: X itself, or a DataFrame named `names` (no copy); read-only either way."""
    X = _read_only(X)
    if names is None:
        return X
    import pandas as pd

    return pd.DataFrame(X, columns=names, copy=False)


def shap_evidence(
    models: Dict[str, Any],
    X_eval: np.ndarray,
    X_train: np.ndarray,
    groups: Dict[str, List[int]],
    categorical: Sequence[int] = (),
    names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    SHAP summary per model on evenly spaced evaluation rows.

    Linear models use evenly spaced training rows as the background. Returns
    (the report's evidence.shap block, model name -> its shap entry). X_eval
    and X_train have every column the models were fit on; groups name real
    columns only; categorical lists the code columns, whose direction is None;
    names, if given, are the column names models get as a DataFrame.
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
            explainer, stacked = _shap_values(
                model, _model_X(rows, names), _model_X(background, names), shap
            )
            per_model[name] = {
                "explainer": explainer,
                "groups": _group_summary(stacked, rows, groups, categorical),
            }
        except Exception as error:
            reason = f"{type(error).__name__}: {str(error)[:REASON_CHARS]}"
            unsupported.append({"model": name, "reason": reason})
    block = {"available": True, "n_rows": len(rows), "unsupported": unsupported}
    return block, per_model
