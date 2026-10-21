"""
Feature importance ranking via permutation importance across multiple ML models.
"""

import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import numpy as np
import sklearn
import yaml
from joblib import Parallel, delayed
from scipy.stats import binom
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.fixes import parse_version
from tqdm import tqdm

from .evidence import _model_X, correlation_evidence, shap_evidence
from .report import Report

try:
    import pandas as _pd
    PANDAS_MISSING: tuple = (type(_pd.NA), type(_pd.NaT))
except ImportError:  # pandas is optional
    _pd = None
    PANDAS_MISSING = ()

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}

OPTIONAL_MODEL_MODULES = {"xgboost", "lightgbm", "catboost"}

BASELINE_NAME = "dummy_baseline"

SHADOW_TEST_NOTE = (
    "Binomial reference values, not calibrated significance: the 0.5 null and "
    "false-discovery-rate control are not established for arbitrary estimators "
    "or overlapping splits."
)

CONFIG_PATH = Path(__file__).with_name("importance_config.yaml")

# scikit-learn < 1.8 ignores LogisticRegression(l1_ratio=...) unless `penalty`
# is set, so L1/elastic-net configs would silently run as L2.
SKLEARN_NEEDS_PENALTY = parse_version(sklearn.__version__) < parse_version("1.8")


def _model_params(class_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    if (
        not SKLEARN_NEEDS_PENALTY
        or class_name != "LogisticRegression"
        or "l1_ratio" not in params
        or "penalty" in params
    ):
        return params
    ratio = params.pop("l1_ratio")
    params["penalty"] = {0: "l2", 1: "l1"}.get(ratio, "elasticnet")
    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = ratio
    return params


def _registry_families() -> Dict[str, str]:
    """Built-in model name -> family, read from the shipped registry."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return {
        entry["name"]: str(entry.get("family") or entry["name"])
        for groups in config.values()
        for entries in groups.values()
        for entry in entries
    }


def _validate_models(models: Any) -> Dict[str, Any]:
    if not isinstance(models, dict):
        raise TypeError("[!] models must be a dict of name -> estimator.")
    if not models:
        raise ValueError("[!] models must be a non-empty dict.")
    for name, estimator in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("[!] Model names must be non-empty strings.")
        if name == BASELINE_NAME:
            raise ValueError(f"[!] Model name '{BASELINE_NAME}' is reserved.")
        if not callable(getattr(estimator, "fit", None)):
            raise TypeError(f"[!] Model '{name}' must have a callable fit().")
    return dict(models)


def _validate_categorical(categorical: Any) -> Any:
    """None, "auto", or a list of feature names (a tuple is copied to a list)."""
    if categorical is None or (isinstance(categorical, str) and categorical == "auto"):
        return categorical
    if isinstance(categorical, (list, tuple)) and all(
        isinstance(name, str) and name for name in categorical
    ):
        return list(categorical)
    raise ValueError("[!] categorical must be None, 'auto', or a list of feature names.")


def _categorical_indices(
    X: Any, categorical: Any, feature_names: Optional[List[str]] = None
) -> List[int]:
    """Column positions of the categorical features of X, in column order."""
    categorical = _validate_categorical(categorical)
    if categorical is None:
        return []
    if categorical == "auto":
        # Object, string and category dtypes all report kind "O".
        return [
            index
            for index, dtype in enumerate(getattr(X, "dtypes", []))
            if getattr(dtype, "kind", "") == "O"
        ]
    names = [str(name) for name in X.columns] if hasattr(X, "columns") else feature_names
    if names is None:
        raise ValueError("[!] feature_names are required for NumPy input.")
    positions = {name: index for index, name in enumerate(names)}
    unknown = [name for name in categorical if name not in positions]
    if unknown:
        raise ValueError(f"[!] categorical contains unknown feature '{unknown[0]}'.")
    return sorted(positions[name] for name in set(categorical))


DATE_TYPES = (datetime.date, datetime.timedelta, np.datetime64, np.timedelta64)
NOT_NUMERIC = (
    "[!] X must be numeric; convert datetime and timedelta columns to numbers first."
)


def _has_dates(column: Any) -> bool:
    kinds = set(map(type, column)) - set(PANDAS_MISSING)  # NaT is missing, not a date
    return any(issubclass(kind, DATE_TYPES) for kind in kinds)


def _validate_X(
    X: Any,
    feature_names: Optional[List[str]] = None,
    expected_names: Optional[List[str]] = None,
    allow_nan: bool = False,
    categorical: Optional[List[int]] = None,
) -> tuple[np.ndarray, List[str]]:
    """
    Validated copy of X and its feature names.

    With categorical column positions, the result is an object array: those
    columns keep their raw values and the others are checked as numbers.
    """
    categorical = categorical or []
    target_dtype = object if categorical else float
    dataframe_names = list(X.columns) if hasattr(X, "columns") else None
    # Always a copy: models and kept evidence data must not alias the caller's array.
    if dataframe_names is not None and hasattr(X, "to_numpy"):
        # to_numpy would turn dates into nanosecond counts, which are not features.
        if any(getattr(dtype, "kind", "") in {"M", "m"} for dtype in X.dtypes):
            raise ValueError(NOT_NUMERIC)
        # Object and category columns of dates would become nanoseconds too.
        if any(
            getattr(dtype, "kind", "") == "O" and _has_dates(X.iloc[:, position])
            for position, dtype in enumerate(X.dtypes)
        ):
            raise ValueError(NOT_NUMERIC)
        # to_numpy turns pandas NA (nullable dtypes) into NaN; np.array would raise.
        values = X.to_numpy(dtype=target_dtype, na_value=np.nan, copy=True)
    else:
        if isinstance(X, np.ndarray) and X.dtype.kind in {"M", "m"}:
            raise ValueError(NOT_NUMERIC)
        values = np.array(X, dtype=target_dtype)
    if values.ndim != 2:
        raise ValueError("[!] X must be a two-dimensional feature matrix.")
    numeric = values
    if categorical:
        if any(_has_dates(values[:, column]) for column in categorical):
            raise ValueError(NOT_NUMERIC)
        numeric = values[:, np.setdiff1d(np.arange(values.shape[1]), categorical)].astype(float)
    if allow_nan:
        if np.isinf(numeric).any():
            raise ValueError("[!] X must not contain infinite values.")
    elif not np.isfinite(numeric).all():
        raise ValueError(
            "[!] X must contain only finite values; impute or drop missing "
            "values before ranking."
        )
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("[!] X must contain at least one row and one feature.")
    if expected_names is not None and values.shape[1] != len(expected_names):
        raise ValueError(
            f"[!] Expected {len(expected_names)} feature names/columns, "
            f"got {values.shape[1]}."
        )

    if dataframe_names is not None:
        names = [str(name) for name in dataframe_names]
        if feature_names is not None and list(feature_names) != names:
            raise ValueError("[!] feature_names conflict with DataFrame columns.")
    elif feature_names is None:
        if expected_names is None:
            raise ValueError("[!] feature_names are required for NumPy input.")
        names = list(expected_names)
    else:
        names = list(feature_names)

    if len(names) != values.shape[1]:
        raise ValueError(
            f"[!] Expected {values.shape[1]} feature names, got {len(names)}."
        )
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("[!] Feature names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("[!] Feature names must be unique.")
    if expected_names is not None and names != list(expected_names):
        raise ValueError("[!] Evaluation feature names and order must match fit().")
    return values, names


def _is_missing(column: np.ndarray) -> np.ndarray:
    """Which values of an object column are missing (None, NaN, pd.NA or pd.NaT)."""
    return np.array(
        [value is None
         or isinstance(value, PANDAS_MISSING)
         or (isinstance(value, (float, np.floating)) and value != value)
         for value in column],
        dtype=bool,
    )


def _level_key(value: Any) -> str:
    """String form of a level; integral numbers drop the float suffix (1.0 -> "1")."""
    integral = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    return str(int(value)) if integral and float(value).is_integer() else str(value)


def _learn_levels(
    values: np.ndarray, indices: List[int], names: List[str]
) -> Dict[str, List[Optional[str]]]:
    """Per categorical column: sorted string forms, then None if any value is missing."""
    levels: Dict[str, List[Optional[str]]] = {}
    for index in indices:
        missing = _is_missing(values[:, index])
        found = sorted({_level_key(value) for value in values[~missing, index]})
        levels[names[index]] = [*found, None] if missing.any() else [*found]
    return levels


def _encode_categorical(
    values: np.ndarray,
    indices: List[int],
    names: List[str],
    levels: Dict[str, List[Optional[str]]],
) -> tuple[np.ndarray, Dict[str, int]]:
    """
    Float matrix with each categorical column replaced by level codes.

    A value not among the learned levels gets the code len(levels); the second
    result counts those values per column.
    """
    coded = np.empty(values.shape, dtype=float)
    numeric = np.setdiff1d(np.arange(values.shape[1]), indices)
    coded[:, numeric] = values[:, numeric].astype(float)
    unseen: Dict[str, int] = {}
    for index in indices:
        known = levels[names[index]]
        codes = {level: code for code, level in enumerate(known)}
        column = values[:, index]
        coded[:, index] = [
            codes.get(None if missing else _level_key(value), len(known))
            for value, missing in zip(column, _is_missing(column))
        ]
        unseen[names[index]] = int((coded[:, index] == len(known)).sum())
    return coded, unseen


def _with_one_hot(pipeline: Any, columns: List[int]) -> Any:
    """A built-in linear pipeline whose numeric steps skip `columns`, which are one-hot coded."""
    *numeric_steps, (_, estimator) = pipeline.steps
    return make_pipeline(
        ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), columns)],
            remainder=make_pipeline(*(step for _, step in numeric_steps)),
            # Dense output: several linear models reject sparse input.
            sparse_threshold=0,
        ),
        estimator,
    )


def _validate_y(y: Any, n_rows: int) -> np.ndarray:
    values = np.array(y)  # a copy, like _validate_X
    if values.ndim != 1:
        raise ValueError("[!] y must be one-dimensional.")
    if len(values) != n_rows:
        raise ValueError(
            f"[!] X has {n_rows} rows but y has {len(values)} rows."
        )
    if len(values) == 0:
        raise ValueError("[!] y must contain at least one value.")
    return values


def _validate_weights(weights: Any, n_rows: int) -> Optional[np.ndarray]:
    """None, or a validated float copy of the sample weights."""
    if weights is None:
        return None
    try:
        values = np.array(weights, dtype=float)  # a copy, like _validate_X
    except (TypeError, ValueError) as error:
        raise ValueError("[!] sample_weight must be numeric.") from error
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(
            f"[!] sample_weight must be one-dimensional with one value per row "
            f"({n_rows}), got shape {values.shape}."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("[!] sample_weight must be finite and non-negative.")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("[!] sample_weight must have a finite, positive sum.")
    return values


class SampleWeightNotSupported(Exception):
    """A model's final estimator has no sample_weight in fit()."""


def _takes_sample_weight(estimator: Any) -> bool:
    """fit() names sample_weight or takes **kwargs (a real error is a fit failure)."""
    parameters = inspect.signature(estimator.fit).parameters
    return "sample_weight" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _fit_weight_key(model: Any) -> Optional[str]:
    """fit() keyword that routes sample_weight to the final estimator, else None."""
    if not isinstance(model, Pipeline):
        return "sample_weight" if _takes_sample_weight(model) else None
    last_name, last = model.steps[-1]
    inner = _fit_weight_key(last)
    return None if inner is None else f"{last_name}__{inner}"


def _score_weights(weights: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """scorer() and dummy fit() keywords; empty without weights, so calls are unchanged."""
    return {} if weights is None else {"sample_weight": weights}


def _encode_labels(label_encoder: LabelEncoder, values: np.ndarray) -> np.ndarray:
    # LabelEncoder.transform casts to the training dtype (e.g. <U3), which
    # silently truncates longer eval labels, so check for unseen labels first.
    seen = set(label_encoder.classes_.tolist())
    unseen = sorted(set(values.tolist()) - seen, key=str)
    if unseen:
        raise ValueError(f"[!] y contains labels not seen during fit(): {unseen}")
    return label_encoder.transform(values)


def _float_target(values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("[!] Regression targets must be numeric.") from error


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _resolve_scorer(scoring: Any) -> tuple[Any, str]:
    if isinstance(scoring, str):
        return get_scorer(scoring), scoring
    if callable(scoring):
        return scoring, getattr(scoring, "__name__", type(scoring).__name__)
    raise TypeError("[!] scoring must be a scikit-learn scorer name or callable.")


def _failure(model: Optional[str], stage: str, error: Exception) -> Dict[str, Any]:
    return {
        "model": model,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _json_params(model: Any) -> Dict[str, Any]:
    """Estimator params restricted to JSON-safe scalars (finite if float)."""
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    return {
        key: value
        for key, value in params.items()
        if value is None
        or isinstance(value, (str, int, bool))
        or (isinstance(value, float) and math.isfinite(value))
    }


def _provenance() -> Dict[str, Optional[str]]:
    try:
        featranker_version = importlib.metadata.version("featranker")
    except importlib.metadata.PackageNotFoundError:
        featranker_version = None
    return {
        "featranker": featranker_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}")
    return result


def _resolve_feature_groups(
    feature_names: List[str],
    feature_groups: Optional[Dict[str, List[str]]],
) -> Dict[str, List[int]]:
    if feature_groups is None:
        return {name: [index] for index, name in enumerate(feature_names)}
    if not isinstance(feature_groups, dict):
        raise TypeError("[!] feature_groups must be a dict or None.")

    positions = {name: index for index, name in enumerate(feature_names)}
    resolved: Dict[str, List[int]] = {}
    assigned: Dict[str, str] = {}
    for group_name, members in feature_groups.items():
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("[!] Feature group names must be non-empty strings.")
        if not isinstance(members, (list, tuple)):
            raise TypeError(f"[!] Feature group '{group_name}' must contain a list.")
        if not members:
            raise ValueError(f"[!] Feature group '{group_name}' is empty.")
        if len(set(members)) != len(members):
            raise ValueError(
                f"[!] Feature group '{group_name}' has duplicate membership."
            )

        indices: List[int] = []
        for member in members:
            if member not in positions:
                raise ValueError(
                    f"[!] Feature group '{group_name}' contains unknown feature '{member}'."
                )
            if member in assigned:
                raise ValueError(
                    f"[!] Feature '{member}' overlaps groups "
                    f"'{assigned[member]}' and '{group_name}'."
                )
            assigned[member] = group_name
            indices.append(positions[member])
        resolved[group_name] = indices

    for name, index in positions.items():
        if name in assigned:
            continue
        if name in resolved:
            raise ValueError(
                f"[!] Feature group name '{name}' collides with implicit singleton '{name}'."
            )
        resolved[name] = [index]
    return resolved


HASH_CHUNK_ROWS = 65536


def _hash_column(digest: Any, column: np.ndarray) -> None:
    """Feed one column: numbers as float64 bytes, anything else as chunked reprs."""
    digest.update(f"{column.dtype.kind in 'biuf'}{len(column)}".encode())
    if column.dtype.kind in "biuf":
        numbers = column.astype(np.float64)
        # one bit pattern for every NaN
        digest.update(np.where(np.isnan(numbers), np.nan, numbers).tobytes())
        return
    for start in range(0, len(column), HASH_CHUNK_ROWS):
        digest.update(repr(column[start : start + HASH_CHUNK_ROWS].tolist()).encode())


def _fingerprint(names: List[str], X_values: np.ndarray, y_values: np.ndarray) -> str:
    """SHA-256 hex digest of the feature names, feature values, and target values."""
    digest = hashlib.sha256(json.dumps(names).encode())
    for column in (*X_values.T, y_values):
        _hash_column(digest, column)
    return digest.hexdigest()


def _weights_fingerprint(weights: Optional[np.ndarray]) -> Optional[str]:
    """SHA-256 hex digest of the sample weights, or None without weights."""
    if weights is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(weights, dtype=np.float64).tobytes()).hexdigest()


def _splits_fingerprint(splits: List[tuple]) -> str:
    """SHA-256 hex digest of every split's train then evaluation indices, in order."""
    digest = hashlib.sha256()
    for indices in (part for pair in splits for part in pair):
        digest.update(np.int64(len(indices)).tobytes())
        digest.update(indices.astype(np.int64).tobytes())
    return digest.hexdigest()


def _cv_record(cv: Any) -> Union[int, str]:
    """The integer given, or the splitter's repr."""
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        return int(cv)
    if not callable(getattr(cv, "split", None)):
        return "iterable of splits"
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(cv))  # addresses differ per run


def _assign_ranks(importance: Dict[str, float]) -> Dict[str, int]:
    """Assign descending minimum ranks; exact ties share a rank."""
    ordered = sorted(importance, key=lambda name: (-importance[name], name))
    ranks: Dict[str, int] = {}
    previous_score: Optional[float] = None
    previous_rank = 0
    for index, name in enumerate(ordered, start=1):
        score = importance[name]
        rank = previous_rank if previous_score is not None and score == previous_score else index
        ranks[name] = rank
        previous_score = score
        previous_rank = rank
    return ranks


def _build_consensus(model_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
    voters = [report for report in model_reports.values() if report["in_consensus"]]
    if not voters:
        warnings.warn(
            "No model beat the dummy baseline, so the consensus is empty.",
            UserWarning,
            # caller -> rank_features -> _build_consensus
            stacklevel=3,
        )
    # group -> family -> member ranks; each family then casts one vote.
    ranks_by_group: Dict[str, Dict[str, List[int]]] = {}
    # group -> family -> member above_shadow flags (only when shadows ran).
    above_by_group: Dict[str, Dict[str, List[bool]]] = {}
    for model_name, model_report in model_reports.items():
        if not model_report["in_consensus"]:
            continue
        family = model_report.get("family", model_name)
        for group_name, importance in model_report["importance"].items():
            ranks_by_group.setdefault(group_name, {}).setdefault(family, []).append(
                importance["rank"]
            )
            if "shadow_max" in model_report:
                above_by_group.setdefault(group_name, {}).setdefault(
                    family, []
                ).append(importance["above_shadow"])

    consensus = []
    for group_name, by_family in ranks_by_group.items():
        family_ranks = {
            family: float(np.median(ranks)) for family, ranks in sorted(by_family.items())
        }
        votes = list(family_ranks.values())
        consensus.append(
            {
                "feature_group": group_name,
                "median_rank": float(np.median(votes)),
                "mean_rank": float(np.mean(votes)),
                "rank_std": float(np.std(votes)),
                "n_models": sum(len(ranks) for ranks in by_family.values()),
                "n_families": len(family_ranks),
                "family_ranks": family_ranks,
            }
        )
        if group_name in above_by_group:
            fractions = {
                family: float(np.mean(flags))
                for family, flags in above_by_group[group_name].items()
            }
            consensus[-1]["above_shadow_share"] = float(np.mean(list(fractions.values())))
            consensus[-1]["family_above_shadow"] = dict(sorted(fractions.items()))
    return sorted(
        consensus,
        key=lambda row: (
            row["median_rank"],
            row["mean_rank"],
            row["feature_group"],
        ),
    )


CV_ERROR = (
    "[!] cv must be an int >= 2, a splitter with split(), or an iterable of "
    "(train_indices, eval_indices) pairs."
)


def _split_indices(indices: Any, n_rows: int) -> np.ndarray:
    values = np.asarray(indices)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.issubdtype(values.dtype, np.integer)
        or values.min() < 0
        or values.max() >= n_rows
    ):
        raise ValueError(
            "[!] Each split needs non-empty integer train and eval indices "
            f"within 0..{n_rows - 1}."
        )
    return values.astype(int)


def _resolve_splits(
    cv: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Any,
    task: str,
    random_state: Optional[int],
) -> List[tuple]:
    """Materialize cv into validated (train, eval) index pairs before any fit."""
    split_y = y_values
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError(CV_ERROR)
        # Shuffled folds: unshuffled KFold on sorted data gives misleading splits.
        splitter = StratifiedKFold if task == "clf" else KFold
        cv = splitter(n_splits=int(cv), shuffle=True, random_state=random_state)
        if task == "clf":
            # Stratify on class codes: fit() accepts any labels (even 0.1/0.2),
            # but StratifiedKFold rejects numeric labels it reads as continuous.
            # Only the fold assignment sees these codes; each split still fits
            # its own encoder on its training rows.
            split_y = LabelEncoder().fit_transform(y_values)
    if isinstance(cv, (str, bytes, bool)):  # str has .split(); reject it first
        raise ValueError(CV_ERROR)
    if callable(getattr(cv, "split", None)):
        pairs = cv.split(X_values, split_y, groups)
    elif not hasattr(cv, "__iter__"):
        raise ValueError(CV_ERROR)
    else:
        pairs = cv

    resolved = []
    for pair in pairs:
        try:
            train, evaluation = pair
        except (TypeError, ValueError) as error:
            raise ValueError(CV_ERROR) from error
        n_rows = len(X_values)
        train, evaluation = _split_indices(train, n_rows), _split_indices(evaluation, n_rows)
        if np.intersect1d(train, evaluation).size:
            raise ValueError("[!] Train and eval indices overlap within a split.")
        resolved.append((train, evaluation))
    if not resolved:
        raise ValueError("[!] cv produced no splits.")
    return resolved


def _consensus_positions(consensus: List[Dict[str, Any]]) -> Dict[str, int]:
    """1-based place in a sorted consensus; equal (median_rank, mean_rank) share the better place."""
    positions: Dict[str, int] = {}
    previous_key = None
    for index, row in enumerate(consensus, start=1):
        key = (row["median_rank"], row["mean_rank"])
        if key != previous_key:
            position, previous_key = index, key
        positions[row["feature_group"]] = position
    return positions


def _benjamini_hochberg(p_values: List[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, in input order, capped at 1."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(adjusted, 1.0)
    return q


def _build_stability(
    split_reports: List[Dict[str, Any]], n_splits: int, top_k: int
) -> List[Dict[str, Any]]:
    """Aggregate each group's per-split consensus position into one row per group."""
    group_names = list(split_reports[0]["report"]["feature_groups"]) if split_reports else []
    split_ranks: Dict[str, List[Optional[int]]] = {name: [] for name in group_names}
    shadow_shares: Dict[str, List[float]] = {}
    # group -> family -> per-split fractions, for the splits where the family voted.
    family_fractions: Dict[str, Dict[str, List[float]]] = {}
    for entry in split_reports:
        by_group = _consensus_positions(entry["report"]["consensus"])
        for name in group_names:
            split_ranks[name].append(by_group.get(name))
        for row in entry["report"]["consensus"]:
            if "above_shadow_share" in row:
                shadow_shares.setdefault(row["feature_group"], []).append(
                    row["above_shadow_share"]
                )
                for family, fraction in row.get("family_above_shadow", {}).items():
                    family_fractions.setdefault(row["feature_group"], {}).setdefault(
                        family, []
                    ).append(fraction)

    if not any(entry["report"]["consensus"] for entry in split_reports):
        warnings.warn(
            "No split produced a consensus, so the stability table is empty.",
            UserWarning,
            # caller -> rank_resampled -> _build_stability
            stacklevel=3,
        )
        return []

    rows = []
    for name, ranks in split_ranks.items():
        ranked = [rank for rank in ranks if rank is not None]
        rows.append(
            {
                "feature_group": name,
                "median_rank": float(np.median(ranked)),
                "min_rank": min(ranked),
                "max_rank": max(ranked),
                "split_ranks": ranks,
                "n_splits_ranked": len(ranked),
                # Failed and consensus-less splits count as "not in top k".
                "top_k_share": sum(rank <= top_k for rank in ranked) / n_splits,
            }
        )
        if name in shadow_shares:
            # Failed, empty and row-less splits count as 0, like shadow_hits.
            rows[-1]["above_shadow_share"] = float(np.sum(shadow_shares[name]) / n_splits)
            # One trial per split; a split without a consensus row is a non-hit.
            hits = sum(share > 0.5 for share in shadow_shares[name])
            rows[-1]["shadow_hits"] = hits
            rows[-1]["shadow_p_value"] = float(binom.sf(hits - 1, n_splits, 0.5))
        if name in family_fractions:
            # Splits where the family did not vote, failed, or were empty count
            # as 0, like shadow_hits: a family seen in a few splits cannot pass.
            shares = {
                family: float(np.sum(values) / n_splits)
                for family, values in sorted(family_fractions[name].items())
            }
            rows[-1]["family_above_shadow_share"] = shares
            rows[-1]["shadow_families"] = sum(share > 0.5 for share in shares.values())
    if shadow_shares:
        q_values = _benjamini_hochberg([row["shadow_p_value"] for row in rows])
        for row, q_value in zip(rows, q_values):
            row["shadow_q_value"] = float(q_value)
    return sorted(
        rows,
        key=lambda row: (row["median_rank"], -row["top_k_share"], row["feature_group"]),
    )


def _unrouted(call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Run call with metadata routing off: weights are passed explicitly, and
    routing would reject them (the setting is thread-local, so set it here).
    """
    with sklearn.config_context(enable_metadata_routing=False):
        return call(*args, **kwargs)


def _run_models(
    worker: Callable[..., tuple],
    jobs: List[tuple],
    n_jobs: Optional[int],
    desc: str,
    verbose: bool,
) -> List[tuple]:
    """Run one worker per model, in order; sequentially unless n_jobs > 1 or -1."""
    if n_jobs is None or n_jobs == 1:
        return [_unrouted(worker, *job) for job in tqdm(jobs, desc=desc, disable=not verbose)]
    return Parallel(n_jobs=n_jobs)(delayed(_unrouted)(worker, *job) for job in jobs)


def _with_shadows(X_values: np.ndarray, seed: Any, mode: str = "joint") -> np.ndarray:
    """
    Real columns followed by a shadow block.

    "joint": the same rows in one shuffled order. "independent": each column
    in its own shuffled order, drawn in column order from one generator.
    """
    rng = np.random.default_rng(seed)
    if mode == "independent":
        shadow = np.column_stack(
            [X_values[rng.permutation(len(X_values)), j] for j in range(X_values.shape[1])]
        )
    else:
        shadow = X_values[rng.permutation(len(X_values))]
    return np.hstack([X_values, shadow])


class ShadowColumnsDropped(Exception):
    """A model's ColumnTransformer ignores some shadow columns."""


def _column_transformers(model: Any) -> Iterator[Any]:
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            yield from _column_transformers(step)
    elif isinstance(model, ColumnTransformer):
        yield model


def _drops_shadows(name: str, model: Any, n_real: int, n_total: int) -> bool:
    """True if a fitted ColumnTransformer inside `model` leaves a shadow column unused."""
    shadows = set(range(n_real, n_total))
    for transformer in _column_transformers(model):
        # One that sees other columns than the model input is not about the shadows.
        if getattr(transformer, "n_features_in_", None) != n_total:
            continue
        try:
            indices = transformer._transformer_to_input_indices
            used: set = set()
            for step_name, step, _ in transformer.transformers:
                if not (isinstance(step, str) and step == "drop"):
                    used.update(indices[step_name])
            if not (isinstance(transformer.remainder, str) and transformer.remainder == "drop"):
                used.update(indices["remainder"])
        except Exception:  # private scikit-learn attributes; cannot verify
            warnings.warn(
                f"Cannot verify that model '{name}' keeps the shadow columns.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if not shadows <= used:
            return True
    return False


def _fit_model(
    name: str,
    model: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    names: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> tuple[str, Any, Optional[Dict[str, str]]]:
    try:
        fit_params: Dict[str, np.ndarray] = {}
        if weights is not None:
            key = _fit_weight_key(model)
            if key is None:
                raise SampleWeightNotSupported(
                    "The final estimator does not accept sample_weight in fit(), "
                    "so it is not fit unweighted."
                )
            fit_params[key] = weights
        fitted = clone(model, safe=False).fit(
            _model_X(X_values, names), y_values, **fit_params
        )
        return name, fitted, None
    except Exception as error:
        return name, None, _failure(name, "fit", error)


def _group_importances(
    model: Any,
    scorer: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Dict[str, List[int]],
    n_repeats: int,
    rng: np.random.Generator,
    baseline: float,
    names: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    # One scratch copy per model; only the group's columns are
    # rewritten per repeat and restored after the group.
    X_permuted = X_values.copy()
    importances: Dict[str, Dict[str, Any]] = {}
    for group_name, column_indices in groups.items():
        columns = np.asarray(column_indices, dtype=np.intp)
        values: List[float] = []
        for _ in range(n_repeats):
            row_order = rng.permutation(len(X_values))
            X_permuted[:, columns] = X_values[np.ix_(row_order, columns)]
            permuted_score = _finite_float(
                scorer(
                    model, _model_X(X_permuted, names), y_values, **_score_weights(weights)
                ),
                "permuted score",
            )
            values.append(
                _finite_float(baseline - permuted_score, "permutation importance")
            )
        X_permuted[:, columns] = X_values[:, columns]
        importances[group_name] = {
            "values": values,
            "mean": _finite_float(np.mean(values), "importance mean"),
            "std": _finite_float(np.std(values), "importance standard deviation"),
        }
    return importances


def _permutation_importances(
    model: Any,
    scorer: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Dict[str, List[int]],
    n_repeats: int,
    random_state: Optional[int],
    baseline: float,
    shadow_groups: Optional[Dict[str, List[int]]] = None,
    names: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]]]:
    rng = np.random.default_rng(random_state)
    importances = _group_importances(
        model, scorer, X_values, y_values, groups, n_repeats, rng, baseline, names,
        weights,
    )
    # Real groups are ranked among themselves only; shadows are compared, not ranked.
    ranks = _assign_ranks(
        {name: importance["mean"] for name, importance in importances.items()}
    )
    for group_name, rank in ranks.items():
        importances[group_name]["rank"] = rank
    # Shadows draw from the same stream after the real groups, so the real
    # values match a run without shadows on the same matrix.
    shadow = None
    if shadow_groups is not None:
        shadow = _group_importances(
            model, scorer, X_values, y_values, shadow_groups, n_repeats, rng, baseline,
            names, weights,
        )
    return importances, shadow


def _rank_model(
    name: str,
    model: Any,
    scorer: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Dict[str, List[int]],
    n_repeats: int,
    random_state: Optional[int],
    baseline_score: Optional[float],
    family: str,
    shadow_groups: Optional[Dict[str, List[int]]] = None,
    train_data: Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
    names: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> tuple[
    str, Optional[Dict[str, Any]], Optional[Dict[str, str]], Optional[Dict[str, str]]
]:
    """Returns (name, report, ranking failure, evidence failure)."""
    try:
        baseline = _finite_float(
            scorer(model, _model_X(X_values, names), y_values, **_score_weights(weights)),
            "baseline score",
        )
        importances, shadow = _permutation_importances(
            model, scorer, X_values, y_values, groups, n_repeats, random_state,
            baseline, shadow_groups, names, weights,
        )
        report = {
            "family": family,
            "evaluation_score": baseline,
            "in_consensus": baseline_score is None or baseline > baseline_score,
            "params": _json_params(model),
            "importance": importances,
        }
        if shadow is not None:
            shadow_max = max(entry["mean"] for entry in shadow.values())
            for entry in importances.values():
                entry["above_shadow"] = entry["mean"] > shadow_max
            report["shadow_importance"] = shadow
            report["shadow_max"] = shadow_max
        evidence_failure = None
        if train_data is not None:
            try:
                train_X, train_y, train_weights = train_data
                train_score = _finite_float(
                    scorer(
                        model, _model_X(train_X, names), train_y,
                        **_score_weights(train_weights),
                    ),
                    "train score",
                )
                report["train_score"] = train_score
                report["overfit_gap"] = _finite_float(
                    train_score - baseline, "overfit gap"
                )
            except Exception as error:
                report["train_score"] = report["overfit_gap"] = None
                evidence_failure = _failure(name, "evidence", error)
        return name, report, None, evidence_failure
    except Exception as error:
        return name, None, _failure(name, "ranking", error), None


class FeatureRanker:
    def __init__(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all", "fast"] = "all",
        models: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = True,
        shadows: bool = False,
        shadow_seed: Optional[int] = None,
        evidence: bool = False,
        shadow_mode: Literal["joint", "independent"] = "joint",
        missing: Literal["error", "impute"] = "error",
        categorical: Union[List[str], Literal["auto"], None] = None,
        model_input: Literal["array", "dataframe"] = "array",
    ) -> None:
        """
        Initialize configured models without loading or fitting data.

        Args:
            task: Task type — "clf" for classification or "reg" for regression.
            group: Built-in model group — "linear", "tree", "all" for both, or
                "fast" for one representative model per family.
            models: Optional name -> scikit-learn estimator mapping that
                replaces the built-in registry. Estimators are cloned at fit().
            n_jobs: Parallel workers over models in fit() and rank_features();
                None means sequential, -1 means all cores.
            verbose: Show progress bars during initialization, fit, and ranking.
            shadows: Train and rank with a shuffled shadow copy of every
                feature, to check which groups beat noise.
            shadow_seed: Seed for the shadow shuffles; None draws fresh entropy.
            evidence: Keep the training data so rank_features() can add
                mechanism evidence: train score, overfit gap, correlated
                groups, and (with the shap extra) a SHAP summary.
            shadow_mode: "joint" shuffles all columns with one shared row
                order (collinear columns stay collinear in the shadows);
                "independent" shuffles each shadow column on its own.
            missing: "error" rejects NaN in features. "impute" accepts NaN and
                gives every built-in model a median imputer as its first
                pipeline step, learned on each training split only; custom
                models receive NaN unchanged and must handle it. Infinite
                values are always rejected.
            categorical: None (default) accepts numeric features only. "auto"
                treats DataFrame columns of object, string, or category dtype
                as categorical; a list of feature names marks those columns
                (any dtype, NumPy included). Each categorical column is coded
                inside fit() from the training rows only, ranked as one
                feature, and one-hot encoded by built-in linear models; trees
                and custom models receive the integer codes.
            model_input: "array" (default) gives models a float NumPy array.
                "dataframe" gives every model a pandas DataFrame whose columns
                are the feature names, then "shadow__<name>" for each shadow
                column, so custom pipelines can select columns by name.
                Needs pandas.
        """
        if task not in {"clf", "reg"}:
            raise ValueError("[!] task must be 'clf' or 'reg'.")
        if group not in {"linear", "tree", "all", "fast"}:
            raise ValueError("[!] group must be 'linear', 'tree', 'all', or 'fast'.")
        if models is not None and group != "all":
            raise ValueError("[!] group selects built-in models; omit it with models.")
        if not isinstance(verbose, bool):
            raise TypeError("[!] verbose must be True or False.")
        if not isinstance(shadows, bool):
            raise TypeError("[!] shadows must be True or False.")
        if not isinstance(evidence, bool):
            raise TypeError("[!] evidence must be True or False.")
        if shadow_seed is not None and (
            not isinstance(shadow_seed, int) or isinstance(shadow_seed, bool)
        ):
            raise TypeError("[!] shadow_seed must be an int or None.")
        if shadow_seed is not None and shadow_seed < 0:
            # NumPy generators only accept non-negative integer seeds.
            raise ValueError("[!] shadow_seed must be non-negative.")
        if not isinstance(shadow_mode, str) or shadow_mode not in {"joint", "independent"}:
            raise ValueError("[!] shadow_mode must be 'joint' or 'independent'.")
        if not isinstance(missing, str) or missing not in {"error", "impute"}:
            raise ValueError("[!] missing must be 'error' or 'impute'.")
        if not isinstance(model_input, str) or model_input not in {"array", "dataframe"}:
            raise ValueError("[!] model_input must be 'array' or 'dataframe'.")
        if model_input == "dataframe" and _pd is None:
            raise ImportError("[!] model_input='dataframe' needs pandas: pip install pandas")
        self.task = task
        self.group = group
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.shadows = shadows
        self.shadow_seed = shadow_seed
        self.shadow_mode = shadow_mode
        self.evidence = evidence
        self.missing = missing
        self.model_input = model_input
        self.categorical = _validate_categorical(categorical)
        self.categorical_features_: List[str] = []
        self.categorical_levels_: Dict[str, List[Optional[str]]] = {}
        self.label_encoder = None
        self.initialization_failures_: List[Dict[str, str]] = []
        # Built-in model name -> registry family; anything else is its own family.
        self.model_families_: Dict[str, str] = {}
        self.user_models = None if models is None else _validate_models(models)
        self.models = {} if models is not None else self.init_models()
        self.trained_models: Dict[str, Any] = {}
        self.baseline_model_: Any = None
        self.fit_failures_: List[Dict[str, str]] = []
        self.is_fitted_ = False

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Rankers pickled by 0.4 lack the attributes added in 0.5.
        self.__dict__.update(state)
        self.__dict__.setdefault("verbose", True)
        # ... and those pickled by 0.6 lack the 0.7 shadow options.
        self.__dict__.setdefault("shadows", False)
        self.__dict__.setdefault("shadow_seed", None)
        # ... and those pickled by 1.0 lack the 1.1 evidence flag.
        self.__dict__.setdefault("evidence", False)
        # ... and those pickled by 1.5 lack the 1.6 shadow mode.
        self.__dict__.setdefault("shadow_mode", "joint")
        # ... and those pickled by 1.6 lack the 1.7 missing-value policy.
        self.__dict__.setdefault("missing", "error")
        # ... and those pickled by 1.8 lack the 1.9 categorical option.
        self.__dict__.setdefault("categorical", None)
        self.__dict__.setdefault("categorical_features_", [])
        self.__dict__.setdefault("categorical_levels_", {})
        # ... and those pickled by 1.9 lack the 1.10 model input.
        self.__dict__.setdefault("model_input", "array")
        # ... and those pickled by 1.10 lack the 1.11 sample weights.
        self.__dict__.setdefault("train_weighted_", False)
        self.__dict__.setdefault("train_weight_", None)
        # ... and those pickled by 1.11 lack the 1.12 training-data fingerprint.
        self.__dict__.setdefault("train_data_sha256_", None)
        self.__dict__.setdefault("train_weights_sha256_", None)
        if "model_families_" not in self.__dict__:
            self.model_families_ = (
                _registry_families() if self.user_models is None else {}
            )

    def _model_columns(self, names: List[str]) -> Optional[List[str]]:
        """Column names models see in "dataframe" mode (shadows last), else None."""
        if self.model_input != "dataframe":
            return None
        return names + [f"shadow__{name}" for name in names] if self.shadows else names

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
        sample_weight: Any = None,
    ) -> "FeatureRanker":
        """
        Fit configured models using training data only.

        sample_weight: optional 1-D non-negative weights, one per row. Every
        model is fit with them (built-in pipelines pass them to the final
        estimator); a model that cannot take weights is recorded as a fit
        failure, not fit unweighted.
        """
        self.is_fitted_ = False
        categorical = _categorical_indices(X, self.categorical, feature_names)
        X_values, names = _validate_X(
            X,
            feature_names=feature_names,
            allow_nan=self.missing == "impute",
            categorical=categorical,
        )
        if self.model_input == "dataframe" and self.shadows:
            clashing = [name for name in names if name.startswith("shadow__")]
            if clashing:
                raise ValueError(
                    f"[!] Feature names {clashing} start with 'shadow__', which "
                    "model_input='dataframe' reserves for shadow columns."
                )
        y_values = _validate_y(y, len(X_values))
        weights = _validate_weights(sample_weight, len(X_values))
        train_data_sha256 = _fingerprint(names, X_values, y_values)
        levels: Dict[str, List[Optional[str]]] = {}
        categorical_missing: Dict[str, int] = {}
        if categorical:
            categorical_missing = {
                names[index]: int(_is_missing(X_values[:, index]).sum())
                for index in categorical
            }
            levels = _learn_levels(X_values, categorical, names)
            X_values, _ = _encode_categorical(X_values, categorical, names, levels)

        # Encode every classification target to 0..n-1 (XGBoost requires it).
        label_encoder = None
        if self.task == "clf":
            label_encoder = LabelEncoder()
            y_values = label_encoder.fit_transform(y_values)
        else:
            y_values = _float_target(y_values)

        candidates: Dict[str, Any] = {}
        if self.user_models is not None:
            candidates.update(self.user_models)
        else:
            groups = (
                ["linear", "tree"] if self.group in {"all", "fast"} else [self.group]
            )
            # Shadow columns of categorical columns are categorical too.
            one_hot = categorical + [
                index + len(names) for index in categorical if self.shadows
            ]
            for group in groups:
                built = self.models.get(self.task, {}).get(group, {})
                if one_hot and group == "linear":
                    built = {
                        name: _with_one_hot(model, one_hot) for name, model in built.items()
                    }
                candidates.update(built)

        self.trained_models = {}
        self.fit_failures_ = []
        # Models see the shadow block too; the dummy below sees real columns only.
        X_models = (
            _with_shadows(X_values, self.shadow_seed, self.shadow_mode)
            if self.shadows
            else X_values
        )
        columns = self._model_columns(names)
        jobs = [
            (name, model, X_models, y_values, columns, weights)
            for name, model in candidates.items()
        ]
        for name, fitted, failure in _run_models(
            _fit_model, jobs, self.n_jobs, "[*] Training models", self.verbose
        ):
            if failure is None:
                self.trained_models[name] = fitted
            else:
                self.fit_failures_.append(failure)

        if not self.trained_models:
            details = json.dumps(self.initialization_failures_ + self.fit_failures_)
            raise RuntimeError(f"[!] No models fit successfully. Failures: {details}")

        baseline_model = None
        try:
            dummy = DummyClassifier() if self.task == "clf" else DummyRegressor()
            baseline_model = _unrouted(
                lambda: dummy.fit(X_values, y_values, **_score_weights(weights))
            )
        except Exception as error:
            self.fit_failures_.append(_failure(BASELINE_NAME, "fit", error))

        self.baseline_model_ = baseline_model
        if self.evidence:
            # As the models saw them (shadow block included), for train scores.
            self.train_X_ = X_models
            self.train_y_ = y_values
            self.train_weight_ = weights
        self.train_weighted_ = weights is not None
        self.train_data_sha256_ = train_data_sha256
        self.train_weights_sha256_ = _weights_fingerprint(weights)
        self.label_encoder = label_encoder
        self.feature_names_ = names
        self.categorical_features_ = [names[index] for index in categorical]
        self.categorical_levels_ = levels
        if self.missing == "impute":
            self.train_missing_counts_ = {
                name: categorical_missing.get(name, int(np.isnan(X_values[:, index]).sum()))
                for index, name in enumerate(names)
            }
        self.n_features_in_ = len(names)
        self.is_fitted_ = True
        return self

    def rank_features(
        self,
        X_eval: Any,
        y_eval: Any,
        *,
        scoring: Any,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        sample_weight: Any = None,
    ) -> Report:
        """
        Rank feature groups on explicitly supplied evaluation data.

        sample_weight: optional 1-D non-negative weights, one per evaluation
        row; the baseline, permuted and dummy scores are weighted with them.
        """
        if not self.is_fitted_:
            raise RuntimeError("[!] Call fit() before rank_features().")
        if not _is_positive_int(n_repeats):
            raise ValueError("[!] n_repeats must be a positive integer.")
        scorer, scoring_name = _resolve_scorer(scoring)

        categorical = [
            index
            for index, name in enumerate(self.feature_names_)
            if name in self.categorical_features_
        ]
        X_values, _ = _validate_X(
            X_eval,
            expected_names=self.feature_names_,
            allow_nan=self.missing == "impute",
            categorical=categorical,
        )
        unseen: Dict[str, int] = {}
        raw_values = X_values  # fingerprinted as given, before levels are coded
        if categorical:
            X_values, unseen = _encode_categorical(
                X_values, categorical, self.feature_names_, self.categorical_levels_
            )
        y_values = _validate_y(y_eval, len(X_values))
        weights = _validate_weights(sample_weight, len(X_values))
        run = {
            "task": self.task,
            "group": self.group,
            "data_sha256": _fingerprint(self.feature_names_, raw_values, y_values),
            "train_data_sha256": self.train_data_sha256_,
        }
        for key, value in (
            ("train_weights_sha256", self.train_weights_sha256_),
            ("weights_sha256", _weights_fingerprint(weights)),
        ):
            if value is not None:
                run[key] = value
        if self.label_encoder is not None:
            y_values = _encode_labels(self.label_encoder, y_values)
        else:
            y_values = _float_target(y_values)
        groups = _resolve_feature_groups(self.feature_names_, feature_groups)
        named_groups = {
            name: [self.feature_names_[index] for index in indices]
            for name, indices in groups.items()
        }

        model_reports: Dict[str, Any] = {}
        ranking_failures: List[Dict[str, str]] = []
        baseline_score = self._score_baseline(
            scorer, X_values, y_values, ranking_failures, weights
        )
        X_models, shadow_groups = X_values, None
        if self.shadows:
            # An independent, reproducible shuffle of the evaluation rows.
            eval_seed = None if self.shadow_seed is None else [self.shadow_seed, 1]
            X_models = _with_shadows(X_values, eval_seed, self.shadow_mode)
            offset = len(self.feature_names_)
            shadow_groups = {
                name: [index + offset for index in indices]
                for name, indices in groups.items()
            }
        train_data = (
            (self.train_X_, self.train_y_, self.train_weight_) if self.evidence else None
        )
        columns = self._model_columns(self.feature_names_)
        # A model that ignores the shadow columns gets no shadow evidence.
        shadow_failures: List[Dict[str, str]] = []
        no_shadows: set = set()
        if self.shadows:
            n_real = len(self.feature_names_)
            for name, model in self.trained_models.items():
                if _drops_shadows(name, model, n_real, 2 * n_real):
                    no_shadows.add(name)
                    shadow_failures.append(
                        _failure(
                            name,
                            "shadows",
                            ShadowColumnsDropped(
                                "A ColumnTransformer in the model does not use every "
                                "shadow column, so it is left out of the shadow evidence."
                            ),
                        )
                    )
        jobs = [
            (name, model, scorer, X_models, y_values, groups, n_repeats,
             random_state, baseline_score, self.model_families_.get(name, name),
             None if name in no_shadows else shadow_groups, train_data, columns, weights)
            for name, model in self.trained_models.items()
        ]
        evidence_failures: List[Dict[str, str]] = []
        for name, model_report, failure, evidence_failure in _run_models(
            _rank_model, jobs, self.n_jobs, "[*] Ranking models", self.verbose
        ):
            if failure is None:
                model_reports[name] = model_report
            else:
                ranking_failures.append(failure)
            if evidence_failure is not None:
                evidence_failures.append(evidence_failure)

        if not model_reports:
            details = json.dumps(ranking_failures)
            raise RuntimeError(
                f"[!] No models ranked successfully. Failures: {details}"
            )

        report = {
            "evaluation_mode": "held_out",
            "scoring": scoring_name,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "feature_groups": named_groups,
            "baseline_score": baseline_score,
            "provenance": _provenance(),
            "run": run,
            "models": model_reports,
            "consensus": _build_consensus(model_reports),
            "failures": self.initialization_failures_
            + self.fit_failures_
            + ranking_failures
            + evidence_failures
            + shadow_failures,
        }
        if self.model_input == "dataframe":
            report["model_input"] = "dataframe"
        if self.train_weighted_ or weights is not None:
            report["sample_weight"] = {
                "train": self.train_weighted_,
                "eval": weights is not None,
            }
        if self.shadows:
            report["shadows"] = True
            report["shadow_seed"] = self.shadow_seed
            report["shadow_mode"] = self.shadow_mode
        if self.missing == "impute":
            report["missing"] = {
                "policy": "impute",
                "strategy": "median",
                "train_counts": dict(self.train_missing_counts_),
            }
        if categorical:
            report["categorical"] = {
                "columns": list(self.categorical_features_),
                "train_levels": {
                    name: len(levels) for name, levels in self.categorical_levels_.items()
                },
                "unseen_eval": unseen,
            }
        if self.evidence:
            n_real = len(self.feature_names_)
            shap_block, shap_models = shap_evidence(
                {name: self.trained_models[name] for name in model_reports},
                X_models,
                self.train_X_,
                groups,
                categorical,
                columns,
            )
            for name, entry in shap_models.items():
                model_reports[name]["shap"] = entry
            report["evidence"] = {
                "correlation": correlation_evidence(
                    self.train_X_[:, :n_real], groups, categorical
                ),
                "shap": shap_block,
            }
        return Report(report)

    def rank_resampled(
        self,
        X: Any,
        y: Any,
        *,
        scoring: Any,
        cv: Any = 5,
        groups: Any = None,
        feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        top_k: int = 5,
        sample_weight: Any = None,
    ) -> Report:
        """
        Fit and rank on every cv split of X/y and report rank stability.

        sample_weight: optional 1-D non-negative weights, one per row; each
        split trains with its training rows' weights and scores with its
        evaluation rows' weights.

        Each split uses a fresh ranker with this ranker's settings; this
        ranker's own fitted state is not changed. Pass development data only,
        never the final test set.
        """
        categorical = _categorical_indices(X, self.categorical, feature_names)
        # Raw values: each split learns its own levels in fit().
        X_values, names = _validate_X(
            X,
            feature_names=feature_names,
            allow_nan=self.missing == "impute",
            categorical=categorical,
        )
        y_values = _validate_y(y, len(X_values))
        weights = _validate_weights(sample_weight, len(X_values))
        if not _is_positive_int(n_repeats):
            raise ValueError("[!] n_repeats must be a positive integer.")
        if not _is_positive_int(top_k):
            raise ValueError("[!] top_k must be a positive integer.")
        _resolve_scorer(scoring)
        _resolve_feature_groups(names, feature_groups)
        splits = _resolve_splits(
            cv, X_values, y_values, groups, self.task, random_state
        )

        split_reports: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for index, (train, evaluation) in enumerate(splits):
            try:
                ranker = FeatureRanker(
                    self.task,
                    self.group,
                    models=self.user_models,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    shadows=self.shadows,
                    shadow_seed=(
                        None if self.shadow_seed is None else self.shadow_seed + index
                    ),
                    evidence=self.evidence,
                    shadow_mode=self.shadow_mode,
                    missing=self.missing,
                    model_input=self.model_input,
                    # Names, so "auto" survives the loss of the DataFrame dtypes.
                    categorical=[names[column] for column in categorical] or None,
                )
                ranker.fit(
                    X_values[train],
                    y_values[train],
                    feature_names=names,
                    sample_weight=None if weights is None else weights[train],
                )
                report = ranker.rank_features(
                    X_values[evaluation],
                    y_values[evaluation],
                    scoring=scoring,
                    feature_groups=feature_groups,
                    n_repeats=n_repeats,
                    random_state=None if random_state is None else random_state + index,
                    sample_weight=None if weights is None else weights[evaluation],
                )
            except Exception as error:
                failures.append({"split": index, **_failure(None, "split", error)})
                continue
            split_reports.append(
                {
                    "split": index,
                    "n_train": len(train),
                    "n_eval": len(evaluation),
                    "report": report,
                }
            )
            failures.extend({"split": index, **f} for f in report["failures"])

        if not split_reports:
            raise RuntimeError(
                f"[!] No split ranked successfully. Failures: {json.dumps(failures)}"
            )
        first = split_reports[0]["report"]
        run: Dict[str, Any] = {
            "task": self.task,
            "group": self.group,
            "data_sha256": _fingerprint(names, X_values, y_values),
            "cv": _cv_record(cv),
            "splits_sha256": _splits_fingerprint(splits),
        }
        if weights is not None:
            run["weights_sha256"] = _weights_fingerprint(weights)
        result = {
            "evaluation_mode": "resampled",
            "scoring": first["scoring"],
            "n_repeats": n_repeats,
            "random_state": random_state,
            "feature_groups": first["feature_groups"],
            "provenance": _provenance(),
            "run": run,
            "top_k": top_k,
            "n_splits": len(splits),
            "splits": split_reports,
            "stability": _build_stability(split_reports, len(splits), top_k),
            "failures": failures,
        }
        if self.model_input == "dataframe":
            result["model_input"] = "dataframe"
        if self.shadows:
            result["shadows"] = True
            result["shadow_seed"] = self.shadow_seed
            result["shadow_mode"] = self.shadow_mode
            result["shadow_test"] = {
                "method": "binomial",
                "hit": "above_shadow_share > 0.5",
                "null_probability": 0.5,
                "correction": "benjamini-hochberg",
                "n_trials": len(splits),
                "min_p_value": 0.5 ** len(splits),
                "calibrated": False,
                "note": SHADOW_TEST_NOTE,
            }
        return Report(result)

    def _score_baseline(
        self,
        scorer: Any,
        X_values: np.ndarray,
        y_values: np.ndarray,
        failures: List[Dict[str, str]],
        weights: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Score the dummy baseline; None (no gating) if it was unusable."""
        if self.baseline_model_ is None:
            return None
        try:
            return _finite_float(
                _unrouted(
                    scorer, self.baseline_model_, X_values, y_values,
                    **_score_weights(weights),
                ),
                "baseline score",
            )
        except Exception as error:
            failures.append(_failure(BASELINE_NAME, "ranking", error))
            return None

    def init_models(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Read model definitions from importance_config.yaml and instantiate them.

        Only models that match the requested task and group are created, so
        unnecessary work is avoided when the user narrows the scope.

        Returns:
            Nested dict keyed by [task][group][model_name] → model instance.
        """
        # Load the YAML config that ships alongside this module.
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        models: Dict[str, Dict[str, Dict[str, Any]]] = {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {}, "tree": {}},
        }

        # Determine which tasks and groups to initialize.
        allowed_tasks = {"clf", "reg"} if self.task == "all" else {self.task}
        allowed_groups = (
            {"linear", "tree"} if self.group in {"all", "fast"} else {self.group}
        )

        for task_name, groups in config.items():
            task_key = task_map.get(task_name)
            if task_key is None or task_key not in allowed_tasks:
                continue
            if not isinstance(groups, dict):
                continue

            for group_name, model_configs in groups.items():
                if (
                    group_name not in models[task_key]
                    or group_name not in allowed_groups
                ):
                    continue
                if not isinstance(model_configs, list):
                    continue

                if self.group == "fast":
                    model_configs = [
                        m
                        for m in model_configs
                        if isinstance(m, dict) and m.get("fast") is True
                    ]
                group_models: Dict[str, Any] = {}
                for model in tqdm(
                    model_configs,
                    desc=f"[*] Initialize {task_name}-{group_name} models",
                    disable=not self.verbose,
                ):
                    if not isinstance(model, dict):
                        continue

                    name = model.get("name")
                    module_path = model.get("import")
                    class_name = model.get("class")
                    params = model.get("params", {})
                    if params is None:
                        params = {}
                    if not name or not module_path or not class_name:
                        continue

                    try:
                        module = importlib.import_module(module_path)
                        model_class = getattr(module, class_name, None)
                        if model_class is None:
                            raise ValueError(
                                f"Model class '{class_name}' not found in "
                                f"module '{module_path}'"
                            )
                        if not isinstance(params, dict):
                            raise ValueError(
                                f"Params must be a dict, got {type(params).__name__}"
                            )
                        estimator = model_class(**_model_params(class_name, params))
                        # Linear models are scale-sensitive; the scaler (and
                        # the median imputer) is fit on training data only,
                        # inside the pipeline.
                        steps: List[Any] = []
                        if self.missing == "impute":
                            # keep_empty_features: a column missing in every
                            # training row stays in place (as 0) instead of
                            # shifting later columns.
                            steps.append(
                                SimpleImputer(strategy="median", keep_empty_features=True)
                            )
                        if group_name == "linear":
                            steps.append(StandardScaler())
                        group_models[name] = (
                            make_pipeline(*steps, estimator) if steps else estimator
                        )
                        self.model_families_[name] = str(model.get("family") or name)
                    except Exception as error:
                        self.initialization_failures_.append(
                            _failure(name, "initialization", error)
                        )
                        dependency = str(module_path).split(".", 1)[0]
                        if dependency in OPTIONAL_MODEL_MODULES and isinstance(
                            error, (ImportError, OSError)
                        ):
                            reason = (
                                f"optional dependency '{dependency}' failed to "
                                f"initialize: {error}"
                            )
                        else:
                            reason = f"initialization failed: {error}"
                        warnings.warn(
                            f"Skipping model '{name}': {reason}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                models[task_key][group_name] = group_models

        return models
