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
import numbers
import platform
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import sklearn
import yaml
from joblib import Parallel, delayed
from scipy.stats import binom
from sklearn.base import clone, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.fixes import parse_version
from tqdm import tqdm

try:
    # Private scikit-learn API (permutation_importance uses it too). If it
    # moves, _MultimetricScorer is None and scorers are called one by one.
    from sklearn.metrics import _scorer as _sklearn_scorer
    from sklearn.metrics._scorer import _MultimetricScorer

    # scikit-learn's own scorer classes, whose _score is all their __call__ does.
    _NATIVE_SCORERS = tuple(
        cls
        for name in ("_Scorer", "_PredictScorer", "_ProbaScorer", "_ThresholdScorer")
        if (cls := getattr(_sklearn_scorer, name, None)) is not None
    )
except ImportError:  # pragma: no cover
    _MultimetricScorer = None
    _NATIVE_SCORERS = ()

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


def _validate_families(families: Any) -> Optional[Dict[str, str]]:
    """None or a dict copy of a non-empty-string model name -> family mapping."""
    if families is None:
        return None
    if not isinstance(families, Mapping) or not all(
        isinstance(name, str) and name and isinstance(family, str) and family
        for name, family in families.items()
    ):
        raise ValueError(
            "[!] families must be None or a mapping of model name -> non-empty family string."
        )
    return dict(families)


def _constant_within_groups(
    X_values: np.ndarray, labels: np.ndarray, groups: Dict[str, List[int]]
) -> List[str]:
    """Sorted names of the groups whose every column is constant within every label."""
    _, first_rows, inverse = np.unique(labels, return_index=True, return_inverse=True)
    constant = []
    for name, indices in groups.items():
        block = np.asarray(X_values[:, indices], dtype=float)
        reference = block[first_rows][inverse]
        same = (block == reference) | (np.isnan(block) & np.isnan(reference))
        if same.all():
            constant.append(name)
    return sorted(constant)


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


def _validate_max_categories(max_categories: Any) -> Optional[int]:
    """None, or an int of at least 2 (a plain int, so the report stays JSON-safe)."""
    if max_categories is None:
        return None
    if (
        not isinstance(max_categories, numbers.Integral)
        or isinstance(max_categories, bool)
        or max_categories < 2
    ):
        raise ValueError("[!] max_categories must be None or an int of at least 2.")
    return int(max_categories)


def _with_one_hot(pipeline: Any, columns: List[int], max_categories: Optional[int] = None) -> Any:
    """
    A built-in linear pipeline whose numeric steps skip `columns`, which are one-hot coded.

    With `max_categories`, each column keeps at most that many one-hot columns
    (the infrequent levels share one). Unseen levels join it only for a column
    with at least `max_categories` levels; with fewer they are all zeros.
    """
    *numeric_steps, (_, estimator) = pipeline.steps
    capped = (
        {}
        if max_categories is None
        else {"max_categories": max_categories, "handle_unknown": "infrequent_if_exist"}
    )
    return make_pipeline(
        ColumnTransformer(
            [("cat", OneHotEncoder(**{"handle_unknown": "ignore", **capped}), columns)],
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


def _resolve_max_samples(value: Any, n_rows: int) -> int:
    """Rows to score: a positive int (capped at n_rows) or a float share in (0, 1]."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise ValueError("[!] max_samples must be a positive int or a float in (0, 1].")
    if isinstance(value, numbers.Integral) and int(value) >= 1:
        return min(int(value), n_rows)
    if not isinstance(value, numbers.Integral) and 0 < float(value) <= 1:
        return max(1, int(float(value) * n_rows))
    raise ValueError("[!] max_samples must be a positive int or a float in (0, 1].")


def _plain_number(value: Any) -> Any:
    """A numpy scalar as a plain int or float, so the report stays JSON-safe."""
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _resolve_scorer(scoring: Any) -> tuple[Any, str]:
    if isinstance(scoring, str):
        return get_scorer(scoring), scoring
    if callable(scoring):
        # make_scorer objects have no __name__; their score function does.
        named = getattr(scoring, "_score_func", scoring)
        return scoring, getattr(named, "__name__", type(scoring).__name__)
    raise TypeError("[!] scoring must be a scikit-learn scorer name or callable.")


def _resolve_scorers(scoring: Any) -> List[tuple[Any, str]]:
    """[(scorer, name)], primary first; a name or callable is a one-element list."""
    if not isinstance(scoring, (list, tuple)):
        return [_resolve_scorer(scoring)]
    if not scoring:
        raise ValueError("[!] scoring must contain at least one scorer.")
    resolved = [_resolve_scorer(entry) for entry in scoring]
    names = [name for _, name in resolved]
    if len(set(names)) != len(names):
        raise ValueError(
            "[!] Scorer names in scoring must be unique. Wrap a scorer in a function "
            "with a distinct __name__ to tell two apart."
        )
    return resolved


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


FINGERPRINT_SCHEME = 2
# Integer and boolean columns hash as int64, unsigned ones as uint64, never as float.
EXACT_INTEGER_TYPES = {"b": "<i8", "i": "<i8", "u": "<u8"}
# Booleans hash as the integers 0 and 1, so they share the integer tag.
INTEGER_TAGS = {"b": "i", "i": "i", "u": "u"}


def _hash_column(digest: Any, column: np.ndarray, scheme: int = FINGERPRINT_SCHEME) -> None:
    """
    Feed one column: numbers as bytes, anything else as chunked reprs.

    Scheme 1 hashed every number as float64, so integers above 2**53 collided.
    Scheme 2 hashes b/i/u columns losslessly, tagged with the dtype kind (not
    its size: int32 and int64 with equal values agree); float columns keep the
    scheme 1 bytes, so their fingerprints do not change.
    """
    kind = column.dtype.kind
    if scheme >= 2 and kind in EXACT_INTEGER_TYPES:
        digest.update(f"{INTEGER_TAGS[kind]}:{len(column)}".encode())
        digest.update(np.ascontiguousarray(column.astype(EXACT_INTEGER_TYPES[kind])).tobytes())
        return
    digest.update(f"{kind in 'biuf'}{len(column)}".encode())
    if kind in "biuf":
        numbers = column.astype(np.float64)
        # one bit pattern for every NaN
        digest.update(np.where(np.isnan(numbers), np.nan, numbers).tobytes())
        return
    for start in range(0, len(column), HASH_CHUNK_ROWS):
        digest.update(repr(column[start : start + HASH_CHUNK_ROWS].tolist()).encode())


def _fingerprint(
    names: List[str],
    X_values: np.ndarray,
    y_values: np.ndarray,
    scheme: int = FINGERPRINT_SCHEME,
) -> str:
    """SHA-256 hex digest of the feature names, feature values, and target values."""
    digest = hashlib.sha256(json.dumps(names).encode())
    for column in (*X_values.T, y_values):
        _hash_column(digest, column, scheme)
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


def _shuffle_labels(
    shuffle: str,
    labels: Any,
    n_rows: int,
    name: str = "shuffle_groups",
    singletons_ok: bool = False,
) -> Optional[np.ndarray]:
    """
    Validated group labels for `n_rows` rows (None unless "within_groups").
    `singletons_ok` skips the all-singleton check, for labels that are not used.
    """
    if shuffle == "circular" and n_rows < 2:
        raise ValueError(f"[!] shuffle='circular' needs at least 2 rows, got {n_rows}.")
    if shuffle != "within_groups":
        if labels is not None:
            raise ValueError(f"[!] {name} needs shuffle='within_groups'.")
        return None
    if labels is None:
        raise ValueError(f"[!] shuffle='within_groups' needs {name}, one label per row.")
    values = np.asarray(labels)
    if values.shape != (n_rows,):
        raise ValueError(
            f"[!] {name} needs one label per row: {n_rows} rows, got shape {values.shape}."
        )
    if any(item is None or item != item for item in values.tolist()):
        raise ValueError(f"[!] {name} must not contain None or NaN.")
    try:
        n_groups = len(np.unique(values))
    except TypeError:
        raise ValueError(f"[!] {name} labels must be sortable, not of mixed types.") from None
    if n_groups == n_rows and not singletons_ok:
        raise ValueError(
            "[!] shuffle='within_groups' needs at least one group with two rows."
        )
    return values


def _sample_whole_groups(rng: np.random.Generator, labels: np.ndarray, size: int) -> np.ndarray:
    """Sorted rows of whole groups, drawn in random order until `size` rows are covered."""
    _, inverse = np.unique(labels, return_inverse=True)
    order = rng.permutation(inverse.max() + 1)
    covered = np.cumsum(np.bincount(inverse)[order])
    chosen = order[: np.searchsorted(covered, size) + 1]
    return np.flatnonzero(np.isin(inverse, chosen))


def _labels_fingerprint(
    labels: Optional[np.ndarray], scheme: int = FINGERPRINT_SCHEME
) -> Optional[str]:
    """SHA-256 hex digest of the shuffle group labels (or rows), or None without labels."""
    if labels is None:
        return None
    digest = hashlib.sha256()
    _hash_column(digest, labels, scheme)
    return digest.hexdigest()


def _shuffle_index(
    rng: np.random.Generator, n: int, mode: str, labels: Optional[np.ndarray]
) -> np.ndarray:
    """Row index for one shuffle of n rows: "rows", "within_groups" or "circular"."""
    if mode == "circular":
        return np.roll(np.arange(n), rng.integers(1, n))
    if mode == "within_groups":
        assert labels is not None  # checked by _shuffle_labels
        # Sort rows by (label, random key): a random order inside each group,
        # and a group of one maps to itself.
        grouped = np.argsort(labels, kind="stable")
        index = np.empty(n, dtype=np.intp)
        index[grouped] = np.lexsort((rng.random(n), labels))
        return index
    return rng.permutation(n)


# Cells gathered at a time when building shadows (2 MB of float64).
SHADOW_CHUNK_CELLS = 1 << 18


def _with_shadows(
    X_values: np.ndarray,
    seed: Any,
    mode: str = "joint",
    shuffle: str = "rows",
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Real columns followed by a shadow block.

    "joint": the same rows in one shuffled order. "independent": each column
    in its own shuffled order, drawn in column order from one generator.
    `shuffle` and `labels` choose how rows are shuffled (_shuffle_index).
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = X_values.shape
    # One preallocated output: the gather goes in row chunks, not a full copy.
    out = np.empty((n_rows, 2 * n_cols), dtype=X_values.dtype)
    out[:, :n_cols] = X_values
    if mode == "independent":
        for j in range(n_cols):
            out[:, n_cols + j] = X_values[_shuffle_index(rng, n_rows, shuffle, labels), j]
        return out
    index = _shuffle_index(rng, n_rows, shuffle, labels)
    step = max(1, SHADOW_CHUNK_CELLS // max(1, n_cols))
    for start in range(0, n_rows, step):
        out[start : start + step, n_cols:] = X_values[index[start : start + step]]
    return out


class ShadowColumnsDropped(Exception):
    """A model's ColumnTransformer ignores some shadow columns."""


def _is_string(value: Any, text: str) -> bool:
    return isinstance(value, str) and value == text


# Steps that return the columns they receive, one for one, in the same order.
ONE_TO_ONE_SCALERS = (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)


def _keeps_columns(step: Any) -> bool:
    """Whether a fitted Pipeline step is known to return its input columns one for one."""
    if step is None or _is_string(step, "passthrough") or isinstance(step, ONE_TO_ONE_SCALERS):
        return True
    if isinstance(step, SimpleImputer):  # an all-NaN column may be dropped
        return not step.add_indicator and len(step.get_feature_names_out()) == step.n_features_in_
    return False


def _is_column_aware(estimator: Any) -> bool:
    """Whether the estimator is, or contains at any depth, a selector, FeatureUnion or ColumnTransformer."""
    parts = [estimator]
    if hasattr(estimator, "get_params"):
        parts.extend(estimator.get_params(deep=True).values())
    return any(isinstance(part, (SelectorMixin, FeatureUnion, ColumnTransformer)) for part in parts)


def _step_used(step: Any, columns: List[int]) -> set:
    """The columns of `columns` a ColumnTransformer or FeatureUnion part uses."""
    if _is_string(step, "drop"):
        return set()
    if _is_string(step, "passthrough"):
        return set(columns)
    return _kept_inputs(step, columns)


def _pipeline_kept(pipeline: Pipeline, columns: List[int]) -> set:
    """
    The columns of a Pipeline's first step, narrowed by later steps that still see them.

    A later step that is or contains a selector, FeatureUnion or ColumnTransformer
    is mapped only while every earlier step is known to keep the columns one for
    one; after any other step it cannot be mapped, so the result is unknown (raises).
    """
    steps = [
        step for _, step in pipeline.steps
        if step is not None and not _is_string(step, "passthrough")
    ]
    if not steps:
        return set(columns)
    kept = _kept_inputs(steps[0], columns)
    preserved = _keeps_columns(steps[0])
    for later in steps[1:]:
        if preserved:
            kept &= _kept_inputs(later, columns)
        elif _is_column_aware(later):
            raise ValueError("a column-aware step follows a step that may change the columns")
        preserved = preserved and _keeps_columns(later)
    return kept


def _kept_inputs(estimator: Any, columns: List[int]) -> set:
    """
    The model-input columns `estimator` uses, given that it receives `columns`.

    Looks into ColumnTransformer branches, FeatureUnion parts, fitted feature
    selectors, and a Pipeline's steps (see _pipeline_kept). Any other estimator
    is assumed to use every column, unless it wraps a selector, FeatureUnion or
    ColumnTransformer (a meta-estimator), which cannot be followed. Raises when the columns
    cannot be determined; the caller treats that as unverifiable.
    """
    if isinstance(estimator, Pipeline):
        return _pipeline_kept(estimator, columns)
    if isinstance(estimator, FeatureUnion):
        return set().union(*(_step_used(part, columns) for _, part in estimator.transformer_list))
    if isinstance(estimator, SelectorMixin):
        return {columns[i] for i in estimator.get_support(indices=True)}
    if not isinstance(estimator, ColumnTransformer):
        if _is_column_aware(estimator):
            raise ValueError("a meta-estimator wraps a selector, FeatureUnion or ColumnTransformer")
        return set(columns)
    if estimator.n_features_in_ != len(columns):
        raise ValueError("transformer width differs from its input")
    indices = estimator._transformer_to_input_indices
    used: set = set()
    for name, step, _ in estimator.transformers_:
        branch = [columns[i] for i in indices[name]]
        if branch:
            used |= _step_used(step, branch)
    return used


SHADOWS_KEPT, SHADOWS_DROPPED, SHADOWS_UNVERIFIED = "kept", "dropped", "unverified"


def _shadow_status(model: Any, n_real: int, n_total: int) -> str:
    """Whether the fitted model provably uses every shadow column."""
    try:
        used = _kept_inputs(model, list(range(n_total)))
    except Exception:  # private scikit-learn attributes; cannot verify
        return SHADOWS_UNVERIFIED
    return SHADOWS_KEPT if set(range(n_real, n_total)) <= used else SHADOWS_DROPPED


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


def _call_scorer(scorer: Any, model: Any, X: Any, y: np.ndarray, kwargs: Dict[str, Any]) -> Any:
    """The score, or the exception the scorer raised."""
    try:
        return scorer(model, X, y, **kwargs)
    except Exception as error:
        return error


_DIRECT = object()  # share key of a scorer that must be called on its own


def _share_key(scorer: Any, is_regression: bool, weighted: bool) -> Any:
    """
    What scorers must agree on to share one prediction: the positive label they
    resolve (it picks the probability column or flips a decision function), or
    _DIRECT for a scorer scikit-learn's multi-metric scorer cannot be trusted
    with (anything but an instance of one of scikit-learn's own scorer classes).
    Under weights it passes sample_weight only to score functions that name the
    parameter, so others are called directly.
    """
    # Exact type: a subclass may override __call__, which the multi-metric path skips.
    if type(scorer) not in _NATIVE_SCORERS:
        return _DIRECT
    try:
        if weighted and "sample_weight" not in inspect.signature(scorer._score_func).parameters:
            return _DIRECT
        if is_regression or getattr(scorer, "_response_method", None) == "predict":
            return None
        label = scorer._get_pos_label()
        hash(label)
        return label
    except Exception:
        return _DIRECT


def _shared_group_scores(
    group: Dict[int, Any], model: Any, X: Any, y: np.ndarray, kwargs: Dict[str, Any]
) -> Dict[int, Any]:
    try:
        scores = _MultimetricScorer(
            scorers={str(i): scorer for i, scorer in group.items()}, raise_exc=False
        )(model, X, y, **kwargs)
        return {int(key): value for key, value in scores.items() if not isinstance(value, str)}
    except Exception:
        return {}  # e.g. a response method is missing or the result is odd: score one by one


def _shared_scores(
    scorers: Sequence[Any], model: Any, X: Any, y: np.ndarray, kwargs: Dict[str, Any]
) -> Dict[int, Any]:
    """
    {position: score} for the scorers that scikit-learn can serve from one shared
    prediction per response method (one group per resolved positive label);
    {} if there is no gain or it is unavailable. A scorer that failed there is
    absent, so the caller re-calls it directly.
    """
    if _MultimetricScorer is None:
        return {}
    try:
        is_regression = is_regressor(model)
    except Exception:
        is_regression = False
    groups: Dict[Any, Dict[int, Any]] = {}
    for i, scorer in enumerate(scorers):
        key = _share_key(scorer, is_regression, "sample_weight" in kwargs)
        if key is not _DIRECT:
            groups.setdefault(key, {})[i] = scorer
    shared: Dict[int, Any] = {}
    for group in groups.values():
        if len(group) >= 2:
            shared.update(_shared_group_scores(group, model, X, y, kwargs))
    return shared


def _score_all(
    scorers: Sequence[Any],
    model: Any,
    X: Any,
    y: np.ndarray,
    weights: Optional[np.ndarray],
    share: bool = True,
) -> List[Any]:
    """
    One entry per scorer: its score, or the exception it raised (the same one
    a direct call raises, since failures are re-run directly). With share, scorers
    that agree on response method and positive label share one prediction.
    """
    kwargs = _score_weights(weights)
    shared = _shared_scores(scorers, model, X, y, kwargs) if share else {}
    return [
        shared[i] if i in shared else _call_scorer(scorer, model, X, y, kwargs)
        for i, scorer in enumerate(scorers)
    ]


def _shared_matches(direct: Sequence[Any], shared: Dict[int, Any]) -> bool:
    """True if every shared score equals the directly computed one (no failure there)."""
    return all(
        not isinstance(direct[i], Exception) and direct[i] == value
        for i, value in shared.items()
    )


def _unwrap(result: Any) -> Any:
    """A score, or raise the exception _score_all returned in its place."""
    if isinstance(result, Exception):
        raise result
    return result


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
    secondary: Optional[Dict[str, tuple[Any, float]]] = None,
    shuffle: tuple[str, Optional[np.ndarray]] = ("rows", None),
    share: bool = True,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Importances per group, and per secondary scorer ({name: (scorer, baseline)})
    its {group: {mean, std}} or, if it raised, the exception. Secondary
    scorers score the same permuted copies as the primary.
    """
    # One scratch copy per model; only the group's columns are
    # rewritten per repeat and restored after the group.
    X_permuted = X_values.copy()
    importances: Dict[str, Dict[str, Any]] = {}
    secondary = secondary or {}
    secondary_drops: Dict[str, Dict[str, List[float]]] = {name: {} for name in secondary}
    errors: Dict[str, Exception] = {}
    for group_name, column_indices in groups.items():
        columns = np.asarray(column_indices, dtype=np.intp)
        values: List[float] = []
        for name in secondary_drops:
            secondary_drops[name][group_name] = []
        for _ in range(n_repeats):
            row_order = _shuffle_index(rng, len(X_values), *shuffle)
            X_permuted[:, columns] = X_values[np.ix_(row_order, columns)]
            X_model = _model_X(X_permuted, names)
            live = [name for name in secondary_drops if name not in errors]
            scored = _score_all(
                [scorer, *(secondary[name][0] for name in live)],
                model, X_model, y_values, weights, share,
            )
            permuted_score = _finite_float(_unwrap(scored[0]), "permuted score")
            values.append(
                _finite_float(baseline - permuted_score, "permutation importance")
            )
            for name, result in zip(live, scored[1:]):
                try:
                    score = _finite_float(_unwrap(result), "permuted score")
                    secondary_drops[name][group_name].append(
                        _finite_float(secondary[name][1] - score, "permutation importance")
                    )
                except Exception as error:
                    errors[name] = error
        X_permuted[:, columns] = X_values[:, columns]
        importances[group_name] = {
            "values": values,
            "mean": _finite_float(np.mean(values), "importance mean"),
            "std": _finite_float(np.std(values), "importance standard deviation"),
        }
    results: Dict[str, Any] = {}
    for name, by_group in secondary_drops.items():
        if name in errors:
            results[name] = errors[name]
            continue
        try:
            results[name] = {
                group: {
                    "mean": _finite_float(np.mean(drops), "importance mean"),
                    "std": _finite_float(np.std(drops), "importance standard deviation"),
                }
                for group, drops in by_group.items()
            }
        except Exception as error:
            results[name] = error
    return importances, results


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
    secondary: Optional[Dict[str, tuple[Any, float]]] = None,
    shuffle: tuple[str, Optional[np.ndarray]] = ("rows", None),
    share: bool = True,
) -> tuple[
    Dict[str, Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]
]:
    rng = np.random.default_rng(random_state)
    importances, secondary_results = _group_importances(
        model, scorer, X_values, y_values, groups, n_repeats, rng, baseline, names,
        weights, secondary, shuffle, share,
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
        shadow, _ = _group_importances(
            model, scorer, X_values, y_values, shadow_groups, n_repeats, rng, baseline,
            names, weights, None, shuffle,
        )
    return importances, shadow, secondary_results


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
    secondary_scorers: Sequence[tuple[Any, str]] = (),
    shuffle: tuple[str, Optional[np.ndarray]] = ("rows", None),
) -> tuple[
    str, Optional[Dict[str, Any]], Optional[Dict[str, str]], List[Dict[str, str]]
]:
    """Returns (name, report, ranking failure, evidence failures)."""
    try:
        X_model = _model_X(X_values, names)
        scorers = [scorer, *(other for other, _ in secondary_scorers)]
        scored = _score_all(scorers, model, X_model, y_values, weights, share=False)
        # Share predictions for this model only if the shared path reproduces
        # these direct baseline scores exactly (scikit-learn's private code may differ).
        share = _shared_matches(
            scored, _shared_scores(scorers, model, X_model, y_values, _score_weights(weights))
        )
        baseline = _finite_float(_unwrap(scored[0]), "baseline score")
        # A secondary scorer that fails is dropped (null) and reported as evidence.
        secondary: Dict[str, tuple[Any, float]] = {}
        secondary_errors: Dict[str, Exception] = {}
        for (other, other_name), result in zip(secondary_scorers, scored[1:]):
            try:
                secondary[other_name] = (
                    other, _finite_float(_unwrap(result), "baseline score")
                )
            except Exception as error:
                secondary_errors[other_name] = error
        importances, shadow, secondary_results = _permutation_importances(
            model, scorer, X_values, y_values, groups, n_repeats, random_state,
            baseline, shadow_groups, names, weights, secondary, shuffle, share,
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
        evidence_failures: List[Dict[str, str]] = []
        if secondary_scorers:
            secondary_errors.update(
                {n: r for n, r in secondary_results.items() if isinstance(r, Exception)}
            )
            report["secondary"] = {
                other_name: None
                if other_name in secondary_errors
                else {
                    "evaluation_score": secondary[other_name][1],
                    "importance": secondary_results[other_name],
                }
                for _, other_name in secondary_scorers
            }
            for other_name, other_error in secondary_errors.items():
                failure = _failure(name, "evidence", other_error)
                evidence_failures.append(
                    {**failure, "message": f"scorer {other_name}: {failure['message']}"}
                )
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
                evidence_failures.append(_failure(name, "evidence", error))
        return name, report, None, evidence_failures
    except Exception as error:
        return name, None, _failure(name, "ranking", error), []


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
        shuffle: Literal["rows", "within_groups", "circular"] = "rows",
        families: Optional[Mapping[str, str]] = None,
        max_categories: Optional[int] = None,
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
            shuffle: How permutations and shadow columns shuffle rows. "rows"
                (default) shuffles freely. "within_groups" moves values only
                among rows with the same label in shuffle_groups (passed to
                fit() for shadows and to rank_features() for permutations).
                "circular" shifts the rows cyclically by a random offset,
                which keeps autocorrelation and assumes rows in time order.
            families: Optional model name -> family mapping, for custom or
                built-in models. Each family casts one vote in the consensus,
                so map near-identical custom models to one family; unmapped
                custom models are separate families. A name that is not a
                configured model raises ValueError in fit().
            max_categories: None (default) or an int of at least 2. Built-in
                linear models one-hot encode at most this many levels per
                categorical column: the most frequent levels, with the rest
                sharing one infrequent column. A level unseen in training
                joins that column only when the column has at least
                max_categories levels; with fewer it is all zeros, as without
                the cap.
                Bounds the dense one-hot matrix, which grows with the number
                of levels; trees and custom models are unaffected.
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
        if not isinstance(shuffle, str) or shuffle not in {"rows", "within_groups", "circular"}:
            raise ValueError("[!] shuffle must be 'rows', 'within_groups', or 'circular'.")
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
        self.shuffle = shuffle
        self.families = _validate_families(families)
        self.max_categories = _validate_max_categories(max_categories)
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
        self.__dict__.setdefault("train_shuffle_groups_sha256_", None)
        # ... and those pickled by 1.14 lack the 1.15 shuffle mode.
        self.__dict__.setdefault("shuffle", "rows")
        # ... and those pickled by 1.17 lack the 1.18 family mapping.
        self.__dict__.setdefault("families", None)
        # ... and those pickled by 1.20 lack the 1.21 category cap and training subsample.
        self.__dict__.setdefault("max_categories", None)
        self.__dict__.setdefault("train_n_rows_", None)
        self.__dict__.setdefault("train_rows_", None)
        self.__dict__.setdefault("train_subsample_", {})
        # ... and those pickled by 1.16 were fingerprinted with scheme 1.
        self.__dict__.setdefault("fingerprint_scheme_", 1)
        if "model_families_" not in self.__dict__:
            self.model_families_ = (
                _registry_families() if self.user_models is None else {}
            )

    def _model_columns(self, names: List[str]) -> Optional[List[str]]:
        """Column names models see in "dataframe" mode (shadows last), else None."""
        if self.model_input != "dataframe":
            return None
        return names + [f"shadow__{name}" for name in names] if self.shadows else names

    def _check_family_names(self, configured: Any) -> None:
        """ValueError if `families` names a model that is not configured.

        Built-in models that failed to initialise still count as configured.
        """
        unknown = sorted(
            set(self.families or {})
            - set(configured)
            - {failure["model"] for failure in self.initialization_failures_}
        )
        if unknown:
            raise ValueError(
                f"[!] families names models that are not configured: {unknown}. "
                f"Configured: {sorted(configured)}."
            )

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
        sample_weight: Any = None,
        shuffle_groups: Any = None,
        max_samples: Any = None,
        random_state: Optional[int] = None,
    ) -> "FeatureRanker":
        """
        Fit configured models using training data only.

        max_samples: optional int (rows) or float in (0, 1] (share of rows).
        Fit on one random subsample of the training rows, drawn with
        random_state (None draws fresh entropy), which also bounds the
        shadows and the evidence copy; whole groups are drawn under
        shuffle="within_groups" and row order is kept. The run record's
        train_data_sha256 still fingerprints every row passed here.

        shuffle_groups: with shuffle="within_groups", one group label per
        training row; shadow columns are shuffled within each group. Not
        allowed with another shuffle.

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
        shuffle_labels = _shuffle_labels(
            self.shuffle, shuffle_groups, len(X_values), singletons_ok=not self.shadows
        )
        # Fingerprints cover every row passed, not only the rows fitted on.
        train_data_sha256 = _fingerprint(names, X_values, y_values)
        train_weights_sha256 = _weights_fingerprint(weights)
        train_shuffle_groups_sha256 = _labels_fingerprint(shuffle_labels)

        # Encode every classification target to 0..n-1 (XGBoost requires it).
        # Done on every row, so the subsample can be checked for a missing class.
        label_encoder = None
        if self.task == "clf":
            label_encoder = LabelEncoder()
            y_values = label_encoder.fit_transform(y_values)
        else:
            y_values = _float_target(y_values)

        train_rows: Optional[np.ndarray] = None
        if max_samples is not None:
            size = _resolve_max_samples(max_samples, len(X_values))
            sampler = np.random.default_rng(None if random_state is None else [random_state, 3])
            # Within groups, sample whole groups: a lone sampled row could never move.
            train_rows = (
                _sample_whole_groups(sampler, shuffle_labels, size)
                if shuffle_labels is not None
                else np.sort(sampler.choice(len(X_values), size, replace=False))
            )
            if weights is not None and weights[train_rows].sum() == 0:
                raise ValueError(
                    "[!] The rows drawn by max_samples all have zero sample_weight, "
                    "so nothing can be fitted; raise max_samples or give those rows weight."
                )
            if label_encoder is not None:
                absent = np.setdiff1d(np.arange(len(label_encoder.classes_)), y_values[train_rows])
                if len(absent):
                    raise ValueError(
                        f"[!] The rows drawn by max_samples miss the classes "
                        f"{label_encoder.classes_[absent].tolist()}, which no model could "
                        "learn; raise max_samples."
                    )
            X_values, y_values = X_values[train_rows], y_values[train_rows]
            weights = None if weights is None else weights[train_rows]
            if shuffle_labels is not None:
                shuffle_labels = shuffle_labels[train_rows]
            # Re-checked on the sample (a circular shift needs two rows).
            _shuffle_labels(
                self.shuffle, shuffle_labels, len(X_values), singletons_ok=not self.shadows
            )
        levels: Dict[str, List[Optional[str]]] = {}
        categorical_missing: Dict[str, int] = {}
        if categorical:
            categorical_missing = {
                names[index]: int(_is_missing(X_values[:, index]).sum())
                for index in categorical
            }
            levels = _learn_levels(X_values, categorical, names)
            X_values, _ = _encode_categorical(X_values, categorical, names, levels)

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
                        name: _with_one_hot(model, one_hot, self.max_categories)
                        for name, model in built.items()
                    }
                candidates.update(built)

        self._check_family_names(candidates)

        self.trained_models = {}
        self.fit_failures_ = []
        # Models see the shadow block too; the dummy below sees real columns only.
        X_models = (
            _with_shadows(
                X_values, self.shadow_seed, self.shadow_mode, self.shuffle, shuffle_labels
            )
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
        self.fingerprint_scheme_ = FINGERPRINT_SCHEME
        self.train_weights_sha256_ = train_weights_sha256
        self.train_shuffle_groups_sha256_ = train_shuffle_groups_sha256
        self.train_rows_ = train_rows
        self.train_n_rows_ = None if train_rows is None else len(train_rows)
        self.train_subsample_ = (
            {}
            if train_rows is None
            else {
                "train_rows_sha256": _labels_fingerprint(train_rows),
                "train_max_samples": _plain_number(max_samples),
                **({} if random_state is None else {"train_random_state": int(random_state)}),
            }
        )
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
        max_samples: Any = None,
        shuffle_groups: Any = None,
    ) -> Report:
        """
        Rank feature groups on explicitly supplied evaluation data.

        shuffle_groups: with shuffle="within_groups", one group label per
        evaluation row; permuted and shadow columns are shuffled within each
        group (after max_samples, the sampled rows' labels are used). Not
        allowed with another shuffle.

        sample_weight: optional 1-D non-negative weights, one per evaluation
        row; the baseline, permuted and dummy scores are weighted with them.

        max_samples: optional int (rows) or float in (0, 1] (share of rows).
        One random subsample, drawn with random_state, is used for the
        baseline, dummy and every permuted score; evidence is unchanged, except
        that overfit_gap compares the train score with the subsampled baseline.
        """
        if not self.is_fitted_:
            raise RuntimeError("[!] Call fit() before rank_features().")
        if not _is_positive_int(n_repeats):
            raise ValueError("[!] n_repeats must be a positive integer.")
        (scorer, scoring_name), *secondary_scorers = _resolve_scorers(scoring)

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
        shuffle_labels = _shuffle_labels(self.shuffle, shuffle_groups, len(X_values))
        rows: Any = slice(None)
        if max_samples is not None:
            size = _resolve_max_samples(max_samples, len(X_values))
            sampler = np.random.default_rng(None if random_state is None else [random_state, 2])
            # Within groups, sample whole groups: a lone sampled row could never move.
            rows = (
                _sample_whole_groups(sampler, shuffle_labels, size)
                if shuffle_labels is not None
                else np.sort(sampler.choice(len(X_values), size, replace=False))
            )
            if weights is not None and weights[rows].sum() == 0:
                raise ValueError(
                    "[!] The rows drawn by max_samples all have zero sample_weight, "
                    "so nothing can be scored; raise max_samples or give those rows weight."
                )
        scored_labels = _shuffle_labels(
            self.shuffle,
            None if shuffle_labels is None else shuffle_labels[rows],
            len(y_values[rows]),
        )
        run = {
            "task": self.task,
            "group": self.group,
            "fingerprint_scheme": self.fingerprint_scheme_,
            "data_sha256": _fingerprint(
                self.feature_names_, raw_values, y_values, self.fingerprint_scheme_
            ),
            "train_data_sha256": self.train_data_sha256_,
        }
        for key, value in (
            ("train_weights_sha256", self.train_weights_sha256_),
            ("weights_sha256", _weights_fingerprint(weights)),
            ("train_shuffle_groups_sha256", self.train_shuffle_groups_sha256_),
            (
                "shuffle_groups_sha256",
                _labels_fingerprint(shuffle_labels, self.fingerprint_scheme_),
            ),
            ("max_categories", self.max_categories),
            ("n_train_rows", self.train_n_rows_),
        ):
            if value is not None:
                run[key] = value
        run.update(self.train_subsample_)
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
        # Scores use the subsample; the shadows and the evidence see every row.
        y_scored = y_values[rows]
        weights_scored = None if weights is None else weights[rows]
        baseline_score = self._score_baseline(
            scorer, X_values[rows], y_scored, ranking_failures, weights_scored
        )
        # The dummy is not ranked, so a failure here is null, not a failure entry.
        secondary_baselines = {
            other_name: self._score_baseline(
                other, X_values[rows], y_scored, [], weights_scored
            )
            for other, other_name in secondary_scorers
        }
        X_models, shadow_groups = X_values, None
        if self.shadows:
            # An independent, reproducible shuffle of the evaluation rows.
            eval_seed = None if self.shadow_seed is None else [self.shadow_seed, 1]
            X_models = _with_shadows(
                X_values, eval_seed, self.shadow_mode, self.shuffle, shuffle_labels
            )
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
                status = _shadow_status(model, n_real, 2 * n_real)
                if status == SHADOWS_KEPT:
                    continue
                no_shadows.add(name)
                reason = (
                    "The model does not use every shadow column, so it is left out "
                    "of the shadow evidence."
                    if status == SHADOWS_DROPPED
                    else "Which columns the model uses could not be verified, so "
                    "it is left out of the shadow evidence."
                )
                shadow_failures.append(
                    _failure(name, "shadows", ShadowColumnsDropped(reason))
                )
        # Mapped families override the registry's; anything else is its own family.
        family_of = {**self.model_families_, **(self.families or {})}
        jobs = [
            (name, model, scorer, X_models[rows], y_scored, groups, n_repeats,
             random_state, baseline_score, family_of.get(name, name),
             None if name in no_shadows else shadow_groups, train_data, columns,
             weights_scored, secondary_scorers, (self.shuffle, scored_labels))
            for name, model in self.trained_models.items()
        ]
        evidence_failures: List[Dict[str, str]] = []
        for name, model_report, failure, model_evidence_failures in _run_models(
            _rank_model, jobs, self.n_jobs, "[*] Ranking models", self.verbose
        ):
            if failure is None:
                model_reports[name] = model_report
            else:
                ranking_failures.append(failure)
            evidence_failures.extend(model_evidence_failures)

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
        if secondary_scorers:
            report["secondary_scoring"] = [name for _, name in secondary_scorers]
            report["secondary_baseline_scores"] = secondary_baselines
        if self.model_input == "dataframe":
            report["model_input"] = "dataframe"
        if max_samples is not None:
            report["max_samples"] = _plain_number(max_samples)
            report["n_eval_rows"] = len(y_scored)
        if self.train_weighted_ or weights is not None:
            report["sample_weight"] = {
                "train": self.train_weighted_,
                "eval": weights is not None,
            }
        if self.shadows:
            report["shadows"] = True
            report["shadow_seed"] = self.shadow_seed
            report["shadow_mode"] = self.shadow_mode
        if self.shuffle != "rows":
            report["shuffle"] = self.shuffle
        if scored_labels is not None:
            constant = _constant_within_groups(X_values[rows], scored_labels, groups)
            report["constant_within_groups"] = constant
            if constant:
                warnings.warn(
                    f"Feature groups {constant} are constant within every shuffle "
                    "group, so shuffle='within_groups' cannot move them and their "
                    "importance is zero by construction. Use shuffle='rows' to "
                    "measure group-level effects.",
                    UserWarning,
                    stacklevel=2,
                )
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
        max_samples: Any = None,
        max_train_samples: Any = None,
    ) -> Report:
        """
        Fit and rank on every cv split of X/y and report rank stability.

        groups: also the group labels of shuffle="within_groups" (required
        then), sliced per split into training and evaluation labels.

        max_samples: as in rank_features(), applied to each split's evaluation
        rows.

        max_train_samples: as fit()'s max_samples, applied to each split's
        training rows and drawn with the split's random state.

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
        if max_samples is not None:
            _resolve_max_samples(max_samples, len(X_values))
        if max_train_samples is not None:
            _resolve_max_samples(max_train_samples, len(X_values))
        _resolve_scorers(scoring)
        _resolve_feature_groups(names, feature_groups)
        # Each split builds its own ranker; check the names once, up front.
        self._check_family_names(
            self.user_models
            if self.user_models is not None
            else {
                name
                for group in (
                    ["linear", "tree"] if self.group in {"all", "fast"} else [self.group]
                )
                for name in self.models.get(self.task, {}).get(group, {})
            }
        )
        shuffle_labels = _shuffle_labels(
            self.shuffle,
            groups if self.shuffle == "within_groups" else None,
            len(X_values),
            "groups",
        )
        splits = _resolve_splits(
            cv, X_values, y_values, groups, self.task, random_state
        )
        if self.shuffle == "circular":
            # Splitters may give unordered indices; a cyclic shift needs time order.
            splits = [(np.sort(train), np.sort(test)) for train, test in splits]

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
                    shuffle=self.shuffle,
                    families=self.families,
                    max_categories=self.max_categories,
                    # Names, so "auto" survives the loss of the DataFrame dtypes.
                    categorical=[names[column] for column in categorical] or None,
                )
                ranker.fit(
                    X_values[train],
                    y_values[train],
                    feature_names=names,
                    sample_weight=None if weights is None else weights[train],
                    shuffle_groups=None if shuffle_labels is None else shuffle_labels[train],
                    max_samples=max_train_samples,
                    random_state=None if random_state is None else random_state + index,
                )
                report = ranker.rank_features(
                    X_values[evaluation],
                    y_values[evaluation],
                    scoring=scoring,
                    feature_groups=feature_groups,
                    n_repeats=n_repeats,
                    random_state=None if random_state is None else random_state + index,
                    sample_weight=None if weights is None else weights[evaluation],
                    max_samples=max_samples,
                    shuffle_groups=(
                        None if shuffle_labels is None else shuffle_labels[evaluation]
                    ),
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
            "fingerprint_scheme": FINGERPRINT_SCHEME,
            "data_sha256": _fingerprint(names, X_values, y_values),
            "cv": _cv_record(cv),
            "splits_sha256": _splits_fingerprint(splits),
        }
        if weights is not None:
            run["weights_sha256"] = _weights_fingerprint(weights)
        if shuffle_labels is not None:
            run["shuffle_groups_sha256"] = _labels_fingerprint(shuffle_labels)
        if self.max_categories is not None:
            run["max_categories"] = self.max_categories
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
        if "secondary_scoring" in first:
            result["secondary_scoring"] = first["secondary_scoring"]
        if self.model_input == "dataframe":
            result["model_input"] = "dataframe"
        if max_samples is not None:
            result["max_samples"] = _plain_number(max_samples)
        if max_train_samples is not None:
            result["max_train_samples"] = _plain_number(max_train_samples)
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
        if self.shuffle != "rows":
            result["shuffle"] = self.shuffle
        if self.shuffle == "within_groups":
            result["constant_within_groups"] = sorted(
                {
                    group
                    for entry in split_reports
                    for group in entry["report"]["constant_within_groups"]
                }
            )
        return Report(result)

    def ablate(
        self,
        X: Any,
        y: Any,
        X_eval: Any,
        y_eval: Any,
        report: Report,
        *,
        ks: Optional[Sequence[int]] = None,
        scoring: Any = None,
        sample_weight: Any = None,
        eval_sample_weight: Any = None,
    ) -> Report:
        """
        A copy of a held-out report with an "ablation" block: for each k, the
        report's voting models are refit on the columns of the top-k consensus
        groups and scored on X_eval.

        X and y must be the training data this ranker was fit on (checked
        against the report's train_data_sha256 when it has one). ks defaults
        to 1, 2, 3, 5, 10, 20, 50, ... up to the number of consensus groups,
        plus that number. scoring is a scorer name or callable and defaults to
        the report's scorer name. sample_weight and eval_sample_weight weight
        the refits and the scores. Raises ValueError for an unfitted ranker, a
        resampled report, an empty consensus, bad ks, or other training data.
        The report is not modified. Costs len(ks) refits of every voting model.
        """
        from .ablation import ablate

        return ablate(
            self, X, y, X_eval, y_eval, report, ks=ks, scoring=scoring,
            sample_weight=sample_weight, eval_sample_weight=eval_sample_weight,
        )

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
