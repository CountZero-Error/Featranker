"""
Feature importance ranking via permutation importance across multiple ML models.
"""

import datetime
from collections import Counter
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import numbers
import platform
import re
import types
import warnings
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Sequence, Union

import numpy as np
import sklearn
import yaml
from joblib import Parallel, delayed
import scipy.sparse
from scipy.stats import binom
from sklearn.base import clone, is_regressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
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

from .evidence import _FeatRankerOneHot, _model_X, _read_only, correlation_evidence, shap_evidence
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
        raw = X if isinstance(X, np.ndarray) else np.asarray(X, dtype=object)
        if raw.dtype.kind in {"M", "m"} or (raw.dtype.kind == "O" and _has_dates(raw.flat)):
            raise ValueError(NOT_NUMERIC)
        values = np.array(raw, dtype=target_dtype)
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
        not _is_int(max_categories)
        or max_categories < 2
        or max_categories >= MAX_COUNT
    ):
        raise ValueError("[!] max_categories must be None or an int of at least 2 and below 2**63.")
    return int(max_categories)


# ponytail: add any new built-in linear model that rejects sparse input here.
DENSE_ONLY_ESTIMATORS = frozenset(
    {
        "LinearDiscriminantAnalysis",
        "QuadraticDiscriminantAnalysis",
        "GaussianNB",
        "PLSRegression",
        "SGDClassifier",  # SGD intercept update differs on sparse input
        "Perceptron",
    }
)
_FORCE_DENSE_ONE_HOT = False  # tests flip this to compare against dense output


def _with_one_hot(
    pipeline: Any, columns: List[int], max_categories: Optional[int] = None, dense: bool = False
) -> Any:
    """
    A built-in linear pipeline whose numeric steps skip `columns`, which are one-hot coded.

    With `max_categories`, each column keeps at most that many one-hot columns
    (the infrequent levels share one). Unseen levels join it only for a column
    with at least `max_categories` levels; with fewer they are all zeros.
    """
    *numeric_steps, (_, estimator) = pipeline.steps
    dense = (
        dense
        or _FORCE_DENSE_ONE_HOT
        or type(estimator).__name__ in DENSE_ONLY_ESTIMATORS
        # sparse input made saga stop early
        or getattr(estimator, "solver", None) in ("sag", "saga")
    )
    capped = (
        {}
        if max_categories is None
        else {"max_categories": max_categories, "handle_unknown": "infrequent_if_exist"}
    )
    return make_pipeline(
        ColumnTransformer(
            [("cat", _FeatRankerOneHot(**{"handle_unknown": "ignore", **capped}), columns)],
            remainder=make_pipeline(*(step for _, step in numeric_steps)),
            # Keep sparse output unless the estimator rejects sparse input.
            sparse_threshold=0 if dense else 1.0,
        ),
        estimator,
    )


def _validate_n_jobs(n_jobs: Any) -> None:
    if n_jobs is not None and (not _is_int(n_jobs) or n_jobs == 0):
        raise ValueError("[!] n_jobs must be None or a non-zero int.")


def _check_index(X: Any, value: Any, name: str) -> None:
    """Refuse or flag a pandas `value` whose index does not match the DataFrame X.

    Values are used by position, so a reordered Series would silently pair
    with the wrong rows.
    """
    if _pd is None or not isinstance(X, _pd.DataFrame):
        return
    if not isinstance(value, (_pd.Series, _pd.DataFrame)) or value.index.equals(X.index):
        return
    if len(value) == len(X) and Counter(value.index) == Counter(X.index):
        raise ValueError(
            f"[!] {name} has the same index labels as X in another order; it would be "
            f"used by position. Align it with {name}.loc[X.index] or pass {name}.to_numpy()."
        )
    warnings.warn(
        f"{name} has a different index from X; it is used by position, not by index label.",
        UserWarning,
        stacklevel=3,
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


MAX_COUNT = 2**63  # counts must stay serialisable in reports
N_REPEATS_MESSAGE = "[!] n_repeats must be a positive integer below 2**63."


def _is_int(value: Any) -> bool:
    """An int or NumPy integer; never a bool or a NumPy duration (which counts as an integer)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.timedelta64))


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.timedelta64))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value < MAX_COUNT


def _resolve_max_samples(value: Any, n_rows: int) -> int:
    """Rows to score: a positive int (capped at n_rows) or a float share in (0, 1]."""
    if not _is_real(value):
        raise ValueError("[!] max_samples must be a positive int or a float in (0, 1].")
    if _is_int(value) and 1 <= int(value) < MAX_COUNT:
        return min(int(value), n_rows)
    if not _is_int(value) and 0 < value <= 1:  # no float(): a huge Fraction overflows
        return max(1, int(float(value) * n_rows))
    raise ValueError("[!] max_samples must be a positive int or a float in (0, 1].")


def _plain_int(value: Any) -> Any:
    """A NumPy integer as a plain int (reports stay strict JSON); anything else, bools too, as is."""
    return int(value) if _is_int(value) else value


def _seed(value: Any, name: str) -> Optional[int]:
    """None, or an int (NumPy integers too, bools never) in 0..2**64 - 1, as a plain int."""
    if value is None:
        return None
    if not _is_int(value):
        raise TypeError(f"[!] {name} must be an int or None.")
    value = int(value)
    if not 0 <= value < 2**64:
        raise ValueError(f"[!] {name} must be non-negative and below 2**64.")
    return value


def _plain_number(value: Any) -> Any:
    """A numpy scalar as a plain int or float, so the report stays JSON-safe."""
    return int(value) if _is_int(value) else float(value)


def _resolve_scorer(scoring: Any) -> tuple[Any, str]:
    if isinstance(scoring, str):
        return get_scorer(scoring), scoring
    if callable(scoring):
        # make_scorer objects have no __name__; their score function does.
        # Only exact scikit-learn scorers: any other object's _score_func means nothing.
        is_scorer = type(scoring) is type(get_scorer("r2"))
        named = getattr(scoring, "_score_func", scoring) if is_scorer else scoring
        name = getattr(named, "__name__", None)
        return scoring, name if isinstance(name, str) and name else type(scoring).__name__
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


def _components(model: Any) -> Dict[str, str]:
    """Class name of each estimator-valued param (not a class), by param name."""
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    return {
        key: type(value).__name__
        for key, value in params.items()
        if hasattr(value, "get_params") and not isinstance(value, type)
    }


def _python_scalar(value: Any) -> Any:
    """bool, int, float and str values (subclasses, enums and NumPy scalars, not timedelta64) as the plain type."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.timedelta64):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)  # a str Enum gives its value
    return value


def _params_equal(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """Same keys, and per key the same plain type and value."""
    if first.keys() != second.keys():
        return False
    for key, value in first.items():
        one, other = _python_scalar(value), _python_scalar(second[key])
        if type(one) is not type(other) or one != other:
            return False
    return True


def _json_params(model: Any, *, legacy: bool = False) -> Dict[str, Any]:
    """
    Estimator params restricted to JSON-safe scalars (finite if float), as plain Python types.

    With legacy=True (reports before schema 2), only str, int, bool, None and finite float instances were
    recorded; NumPy scalars of other types were left out.
    """
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    if not legacy:
        params = {key: _python_scalar(value) for key, value in params.items()}
    params = {
        key: value
        for key, value in params.items()
        if value is None
        or isinstance(value, (str, int, bool))
        or (isinstance(value, float) and math.isfinite(value))
    }
    return {key: _python_scalar(value) for key, value in params.items()}


MAX_DESCRIBE_DEPTH = 20


def _module_name(name: Any) -> Any:
    """A spawned process runs the main script as __mp_main__; it is the same __main__ to a report."""
    return "__main__" if name == "__mp_main__" else name


def _type_name(value: Any) -> str:
    return f"{_module_name(type(value).__module__)}.{type(value).__qualname__}"


def _array_text(array: Any, inner: int, path: FrozenSet[int]) -> str:
    """dtype (native byte order), shape and bytes (C order) of any array subclass; object arrays as lists."""
    array = np.asarray(array)
    if array.dtype == object:
        return f"ndarray{array.shape}{_describe(array.tolist(), inner, path)}"
    if array.dtype.names:  # one text per field: the padding bytes of a record are not values
        fields = ",".join(f"{name}={_array_text(array[name], inner, path)}" for name in array.dtype.names)
        return f"ndarray{array.shape}{{{fields}}}"
    data = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("=")))
    return f"ndarray{data.dtype}{array.shape}{hashlib.sha256(data.tobytes()).hexdigest()}"


def _describe(value: Any, depth: int = 0, path: FrozenSet[int] = frozenset()) -> str:
    """
    A canonical text of a parameter value, equal for equal values and free of addresses.

    Strange values (failing __repr__ or __eq__, cycles, huge ints) and values nested deeper than
    MAX_DESCRIBE_DEPTH, or inside themselves, are described by their type alone.
    """
    kind = _type_name(value)
    if depth > MAX_DESCRIBE_DEPTH or id(value) in path:
        return kind
    try:
        inner, path = depth + 1, path | {id(value)}
        value = _python_scalar(value)
        if value is None or isinstance(value, (bool, int, float, str, bytes, complex, range)):
            return f"{_type_name(value)}:{value!r}"
        if isinstance(value, np.void):  # a record: a 0-d structured array; bytes if unstructured
            return _array_text(np.asarray(value), inner, path) if value.dtype.names else f"{kind}:{value.tobytes()!r}"
        if isinstance(value, np.generic):
            return f"{kind}:{value}"  # str, not repr: NumPy 1 and 2 differ
        if isinstance(value, slice):
            parts = (value.start, value.stop, value.step)
            return f"{kind}[{','.join(_describe(part, inner, path) for part in parts)}]"
        if isinstance(value, np.random.RandomState):
            return f"{kind}[{_describe(value.get_state(), inner, path)}]"
        if isinstance(value, np.random.Generator):
            return f"{kind}[{_describe(value.bit_generator.state, inner, path)}]"
        if isinstance(value, (list, tuple)):
            return f"{kind}[{','.join(_describe(item, inner, path) for item in value)}]"
        if isinstance(value, (set, frozenset)):
            return f"{kind}{{{','.join(sorted(_describe(item, inner, path) for item in value))}}}"
        if isinstance(value, dict):
            pairs = ((_describe(k, inner, path), _describe(v, inner, path)) for k, v in value.items())
            return f"dict{{{','.join(sorted(f'{k}:{v}' for k, v in pairs))}}}"
        if isinstance(value, np.ndarray):
            text = _array_text(value, inner, path)
            if np.ma.isMaskedArray(value):
                fill = getattr(value, "fill_value")
                text += _array_text(np.ma.getmaskarray(value), inner, path) + _array_text(fill, inner, path)
            return text
        if scipy.sparse.issparse(value):
            matrix = value.tocsr(copy=True) if value.ndim == 2 else value.tocoo(copy=True)
            matrix.sum_duplicates()
            parts = (matrix.data, matrix.indices, matrix.indptr) if value.ndim == 2 else (
                matrix.data, *matrix.coords)
            return f"sparse{matrix.shape}{','.join(_array_text(part, inner, path) for part in parts)}"
        if kind.startswith("pandas."):
            is_index = kind.endswith("Index")
            names = list(value.columns) if hasattr(value, "columns") else getattr(value, "name", None)
            labels = (value.names, None) if is_index else (names, getattr(value, "index", None))
            return f"{kind}[{_describe(value.to_numpy(), inner, path)},{_describe(labels, inner, path)}]"
        if isinstance(value, functools.partial):
            parts = (value.func, value.args, value.keywords)
            return f"{kind}[{','.join(_describe(part, inner, path) for part in parts)}]"
        if depth and hasattr(value, "get_params") and not isinstance(value, type):
            return f"{kind}[{_describe(value.get_params(deep=False), inner, path)}]"
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if callable(value) and name and (isinstance(value, type) or not hasattr(value, "get_params")):
            # np.ufunc names vary in module; method descriptors only know their class.
            owner_class = getattr(value, "__objclass__", None)
            home = _module_name(getattr(value, "__module__", None) or getattr(owner_class, "__module__", None))
            module = "" if isinstance(value, np.ufunc) else home
            owner = getattr(value, "__self__", None)
            bound = "" if owner is None or isinstance(owner, (types.ModuleType, type)) else _describe(
                owner, inner, path)
            return f"callable:{module}.{name}{bound}"
    except Exception:
        pass
    return kind


def _param_hashes(model: Any) -> Dict[str, str]:
    """sha256 of the described value of each get_params(deep=True) parameter that _json_params leaves out."""
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    held = _json_params(model)
    # An estimator that is itself a deep parameter is described by its type: its params are keys of their own.
    path = frozenset(id(v) for v in params.values() if hasattr(v, "get_params") and not isinstance(v, type))
    return {
        key: hashlib.sha256(_describe(value, 0, path).encode("utf-8", "backslashreplace")).hexdigest()
        for key, value in params.items()
        if key not in held
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
# Stamped on every report; see report.schema.json.
REPORT_SCHEMA_VERSION = 2
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
    if _is_int(cv):
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
        or values.dtype.kind not in "iu"  # not a duration, which NumPy counts as an integer
        or values.min() < 0
        or values.max() >= n_rows
    ):
        raise ValueError(
            "[!] Each split needs non-empty integer train and eval indices "
            f"within 0..{n_rows - 1}."
        )
    return values.astype(int)


def _check_cv(cv: Any) -> None:
    """Raise unless cv is an int >= 2, a splitter or an iterable; consumes nothing."""
    if _is_int(cv):
        if cv < 2:
            raise ValueError(CV_ERROR)
    elif isinstance(cv, (str, bytes, bool)) or not (
        callable(getattr(cv, "split", None)) or hasattr(cv, "__iter__")
    ):  # str has .split(); reject it first
        raise ValueError(CV_ERROR)


def _resolve_splits(
    cv: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Any,
    task: str,
    random_state: Optional[int],
) -> List[tuple]:
    """Materialize cv into validated (train, eval) index pairs before any fit."""
    _check_cv(cv)
    split_y = y_values
    if _is_int(cv):
        if groups is not None:
            warnings.warn(
                "groups is not used for splitting by an integer cv (shuffled KFold or "
                "StratifiedKFold); pass a group splitter such as GroupKFold to keep "
                "groups together.",
                UserWarning,
                stacklevel=3,
            )
        # Shuffled folds: unshuffled KFold on sorted data gives misleading splits.
        splitter = StratifiedKFold if task == "clf" else KFold
        # sklearn uses RandomState. Keep historical folds for 32-bit seeds;
        # mix both halves of wider seeds for a reproducible splitter seed.
        fold_seed = random_state
        if random_state is not None and random_state >= 2**32:
            fold_seed = int(np.random.SeedSequence(random_state).generate_state(1)[0])
        cv = splitter(n_splits=int(cv), shuffle=True, random_state=fold_seed)
        if task == "clf":
            # Stratify on class codes: fit() accepts any labels (even 0.1/0.2),
            # but StratifiedKFold rejects numeric labels it reads as continuous.
            # Only the fold assignment sees these codes; each split still fits
            # its own encoder on its training rows.
            split_y = LabelEncoder().fit_transform(y_values)
    if callable(getattr(cv, "split", None)):
        pairs = cv.split(X_values, split_y, groups)
    else:
        pairs = cv

    resolved: List[tuple] = []
    seen: Dict[bytes, int] = {}
    for pair in pairs:
        try:
            train, evaluation = pair
        except (TypeError, ValueError) as error:
            raise ValueError(CV_ERROR) from error
        n_rows = len(X_values)
        train, evaluation = _split_indices(train, n_rows), _split_indices(evaluation, n_rows)
        if np.intersect1d(train, evaluation).size:
            raise ValueError("[!] Train and eval indices overlap within a split.")
        # Digests, not the index bytes: a million-row cv would keep every split twice.
        key = hashlib.sha256(
            np.sort(train).tobytes() + b"|" + np.sort(evaluation).tobytes()
        ).digest()
        if key in seen:
            raise ValueError(
                f"[!] Splits {seen[key]} and {len(resolved)} have identical train and "
                "eval indices; a repeated split would count as an independent trial. "
                "Small data with a repeated splitter can repeat a split by chance; "
                "use a splitter without repeats or more rows."
            )
        seen[key] = len(resolved)
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

    rows: List[Dict[str, Any]] = []
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


def _python_group_label(value: Any) -> Any:
    """Compare mixed numeric labels without NumPy rounding integers to floats."""
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError("[!] Complex object labels are not sortable.")
    if isinstance(value, np.number) and not isinstance(value, np.timedelta64):
        scalar = value.item()
        if isinstance(scalar, np.floating):  # longdouble can exceed Python float precision
            if np.isfinite(scalar):
                return Fraction(*scalar.as_integer_ratio())
            return float(scalar)
        return scalar
    return value


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
    if values.ndim == 1 and values.size and not hasattr(labels, "dtype"):
        # A Python sequence can lose label identity when NumPy infers a
        # common string, floating or complex dtype. Keep heterogeneous values
        # as Python objects so validation and permutation use their own order.
        first = next(iter(labels))
        # Fixed-width strings use trailing NULs as padding.
        has_trailing_nul = values.dtype.kind in "US" and any(
            (isinstance(item, str) and item.endswith("\0"))
            or (isinstance(item, bytes) and item.endswith(b"\0"))
            for item in labels
        )
        if (any(type(item) is not type(first) for item in labels)
                or (_is_int(first) and values.dtype.kind == "f")
                or has_trailing_nul):
            values = np.asarray(labels, dtype=object)
    if values.shape != (n_rows,):
        raise ValueError(
            f"[!] {name} needs one label per row: {n_rows} rows, got shape {values.shape}."
        )
    if values.dtype.kind == "O":
        values = np.fromiter((_python_group_label(v) for v in values), dtype=object, count=n_rows)
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
    order = rng.permutation(int(np.max(inverse)) + 1)
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


def _is_column_aware(estimator: Any, *, transforming: bool = False) -> bool:
    """
    Whether the estimator is, or contains at any depth, a selector, FeatureUnion,
    ColumnTransformer or opaque transformer (see _is_opaque_transformer). A target transformer of a
    TransformedTargetRegressor is not counted.
    """
    if isinstance(estimator, (SelectorMixin, FeatureUnion, ColumnTransformer)):
        return True
    if _is_opaque_transformer(estimator, transforming=transforming):
        return True
    if isinstance(estimator, Pipeline):
        last = len(estimator.steps) - 1
        return any(
            _is_column_aware(step, transforming=transforming or index < last)
            for index, (_, step) in enumerate(estimator.steps)
        )
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV)):
        selected = getattr(estimator, "best_estimator_", estimator.estimator)
        return _is_column_aware(selected, transforming=transforming)
    if isinstance(estimator, TransformedTargetRegressor):
        # Only the regressor sees X. The fitted transformer sees y.
        return _is_column_aware(getattr(estimator, "regressor_", estimator.regressor))
    if isinstance(estimator, (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)):
        # The fitted members can contain a different preprocessing pipeline
        # from their templates, for example after an inner parameter search.
        members = getattr(estimator, "estimators_", estimator.estimators)
        if isinstance(estimator, (StackingClassifier, StackingRegressor)):
            # The final estimator can discard member predictions or,
            # with passthrough, original input columns.
            final = getattr(estimator, "final_estimator_", estimator.final_estimator)
            if _is_column_aware(final):
                return True
        return any(_is_column_aware(member) for member in members)
    if isinstance(estimator, Mapping):
        return any(_is_column_aware(value, transforming=transforming) for value in estimator.values())
    if isinstance(estimator, (list, tuple)):
        return any(_is_column_aware(value, transforming=transforming) for value in estimator)
    fitted = [*getattr(estimator, "estimators_", ()), *([estimator.estimator_] if hasattr(estimator, "estimator_") else [])]
    if any(_is_column_aware(member) for member in fitted):
        return True
    if hasattr(estimator, "get_params"):
        return any(
            _is_column_aware(value, transforming=transforming)
            for value in estimator.get_params(deep=False).values()
        )
    return False


def _step_used(step: Any, columns: List[int]) -> set:
    """The columns of `columns` a ColumnTransformer or FeatureUnion part uses."""
    if _is_string(step, "drop"):
        return set()
    if _is_string(step, "passthrough"):
        return set(columns)
    return _kept_inputs(step, columns, transforming=True)


def _pipeline_kept(pipeline: Pipeline, columns: List[int], transforming: bool) -> set:
    """
    The columns of a Pipeline's first step, narrowed by later steps that still see them.

    A later step that is or contains a selector, FeatureUnion, ColumnTransformer or
    opaque transformer is mapped only while every earlier step is known to keep the columns one for
    one; after any other step it cannot be mapped, so the result is unknown (raises).
    """
    steps = [
        step for _, step in pipeline.steps
        if step is not None and not _is_string(step, "passthrough")
    ]
    kept = set(columns)
    preserved = True
    for index, step in enumerate(steps):
        transforms = transforming or index < len(steps) - 1
        if preserved:
            kept &= _kept_inputs(step, columns, transforming=transforms)
        elif _is_column_aware(step, transforming=transforms):
            raise ValueError("a column-aware or opaque step follows a step that may change the columns")
        preserved = preserved and _keeps_columns(step)
    return kept


def _is_opaque_transformer(estimator: Any, *, transforming: bool = False) -> bool:
    """A transformer whose column use is unknown in the role where it is used."""
    if not hasattr(estimator, "transform") or (not transforming and hasattr(estimator, "predict")):
        return False
    if type(estimator) is FunctionTransformer:
        # func=None is the identity: a fitted ColumnTransformer stores "passthrough" this way.
        return estimator.func is not None
    # FeatRanker's own one-hot encoder subclass lives outside sklearn but keeps every column.
    return not (
        type(estimator).__module__.startswith("sklearn.") or type(estimator) is _FeatRankerOneHot
    )


def _kept_inputs(estimator: Any, columns: List[int], *, transforming: bool = False) -> set:
    """
    The model-input columns `estimator` uses, given that it receives `columns`.

    Looks into ColumnTransformer branches, FeatureUnion parts, fitted feature
    selectors, and a Pipeline's steps (see _pipeline_kept). Any other estimator
    is assumed to use every column, unless it is or wraps (a meta-estimator) a
    selector, FeatureUnion, ColumnTransformer or opaque transformer (see
    _is_opaque_transformer: it may drop columns; final predictors are not
    affected), which cannot be followed. Raises when the columns
    cannot be determined; the caller treats that as unverifiable.
    """
    if _is_opaque_transformer(estimator, transforming=transforming):
        raise ValueError("an opaque transformer may drop columns")
    if isinstance(estimator, Pipeline):
        return _pipeline_kept(estimator, columns, transforming)
    if isinstance(estimator, FeatureUnion):
        return set().union(*(_step_used(part, columns) for _, part in estimator.transformer_list))
    if isinstance(estimator, SelectorMixin):
        return {columns[i] for i in estimator.get_support(indices=True)}
    if not isinstance(estimator, ColumnTransformer):
        if _is_column_aware(estimator, transforming=transforming):
            raise ValueError(
                "an opaque transformer, or a meta-estimator wrapping a selector, "
                "FeatureUnion, ColumnTransformer or opaque transformer"
            )
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
            fit_params[key] = _read_only(weights)
        # Read-only (X via _model_X), so a model that writes to its input fails
        # alone instead of corrupting the arrays every later model fits on.
        fitted = clone(model, safe=False).fit(
            _model_X(X_values, names), _read_only(y_values), **fit_params
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
        # Scoring sees read-only views too, as fitting does (X via _model_X).
        y_values, weights = _read_only(y_values), _read_only(weights)
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
            "class": type(model).__name__,
            "qualified_class": _type_name(model),
            "evaluation_score": baseline,
            "in_consensus": baseline_score is None or baseline > baseline_score,
            "params": _json_params(model),
            "components": _components(model),
            "param_hashes": _param_hashes(model),
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
                        model, _model_X(train_X, names), _read_only(train_y),
                        **_score_weights(_read_only(train_weights)),
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
    """
    Rank feature groups by permutation importance across many models.

    Fit a set of models on training data with ``fit()``, then call
    ``rank_features()`` on separate evaluation rows. Each model's score drop
    when a feature group is shuffled is turned into a rank; models that beat a
    dummy baseline vote, one vote per model family, and the votes give a
    consensus ranking. Keeping the training and evaluation rows apart is up to
    you; FeatureRanker cannot detect reuse.

    Parameters
    ----------
    task : {"clf", "reg"}
        ``"clf"`` for classification, ``"reg"`` for regression. Required.
    group : {"all", "linear", "tree", "fast"}, default "all"
        Built-in model group. ``"fast"`` runs one model per family. Cannot be
        combined with ``models``.
    models : dict, optional
        ``{name: estimator}`` that replaces the built-in models. Estimators
        are cloned at ``fit()``. A name must be a non-empty string other than
        ``"dummy_baseline"``.
    n_jobs : int, optional
        Parallel workers over models. ``None`` or 1 is sequential, -1 uses all
        cores.
    verbose : bool, default True
        Show progress bars. Warnings are always shown.
    shadows : bool, default False
        Also train and rank a shuffled shadow copy of every feature, to show
        which groups beat noise.
    shadow_seed : int, optional
        Seed (0 to 2**64 - 1) of the shadow shuffles. ``None`` gives different
        shadows on every run.
    evidence : bool, default False
        Keep a copy of the training data and add train scores, overfit gaps,
        correlated groups and (with ``featranker[shap]``) a SHAP summary to
        the report.
    shadow_mode : {"joint", "independent"}, default "joint"
        ``"joint"`` shuffles all shadow columns with one shared row order;
        ``"independent"`` shuffles each column on its own. Used only with
        ``shadows=True``.
    missing : {"error", "impute"}, default "error"
        ``"error"`` rejects NaN in the features. ``"impute"`` accepts NaN and
        gives every built-in model a median imputer fit on its training rows
        only; custom models receive NaN unchanged. Infinite values are always
        rejected.
    categorical : "auto" or list of str, optional
        ``None`` accepts numeric features only. ``"auto"`` treats DataFrame
        columns of object, string or category dtype as categorical; a list
        names the columns. Levels are learned in each ``fit()`` from the
        training rows only, and each column stays one feature.
    model_input : {"array", "dataframe"}, default "array"
        ``"array"`` gives models a float NumPy array. ``"dataframe"`` gives
        every model a pandas DataFrame with the feature names (and
        ``shadow__<name>`` columns with shadows). Needs pandas.
    shuffle : {"rows", "within_groups", "circular"}, default "rows"
        How permutations and shadows shuffle rows: freely, only among rows
        with the same label in ``shuffle_groups``, or by a cyclic shift (rows
        in time order).
    families : dict, optional
        ``{model_name: family}`` that sets the consensus family of custom or
        built-in models. Models in one family cast one vote. Unmapped custom
        models are separate families. A name that is not a configured model
        raises ``ValueError`` in ``fit()``.
    max_categories : int, optional
        At least 2. Built-in linear models one-hot encode at most this many
        levels per categorical column; the rest share one column. Trees and
        custom models are unaffected.

    Attributes
    ----------
    trained_models : dict
        Fitted models by name, after ``fit()``.
    feature_names_ : list of str
        Feature names seen by ``fit()``.
    label_encoder : LabelEncoder or None
        For ``task="clf"``: ``label_encoder.classes_[i]`` is the label of code ``i``.
    categorical_features_, categorical_levels_ : list, dict
        Categorical columns and their learned levels.
    fit_failures_, initialization_failures_ : list of dict
        Models that failed to fit or to initialize.

    Raises
    ------
    ValueError
        For an invalid option, such as an unknown ``task`` or ``group``.
    TypeError
        For an option of the wrong type, such as a non-bool ``verbose``.
    ImportError
        If ``model_input="dataframe"`` and pandas is missing.

    See Also
    --------
    ConsensusSelector : the same ranking as a scikit-learn pipeline step.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from featranker import FeatureRanker
    >>> data = load_diabetes()
    >>> X_train, X_eval, y_train, y_eval = train_test_split(
    ...     data.data, data.target, test_size=0.3, random_state=0)
    >>> ranker = FeatureRanker(task="reg", group="fast", verbose=False)
    >>> ranker = ranker.fit(X_train, y_train, feature_names=list(data.feature_names))
    >>> report = ranker.rank_features(
    ...     X_eval, y_eval, scoring="r2", n_repeats=3, random_state=0)
    >>> [row["feature_group"] for row in report["consensus"]][:2]  # doctest: +SKIP
    ['s5', 'bmi']
    """

    # Names of built-in models one-hot coded densely (ablate() on pre-2.0 reports sets it on its copy).
    _dense_one_hot: FrozenSet[str] = frozenset()
    # missing, model_input and max_categories as of the last successful fit() (None: not
    # recorded, as in a pickle from before 2.0; ablate() then uses the current attributes).
    _fit_settings: Optional[Dict[str, Any]] = None
    # `missing` as of the last init_models() (None: not recorded; the built-in pipelines carry an imputer or not).
    _built_missing: Optional[str] = None

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
        """Create a ranker and set up its models; no data is read here."""
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
        shadow_seed = _seed(shadow_seed, "shadow_seed")
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
        _validate_n_jobs(n_jobs)
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
        # Unfitted clones of the templates the last fit() used (what ablate() refits).
        self.fit_models_: Dict[str, Any] = {}
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
        self.shadow_seed = _plain_int(self.shadow_seed)  # NumPy ints made reports non-JSON
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
        # ... and those pickled before 2.0 lack the frozen refit configuration.
        self.__dict__.setdefault("fit_models_", {})
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
        Fit every configured model on the training data.

        Parameters
        ----------
        X : array-like or DataFrame
            Training features, at least one row and one column. Numeric, plus
            the columns named by ``categorical``. Infinite values are
            rejected, and so is NaN unless ``missing="impute"``.
        y : array-like
            Training target, one value per row. Classification labels may be
            numbers or strings; regression targets must be numeric.
        feature_names : list of str, optional
            Unique, non-empty column names. Required for NumPy input; taken
            from the columns of a DataFrame.
        sample_weight : array-like, optional
            Finite, non-negative training weights, one per row, with a
            positive sum. Every model is fit with them. A model that cannot
            take weights is recorded as a fit failure, not fit unweighted.
        shuffle_groups : array-like, optional
            One group label per training row. Required with
            ``shuffle="within_groups"`` (shadows are shuffled within groups);
            an error with another ``shuffle``.
        max_samples : int or float, optional
            Fit on a random subsample of the training rows: an int (rows) or a
            float in (0, 1] (share of rows). Whole groups are drawn under
            ``shuffle="within_groups"``; row order is kept. Shadows and the
            evidence copy use the sample too.
        random_state : int, optional
            Seed (0 to 2**64 - 1) of the ``max_samples`` draw. ``None`` draws
            fresh entropy. Unused without ``max_samples``.

        Returns
        -------
        FeatureRanker
            The ranker itself, fitted.

        Raises
        ------
        ValueError
            For invalid data or options, or a subsample that misses a class
            of a classification target.
        RuntimeError
            If no model fits.
        """
        random_state = _seed(random_state, "random_state")
        self.is_fitted_ = False
        for attribute in ("train_X_", "train_y_", "train_weight_"):
            self.__dict__.pop(attribute, None)
        if max_samples is not None:
            _resolve_max_samples(max_samples, 1)
        self._validate_assigned()
        self._check_built_missing()
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
        for label, value in (("y", y), ("sample_weight", sample_weight), ("shuffle_groups", shuffle_groups)):
            _check_index(X, value, label)
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
        templates: Dict[str, Any] = {}  # before one-hot wrapping
        if self.user_models is not None:
            candidates.update(self.user_models)
            templates.update(self.user_models)
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
                templates.update(built)
                candidates.update(
                    {
                        name: _with_one_hot(model, one_hot, self.max_categories, name in self._dense_one_hot)
                        for name, model in built.items()
                    }
                    if one_hot and group == "linear"
                    else built
                )

        self._check_family_names(candidates)

        self.trained_models = {}
        self.fit_models_ = {}
        for name, template in templates.items():
            try:
                self.fit_models_[name] = clone(template, safe=False)
            except Exception:
                pass  # that model's fit fails and is recorded below
        if self.user_models is not None:  # train what ablate() refits: a shared RandomState gives one seed
            candidates = {name: self.fit_models_.get(name, model) for name, model in candidates.items()}
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
        self._fit_settings = {
            "missing": self.missing,
            "model_input": self.model_input,
            "max_categories": self.max_categories,
            **self._shadow_options(),
        }
        self.is_fitted_ = True
        return self

    def _shadow_options(self) -> Dict[str, Any]:
        return {
            "shadows": self.shadows,
            "shadow_seed": _plain_int(self.shadow_seed),
            "shadow_mode": self.shadow_mode,
            "shuffle": self.shuffle,
        }

    def _check_built_missing(self) -> None:
        if self._built_missing is not None and self._built_missing != self.missing:
            raise ValueError(
                "[!] missing was changed after the built-in models were made; make a new FeatureRanker."
            )

    def _check_fit_options(self) -> None:
        """Raise when an option that shaped the fit was assigned since; pickles without a record pass."""
        self._check_built_missing()
        current = {
            "missing": self.missing,
            "model_input": self.model_input,
            "max_categories": self.max_categories,
            **self._shadow_options(),
        }
        recorded = self._fit_settings or {}
        for key, value in recorded.items():
            if key == "shuffle" and not recorded.get("shadows"):
                continue  # the shuffle only shapes the shadows
            if current[key] != value:
                raise ValueError(f"[!] {key} was changed after fit(); call fit() again.")

    def _validate_assigned(self) -> None:
        """Options that may have been assigned, or unpickled, since __init__."""
        self.shadow_seed = _seed(self.shadow_seed, "shadow_seed")
        _validate_n_jobs(self.n_jobs)
        self.max_categories = _validate_max_categories(self.max_categories)

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
        Rank feature groups on evaluation data that ``fit()`` did not see.

        Parameters
        ----------
        X_eval : array-like or DataFrame
            Evaluation features with the same columns, in the same order, as
            ``fit()``.
        y_eval : array-like
            Evaluation target. Class labels must have appeared in training.
        scoring : str, callable or list
            scikit-learn scorer name, a callable ``(estimator, X, y) -> float``
            where higher is better, or a non-empty list of them (the first is
            primary and decides the ranking). Keyword-only.
        feature_groups : dict, optional
            ``{group_name: [feature, ...]}`` of columns to shuffle together.
            Ungrouped features are ranked on their own.
        n_repeats : int, default 10
            Shuffles per feature group; a positive integer.
        random_state : int, optional
            Seed (0 to 2**64 - 1) of the permutation shuffles.
        sample_weight : array-like, optional
            Finite, non-negative evaluation weights, one per row, with a
            positive sum. The baseline, permuted and dummy scores use them.
        max_samples : int or float, optional
            Score on one random subsample of the evaluation rows: an int
            (rows) or a float in (0, 1] (share of rows).
        shuffle_groups : array-like, optional
            One group label per evaluation row. Required with
            ``shuffle="within_groups"``; an error with another ``shuffle``.

        Returns
        -------
        Report
            A dict subclass; see docs/report.md for its structure.

        Raises
        ------
        RuntimeError
            If ``fit()`` was not called, or every model fails ranking.
        ValueError
            For invalid data or options, or if the columns differ from
            ``fit()``.
        """
        if not self.is_fitted_:
            raise RuntimeError("[!] Call fit() before rank_features().")
        self._validate_assigned()
        self._check_fit_options()
        if self.evidence and not hasattr(self, "train_X_"):
            raise ValueError("[!] Training evidence requires fit() with evidence=True.")
        random_state = _seed(random_state, "random_state")
        n_repeats = _plain_int(n_repeats)
        if not _is_positive_int(n_repeats):
            raise ValueError(N_REPEATS_MESSAGE)
        if max_samples is not None:
            _resolve_max_samples(max_samples, 1)
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
        for label, value in (
            ("y_eval", y_eval), ("sample_weight", sample_weight), ("shuffle_groups", shuffle_groups)
        ):
            _check_index(X_eval, value, label)
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
            ("max_categories", _plain_int(self.max_categories)),
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
        # The dummy sees the models' input form and read-only views (no shadows).
        baseline_X = _model_X(
            X_values[rows], self.feature_names_ if self.model_input == "dataframe" else None
        )
        baseline_y, baseline_w = _read_only(y_scored), _read_only(weights_scored)
        baseline_score = self._score_baseline(
            scorer, baseline_X, baseline_y, ranking_failures, baseline_w
        )
        # The dummy is not ranked, so a failure here is null, not a failure entry.
        secondary_baselines = {
            other_name: self._score_baseline(other, baseline_X, baseline_y, [], baseline_w)
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
            "schema_version": REPORT_SCHEMA_VERSION,
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
            report["shadow_seed"] = _plain_int(self.shadow_seed)
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
                self.user_models is None,
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
        Fit and rank on every cross-validation split and report how stable
        the ranking is.

        Parameters
        ----------
        X : array-like or DataFrame
            All development data, with the same rules as ``fit()``. Each split
            takes its own training and evaluation rows from it. Keep your final
            test set out.
        y : array-like
            Target, one value per row.
        scoring : str, callable or list
            As in ``rank_features()``; stability uses the primary scorer.
            Keyword-only.
        cv : int, splitter or iterable, default 5
            Number of folds (shuffled; stratified for classification), a
            scikit-learn splitter such as ``RepeatedKFold`` or
            ``GroupKFold``, or a list of ``(train_indices, eval_indices)``.
        groups : array-like, optional
            Passed to the splitter's ``split()``. With
            ``shuffle="within_groups"`` they are also the shuffle groups and
            are required.
        feature_names : list of str, optional
            As in ``fit()``.
        feature_groups : dict, optional
            As in ``rank_features()``.
        n_repeats : int, default 10
            Shuffles per feature group in each split.
        random_state : int, optional
            Seeds integer-``cv`` shuffling. Split ``i`` uses seed
            ``(random_state + i)`` modulo 2**64.
        top_k : int, default 5
            A group is "top" in a split when its place is at most ``top_k``;
            used for ``top_k_share``.
        sample_weight : array-like, optional
            Weights for all rows of ``X``, sliced per split.
        max_samples : int or float, optional
            As in ``rank_features()``, applied to each split's evaluation rows.
        max_train_samples : int or float, optional
            As ``fit()``'s ``max_samples``, applied to each split's training
            rows.

        Returns
        -------
        Report
            A resampled report with a ``stability`` table and one report per
            split. This ranker's own fitted state is not changed.

        Raises
        ------
        ValueError
            For an invalid ``cv``, ``top_k``, scorer, ``n_repeats`` or group
            before any model is fit.
        RuntimeError
            If no split ranks. A split that fails alone is recorded in
            ``failures``.
        """
        random_state = _seed(random_state, "random_state")
        n_repeats, top_k = _plain_int(n_repeats), _plain_int(top_k)
        if not _is_positive_int(n_repeats):
            raise ValueError(N_REPEATS_MESSAGE)
        if not _is_positive_int(top_k):
            raise ValueError("[!] top_k must be a positive integer below 2**63.")
        if max_samples is not None:
            _resolve_max_samples(max_samples, 1)
        if max_train_samples is not None:
            _resolve_max_samples(max_train_samples, 1)
        self._validate_assigned()
        _resolve_scorers(scoring)
        _check_cv(cv)
        categorical = _categorical_indices(X, self.categorical, feature_names)
        # Raw values: each split learns its own levels in fit().
        X_values, names = _validate_X(
            X,
            feature_names=feature_names,
            allow_nan=self.missing == "impute",
            categorical=categorical,
        )
        for label, value in (("y", y), ("sample_weight", sample_weight), ("groups", groups)):
            _check_index(X, value, label)
        y_values = _validate_y(y, len(X_values))
        weights = _validate_weights(sample_weight, len(X_values))
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
        split_groups = groups if shuffle_labels is None else shuffle_labels
        splits = _resolve_splits(cv, X_values, y_values, split_groups, self.task, random_state)
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
                        None if self.shadow_seed is None else (_plain_int(self.shadow_seed) + index) % 2**64
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
                    random_state=None if random_state is None else (random_state + index) % 2**64,
                )
                report = ranker.rank_features(
                    X_values[evaluation],
                    y_values[evaluation],
                    scoring=scoring,
                    feature_groups=feature_groups,
                    n_repeats=n_repeats,
                    random_state=None if random_state is None else (random_state + index) % 2**64,
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
            run["max_categories"] = _plain_int(self.max_categories)
        result = {
            "schema_version": REPORT_SCHEMA_VERSION,
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
            result["shadow_seed"] = _plain_int(self.shadow_seed)
            result["shadow_mode"] = self.shadow_mode
            result["shadow_test"] = {
                "method": "binomial",
                "hit": "above_shadow_share > 0.5",
                "null_probability": 0.5,
                "correction": "benjamini-hochberg",
                "n_trials": len(splits),
                "min_p_value": 0.5 ** len(splits),
                "calibrated": False,
                "experimental": True,
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
        Refit the voting models on the top-k groups to check the ranking.

        For each ``k``, every model that voted in ``report`` is fit again on
        the columns of the top-``k`` consensus groups and scored on
        ``X_eval``. Costs ``len(ks)`` refits of every voting model.

        Parameters
        ----------
        X : array-like or DataFrame
            The training data this ranker was fit on (checked against the
            report's ``run["train_data_sha256"]`` when present).
        y : array-like
            The training target.
        X_eval : array-like or DataFrame
            Data to score the refit models on, with the same columns as
            ``fit()``. Every row is scored.
        y_eval : array-like
            Target of ``X_eval``.
        report : Report
            A held-out report from this ranker's ``rank_features()``.
        ks : list of int, optional
            Numbers of top groups to keep, from 1 to the number of consensus
            groups. Default: 1, 2, 3, 5, 10, 20, 50, ... up to that number,
            plus that number. Keyword-only.
        scoring : str or callable, optional
            Defaults to the report's scorer name; pass it when the report was
            ranked with a callable.
        sample_weight : array-like, optional
            Training weights; pass the ones used for ``fit()``.
        eval_sample_weight : array-like, optional
            Evaluation weights; pass the ones used for ``rank_features()``.

        Returns
        -------
        Report
            A copy of ``report`` plus an ``ablation`` block. ``report`` is not
            modified.

        Raises
        ------
        ValueError
            For an unfitted ranker, a resampled report, an empty consensus,
            bad ``ks``, other training data or weights than the report was
            made with, or a voting model whose class or parameters differ from
            the ranker's.
        """
        from .ablation import ablate

        return ablate(
            self, X, y, X_eval, y_eval, report, ks=ks, scoring=scoring,
            sample_weight=sample_weight, eval_sample_weight=eval_sample_weight,
            _stacklevel=4,
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
        self._built_missing = self.missing
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
