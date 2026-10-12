"""
Feature importance ranking via permutation importance across multiple ML models.
"""

import importlib
import importlib.metadata
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import sklearn
import yaml
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.fixes import parse_version
from tqdm import tqdm

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}

OPTIONAL_MODEL_MODULES = {"xgboost", "lightgbm", "catboost"}

BASELINE_NAME = "dummy_baseline"

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


def _validate_X(
    X: Any,
    feature_names: Optional[List[str]] = None,
    expected_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, List[str]]:
    dataframe_names = list(X.columns) if hasattr(X, "columns") else None
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("[!] X must be a two-dimensional feature matrix.")
    if not np.isfinite(values).all():
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


def _validate_y(y: Any, n_rows: int) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("[!] y must be one-dimensional.")
    if len(values) != n_rows:
        raise ValueError(
            f"[!] X has {n_rows} rows but y has {len(values)} rows."
        )
    if len(values) == 0:
        raise ValueError("[!] y must contain at least one value.")
    return values


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


def _failure(model: str, stage: str, error: Exception) -> Dict[str, str]:
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
    ranks_by_group: Dict[str, List[int]] = {}
    for model_report in voters:
        for group_name, importance in model_report["importance"].items():
            ranks_by_group.setdefault(group_name, []).append(importance["rank"])

    consensus = []
    for group_name, ranks in ranks_by_group.items():
        consensus.append(
            {
                "feature_group": group_name,
                "median_rank": float(np.median(ranks)),
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks)),
                "n_models": len(ranks),
            }
        )
    return sorted(
        consensus,
        key=lambda row: (
            row["median_rank"],
            row["mean_rank"],
            row["feature_group"],
        ),
    )


def _run_models(
    worker: Callable[..., tuple], jobs: List[tuple], n_jobs: Optional[int], desc: str
) -> List[tuple]:
    """Run one worker per model, in order; sequentially unless n_jobs > 1 or -1."""
    if n_jobs is None or n_jobs == 1:
        return [worker(*job) for job in tqdm(jobs, desc=desc)]
    return Parallel(n_jobs=n_jobs)(delayed(worker)(*job) for job in jobs)


def _fit_model(
    name: str, model: Any, X_values: np.ndarray, y_values: np.ndarray
) -> tuple[str, Any, Optional[Dict[str, str]]]:
    try:
        return name, clone(model, safe=False).fit(X_values, y_values), None
    except Exception as error:
        return name, None, _failure(name, "fit", error)


def _permutation_importances(
    model: Any,
    scorer: Any,
    X_values: np.ndarray,
    y_values: np.ndarray,
    groups: Dict[str, List[int]],
    n_repeats: int,
    random_state: Optional[int],
    baseline: float,
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    # One scratch copy per model; only the group's columns are
    # rewritten per repeat and restored after the group.
    X_permuted = X_values.copy()
    importances: Dict[str, Dict[str, Any]] = {}
    for group_name, column_indices in groups.items():
        values: List[float] = []
        for _ in range(n_repeats):
            row_order = rng.permutation(len(X_values))
            X_permuted[:, column_indices] = X_values[np.ix_(row_order, column_indices)]
            permuted_score = _finite_float(
                scorer(model, X_permuted, y_values), "permuted score"
            )
            values.append(
                _finite_float(baseline - permuted_score, "permutation importance")
            )
        X_permuted[:, column_indices] = X_values[:, column_indices]
        importances[group_name] = {
            "values": values,
            "mean": _finite_float(np.mean(values), "importance mean"),
            "std": _finite_float(np.std(values), "importance standard deviation"),
        }
    ranks = _assign_ranks(
        {name: importance["mean"] for name, importance in importances.items()}
    )
    for group_name, rank in ranks.items():
        importances[group_name]["rank"] = rank
    return importances


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
) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    try:
        baseline = _finite_float(scorer(model, X_values, y_values), "baseline score")
        importances = _permutation_importances(
            model, scorer, X_values, y_values, groups, n_repeats, random_state,
            baseline,
        )
        report = {
            "evaluation_score": baseline,
            "in_consensus": baseline_score is None or baseline > baseline_score,
            "params": _json_params(model),
            "importance": importances,
        }
        return name, report, None
    except Exception as error:
        return name, None, _failure(name, "ranking", error)


class FeatureRanker:
    def __init__(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all"] = "all",
        models: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Initialize configured models without loading or fitting data.

        Args:
            task: Task type — "clf" for classification or "reg" for regression.
            group: Model family — "linear", "tree", or "all" for both.
            models: Optional name -> scikit-learn estimator mapping that
                replaces the built-in registry. Estimators are cloned at fit().
            n_jobs: Parallel workers over models in fit() and rank_features();
                None means sequential, -1 means all cores.
        """
        if task not in {"clf", "reg"}:
            raise ValueError("[!] task must be 'clf' or 'reg'.")
        if group not in {"linear", "tree", "all"}:
            raise ValueError("[!] group must be 'linear', 'tree', or 'all'.")
        if models is not None and group != "all":
            raise ValueError("[!] group selects built-in models; omit it with models.")
        self.task = task
        self.group = group
        self.n_jobs = n_jobs
        self.label_encoder = None
        self.initialization_failures_: List[Dict[str, str]] = []
        self.user_models = None if models is None else _validate_models(models)
        self.models = {} if models is not None else self.init_models()
        self.trained_models: Dict[str, Any] = {}
        self.baseline_model_: Any = None
        self.fit_failures_: List[Dict[str, str]] = []
        self.is_fitted_ = False

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureRanker":
        """Fit configured models using training data only."""
        self.is_fitted_ = False
        X_values, names = _validate_X(X, feature_names=feature_names)
        y_values = _validate_y(y, len(X_values))

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
            groups = ["linear", "tree"] if self.group == "all" else [self.group]
            for group in groups:
                candidates.update(self.models.get(self.task, {}).get(group, {}))

        self.trained_models = {}
        self.fit_failures_ = []
        jobs = [(name, model, X_values, y_values) for name, model in candidates.items()]
        for name, fitted, failure in _run_models(
            _fit_model, jobs, self.n_jobs, "[*] Training models"
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
            baseline_model = dummy.fit(X_values, y_values)
        except Exception as error:
            self.fit_failures_.append(_failure(BASELINE_NAME, "fit", error))

        self.baseline_model_ = baseline_model
        self.label_encoder = label_encoder
        self.feature_names_ = names
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
    ) -> Dict[str, Any]:
        """Rank feature groups on explicitly supplied evaluation data."""
        if not self.is_fitted_:
            raise RuntimeError("[!] Call fit() before rank_features().")
        if not isinstance(n_repeats, int) or isinstance(n_repeats, bool) or n_repeats < 1:
            raise ValueError("[!] n_repeats must be a positive integer.")
        if isinstance(scoring, str):
            scorer = get_scorer(scoring)
            scoring_name = scoring
        elif callable(scoring):
            scorer = scoring
            scoring_name = getattr(scoring, "__name__", type(scoring).__name__)
        else:
            raise TypeError(
                "[!] scoring must be a scikit-learn scorer name or callable."
            )

        X_values, _ = _validate_X(X_eval, expected_names=self.feature_names_)
        y_values = _validate_y(y_eval, len(X_values))
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
            scorer, X_values, y_values, ranking_failures
        )
        jobs = [
            (name, model, scorer, X_values, y_values, groups, n_repeats,
             random_state, baseline_score)
            for name, model in self.trained_models.items()
        ]
        for name, model_report, failure in _run_models(
            _rank_model, jobs, self.n_jobs, "[*] Ranking models"
        ):
            if failure is None:
                model_reports[name] = model_report
            else:
                ranking_failures.append(failure)

        if not model_reports:
            details = json.dumps(ranking_failures)
            raise RuntimeError(
                f"[!] No models ranked successfully. Failures: {details}"
            )

        return {
            "evaluation_mode": "held_out",
            "scoring": scoring_name,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "feature_groups": named_groups,
            "baseline_score": baseline_score,
            "provenance": _provenance(),
            "models": model_reports,
            "consensus": _build_consensus(model_reports),
            "failures": self.initialization_failures_
            + self.fit_failures_
            + ranking_failures,
        }

    def _score_baseline(
        self,
        scorer: Any,
        X_values: np.ndarray,
        y_values: np.ndarray,
        failures: List[Dict[str, str]],
    ) -> Optional[float]:
        """Score the dummy baseline; None (no gating) if it was unusable."""
        if self.baseline_model_ is None:
            return None
        try:
            return _finite_float(
                scorer(self.baseline_model_, X_values, y_values), "baseline score"
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
        config_path = Path(__file__).with_name("importance_config.yaml")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        models: Dict[str, Dict[str, Dict[str, Any]]] = {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {}, "tree": {}},
        }

        # Determine which tasks and groups to initialize.
        allowed_tasks = {"clf", "reg"} if self.task == "all" else {self.task}
        allowed_groups = {"linear", "tree"} if self.group == "all" else {self.group}

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

                group_models: Dict[str, Any] = {}
                for model in tqdm(
                    model_configs,
                    desc=f"[*] Initialize {task_name}-{group_name} models",
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
                        # Linear models are scale-sensitive; the scaler is fit
                        # on training data only, inside the pipeline.
                        group_models[name] = (
                            make_pipeline(StandardScaler(), estimator)
                            if group_name == "linear"
                            else estimator
                        )
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
