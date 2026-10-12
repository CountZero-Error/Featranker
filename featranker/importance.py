"""
Feature importance ranking via permutation importance across multiple ML models.
"""

import argparse
import csv
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import sklearn
import yaml
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import get_scorer
from sklearn.preprocessing import LabelEncoder
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


def _load_csv(
    path: Path, target: str
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a numeric feature matrix and target column from a CSV file."""
    # utf-8-sig drops the BOM that Excel "CSV UTF-8" exports prepend.
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration as error:
            raise ValueError(f"[!] CSV '{path}' is empty.") from error
        rows = [
            (row_number, row)
            for row_number, row in enumerate(reader, start=2)
            if row
        ]

    if any(not name for name in header):
        raise ValueError("[!] CSV column names must be non-empty.")
    if len(set(header)) != len(header):
        raise ValueError("[!] CSV column names must be unique.")
    if target not in header:
        raise ValueError(f"[!] Target column '{target}' was not found in '{path}'.")
    if len(header) == 1:
        raise ValueError("[!] CSV must contain at least one feature column.")
    if not rows:
        raise ValueError(f"[!] CSV '{path}' must contain at least one data row.")

    target_index = header.index(target)
    feature_indices = [index for index in range(len(header)) if index != target_index]
    features: List[List[float]] = []
    targets: List[str] = []
    for row_number, row in rows:
        if len(row) != len(header):
            raise ValueError(
                f"[!] CSV row {row_number} has {len(row)} columns; "
                f"expected {len(header)}."
            )
        try:
            features.append([float(row[index]) for index in feature_indices])
        except ValueError as error:
            raise ValueError(
                f"[!] CSV feature values must be numeric (row {row_number})."
            ) from error
        targets.append(row[target_index])

    if any(value == "" for value in targets):
        raise ValueError("[!] Target values must be non-empty.")
    # Raw strings: FeatureRanker types the target by task, so train and eval
    # labels stay comparable even when only one file looks numeric.
    return (
        np.asarray(features, dtype=float),
        np.asarray(targets),
        [header[index] for index in feature_indices],
    )


def _validate_X(
    X: Any,
    feature_names: Optional[List[str]] = None,
    expected_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, List[str]]:
    dataframe_names = list(X.columns) if hasattr(X, "columns") else None
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError("[!] X must be a two-dimensional feature matrix.")
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


class FeatureRanker:
    def __init__(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all"] = "all",
    ) -> None:
        """
        Initialize configured models without loading or fitting data.

        Args:
            task: Task type — "clf" for classification or "reg" for regression.
            group: Model family — "linear", "tree", or "all" for both.
        """
        if task not in {"clf", "reg"}:
            raise ValueError("[!] task must be 'clf' or 'reg'.")
        if group not in {"linear", "tree", "all"}:
            raise ValueError("[!] group must be 'linear', 'tree', or 'all'.")
        self.task = task
        self.group = group
        self.label_encoder = None
        self.initialization_failures_: List[Dict[str, str]] = []
        self.models = self.init_models()
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

        groups = ["linear", "tree"] if self.group == "all" else [self.group]
        candidates: Dict[str, Any] = {}
        for group in groups:
            candidates.update(self.models.get(self.task, {}).get(group, {}))

        self.trained_models = {}
        self.fit_failures_ = []
        for model_name, model in tqdm(candidates.items(), desc="[*] Training models"):
            try:
                self.trained_models[model_name] = model.fit(X_values, y_values)
            except Exception as error:
                self.fit_failures_.append(_failure(model_name, "fit", error))

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
            y_values = self.label_encoder.transform(y_values)
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
        for model_name, model in self.trained_models.items():
            try:
                baseline = _finite_float(
                    scorer(model, X_values, y_values), "baseline score"
                )
                rng = np.random.default_rng(random_state)
                # One scratch copy per model; only the group's columns are
                # rewritten per repeat and restored after the group.
                X_permuted = X_values.copy()
                importances: Dict[str, Dict[str, Any]] = {}
                for group_name, column_indices in groups.items():
                    values: List[float] = []
                    for _ in range(n_repeats):
                        row_order = rng.permutation(len(X_values))
                        X_permuted[:, column_indices] = X_values[
                            np.ix_(row_order, column_indices)
                        ]
                        permuted_score = _finite_float(
                            scorer(model, X_permuted, y_values), "permuted score"
                        )
                        values.append(
                            _finite_float(
                                baseline - permuted_score,
                                "permutation importance",
                            )
                        )
                    X_permuted[:, column_indices] = X_values[:, column_indices]
                    importances[group_name] = {
                        "values": values,
                        "mean": _finite_float(
                            np.mean(values), "importance mean"
                        ),
                        "std": _finite_float(
                            np.std(values), "importance standard deviation"
                        ),
                    }
                ranks = _assign_ranks(
                    {
                        group_name: importance["mean"]
                        for group_name, importance in importances.items()
                    }
                )
                for group_name, rank in ranks.items():
                    importances[group_name]["rank"] = rank
                model_reports[model_name] = {
                    "evaluation_score": baseline,
                    "in_consensus": baseline_score is None or baseline > baseline_score,
                    "importance": importances,
                }
            except Exception as error:
                ranking_failures.append(_failure(model_name, "ranking", error))

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
                        group_models[name] = model_class(
                            **_model_params(class_name, params)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit feature-ranking models on explicit training data and rank "
            "features on separate evaluation data."
        )
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg"],
        required=True,
        help="Task type: 'clf' for classification, 'reg' for regression.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="CSV containing training features and target.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        required=True,
        help="CSV containing held-out evaluation features and target.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column in both CSV files.",
    )
    parser.add_argument(
        "--scoring",
        required=True,
        help="Scikit-learn scorer name, such as neg_mean_absolute_error.",
    )
    parser.add_argument(
        "--feature-groups",
        type=Path,
        default=None,
        help="Optional YAML mapping of feature-group names to column names.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature group (default: 10).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for reproducible permutations.",
    )
    parser.add_argument(
        "--group",
        choices=["linear", "tree", "all"],
        default="all",
        help=(
            "Model family: 'linear' (linear models only), "
            "'tree' (tree/ensemble models only), or 'all' (both)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write results as JSON to this file path. "
            "A .json extension is appended automatically if missing. "
            "When omitted, results are printed to stdout."
        ),
    )
    return parser


def _load_feature_groups(path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as group_file:
        feature_groups = yaml.safe_load(group_file)
    if not isinstance(feature_groups, dict):
        raise ValueError("[!] Feature groups YAML must contain a mapping.")
    return feature_groups


def _load_cli_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the CSV files and validate options before any model is fit."""
    X_train, y_train, feature_names = _load_csv(args.train_data, args.target)
    X_eval, y_eval, eval_names = _load_csv(args.eval_data, args.target)
    if eval_names != feature_names:
        raise ValueError(
            "[!] Evaluation feature names and order must match training data."
        )
    if args.task == "reg":
        y_train, y_eval = _float_target(y_train), _float_target(y_eval)
    feature_groups = _load_feature_groups(args.feature_groups)
    # Fail fast: rank_features repeats these checks, but only after
    # every model has been fit.
    get_scorer(args.scoring)
    if args.n_repeats < 1:
        raise ValueError("[!] n_repeats must be a positive integer.")
    _resolve_feature_groups(feature_names, feature_groups)
    return {
        "train": (X_train, y_train, feature_names),
        "eval": (X_eval, y_eval),
        "feature_groups": feature_groups,
    }


def _run_ranking(args: argparse.Namespace) -> Dict[str, Any]:
    inputs = _load_cli_inputs(args)
    ranker = FeatureRanker(task=args.task, group=args.group)
    ranker.fit(*inputs["train"])
    return ranker.rank_features(
        *inputs["eval"],
        scoring=args.scoring,
        feature_groups=inputs["feature_groups"],
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )


def _write_report(results: Dict[str, Any], output_path: Optional[Path]) -> None:
    output = json.dumps(results, indent=2)
    if output_path is None:
        print(output)
        return
    # Guarantee a .json extension when writing to disk.
    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(".json")
    output_path.write_text(output, encoding="utf-8")


def main() -> int:
    """CLI entry point — parse arguments, run the ranking, and output results."""
    parser = _build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    try:
        results = _run_ranking(args)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        yaml.YAMLError,
    ) as error:
        parser.error(str(error))
    _write_report(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
