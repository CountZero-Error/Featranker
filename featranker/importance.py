"""
Feature importance ranking via permutation importance across multiple ML models.
"""

import argparse
import importlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import yaml
from sklearn.metrics import get_scorer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}

OPTIONAL_MODEL_MODULES = {"xgboost", "lightgbm", "catboost"}


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
    ranks_by_group: Dict[str, List[int]] = {}
    for model_report in model_reports.values():
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


def _load_prep_data(
    prep_file: str,
    prep_class: str,
    kwargs: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    prep_path = Path(prep_file).resolve()
    if not prep_path.is_file():
        raise FileNotFoundError(f"[!] Prep file not found: {prep_path}")

    spec = importlib.util.spec_from_file_location(prep_path.stem, str(prep_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"[!] Could not load module from: {prep_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[prep_path.stem] = module
    spec.loader.exec_module(module)
    prep_type = getattr(module, prep_class)
    features = prep_type(**kwargs)._calc_features()
    if not isinstance(features, dict) or not features:
        raise ValueError("[!] Prep class must return a non-empty feature dict.")
    if "label" not in features:
        raise ValueError("[!] Prep feature dict must include 'label'.")

    names = [name for name in features if name != "label"]
    if not names:
        raise ValueError("[!] Prep feature dict has no feature columns.")
    X = np.column_stack([features[name] for name in names])
    y = np.asarray(features["label"])
    return X, y, names


class FeatureRanker:
    def __init__(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all"] = "all",
        prep_file: Optional[str] = None,
        prep_class: Any = "prepFeature",
        **kwargs,
    ) -> None:
        """
        Initialize configured models without loading or fitting data.

        Args:
            task: Task type — "clf" for classification or "reg" for regression.
            group: Model family — "linear", "tree", or "all" for both.
            prep_file: Deprecated preparation-file compatibility path.
            prep_class: Deprecated preparation class name.
        """
        if task not in {"clf", "reg"}:
            raise ValueError("[!] task must be 'clf' or 'reg'.")
        if group not in {"linear", "tree", "all"}:
            raise ValueError("[!] group must be 'linear', 'tree', or 'all'.")
        self.task = task
        self.group = group
        self.prep_file = prep_file
        self.prep_class = prep_class
        self.prep_kwargs = kwargs
        self.label_encoder = None
        self.initialization_failures_: List[Dict[str, str]] = []
        self.models = self.init_models()
        self.trained_models: Dict[str, Any] = {}
        self.fit_failures_: List[Dict[str, str]] = []
        self.is_fitted_ = False
        if prep_file is not None:
            self._fit_legacy_prep(prep_file, str(prep_class), kwargs)

    def _fit_legacy_prep(
        self,
        prep_file: str,
        prep_class: str,
        kwargs: Dict[str, Any],
    ) -> "FeatureRanker":
        X, y, feature_names = _load_prep_data(prep_file, prep_class, kwargs)
        self.fit(X, y, feature_names)
        self._legacy_X = X.copy()
        self._legacy_y = y.copy()
        return self

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureRanker":
        """Fit configured models using training data only."""
        for attribute in ("_legacy_X", "_legacy_y"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        X_values, names = _validate_X(X, feature_names=feature_names)
        y_values = _validate_y(y, len(X_values))

        self.label_encoder = None
        if y_values.dtype.kind in ("U", "S", "O"):
            self.label_encoder = LabelEncoder()
            y_values = self.label_encoder.fit_transform(y_values)

        groups = ["linear", "tree"] if self.group == "all" else [self.group]
        candidates: Dict[str, Any] = {}
        for group in groups:
            candidates.update(self.models.get(self.task, {}).get(group, {}))

        self.trained_models = {}
        self.fit_failures_ = []
        self.is_fitted_ = False
        for model_name, model in tqdm(candidates.items(), desc="[*] Training models"):
            try:
                self.trained_models[model_name] = model.fit(X_values, y_values)
            except Exception as error:
                self.fit_failures_.append(_failure(model_name, "fit", error))

        if not self.trained_models:
            details = json.dumps(self.initialization_failures_ + self.fit_failures_)
            raise RuntimeError(f"[!] No models fit successfully. Failures: {details}")

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
        return self._rank_features(
            X_eval,
            y_eval,
            scoring=scoring,
            feature_groups=feature_groups,
            n_repeats=n_repeats,
            random_state=random_state,
            evaluation_mode="held_out",
        )

    def _rank_features(
        self,
        X_eval: Any,
        y_eval: Any,
        *,
        scoring: Any,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        evaluation_mode: str,
    ) -> Dict[str, Any]:
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
        groups = _resolve_feature_groups(self.feature_names_, feature_groups)
        named_groups = {
            name: [self.feature_names_[index] for index in indices]
            for name, indices in groups.items()
        }

        model_reports: Dict[str, Any] = {}
        ranking_failures: List[Dict[str, str]] = []
        for model_name, model in self.trained_models.items():
            try:
                baseline = _finite_float(
                    scorer(model, X_values, y_values), "baseline score"
                )
                rng = np.random.default_rng(random_state)
                importances: Dict[str, Dict[str, Any]] = {}
                for group_name, column_indices in groups.items():
                    values: List[float] = []
                    for _ in range(n_repeats):
                        row_order = rng.permutation(len(X_values))
                        X_permuted = X_values.copy()
                        X_permuted[:, column_indices] = X_values[row_order][
                            :, column_indices
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
            "evaluation_mode": evaluation_mode,
            "scoring": scoring_name,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "feature_groups": named_groups,
            "models": model_reports,
            "consensus": _build_consensus(model_reports),
            "failures": self.initialization_failures_
            + self.fit_failures_
            + ranking_failures,
        }

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
                        group_models[name] = model_class(**dict(params))
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

    def run_ML(self) -> Dict[str, Any]:
        """Deprecated alias that refits retained preparation-file data."""
        if not hasattr(self, "_legacy_X") or not hasattr(self, "_legacy_y"):
            raise RuntimeError(
                "[!] run_ML() requires the legacy preparation-file workflow; "
                "use fit(X_train, y_train, feature_names=...) instead."
            )
        warnings.warn(
            "run_ML() is deprecated; use fit(X_train, y_train, feature_names=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        X = self._legacy_X.copy()
        y = self._legacy_y.copy()
        self.fit(X, y, self.feature_names_)
        self._legacy_X = X
        self._legacy_y = y
        return self.trained_models

    def rankFeatures(self):
        """Deprecated in-sample ranking for preparation-file users."""
        if not hasattr(self, "_legacy_X") or not hasattr(self, "_legacy_y"):
            raise RuntimeError(
                "[!] rankFeatures() has no retained training data. Supply separate "
                "evaluation data with rank_features(X_eval, y_eval, scoring=...)."
            )
        warnings.warn(
            "rankFeatures() performs in-sample ranking and is deprecated. "
            "Use rank_features(X_eval, y_eval, scoring=...) with held-out data.",
            DeprecationWarning,
            stacklevel=2,
        )
        scoring = "accuracy" if self.task == "clf" else "r2"
        return self._rank_features(
            self._legacy_X,
            self._legacy_y,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
            evaluation_mode="in_sample",
        )


def main() -> int:
    """CLI entry point — parse arguments, run the ranking, and output results."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank feature importance using ML models defined in "
            "importance_config.yaml.  Requires a prep class (default: "
            "prepFeature in featureCalc.py) that supplies features and labels."
        )
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg"],
        help="Task type: 'clf' for classification, 'reg' for regression.",
    )
    parser.add_argument(
        "--prep-class",
        default="prepFeature",
        help=(
            "Name of the feature-preparation class to instantiate. "
            "Loaded from the file specified by --prep-file (or the default "
            "featureCalc.py when --prep-file is omitted)."
        ),
    )
    parser.add_argument(
        "--prep-file",
        default=None,
        help=(
            "Path to a custom Python file containing the prep class. "
            "Use this to avoid editing installed package files."
        ),
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
    args = parser.parse_args()

    # Show help when invoked with no arguments.
    if args.task is None:
        parser.print_help()
        return 0

    ranker = build_ranker(
        task=args.task,
        group=args.group,
        prep_class=args.prep_class,
        prep_file=args.prep_file,
    )
    results = ranker.rankFeatures()

    output = json.dumps(results, indent=2)

    if args.output:
        # Guarantee a .json extension when writing to disk.
        output_path = args.output
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


def build_ranker(
    task: Literal["clf", "reg"],
    group: Literal["linear", "tree", "all"] = "all",
    prep_file: Optional[str] = None,
    prep_class: Any = "prepFeature",
    **kwargs,
) -> "FeatureRanker":
    """
    Convenience factory that creates and returns a fully initialized
    FeatureRanker (features loaded, models trained, ready to rank).

    Args:
        task: Task type — "clf" or "reg".
        group: Model family — "linear", "tree", or "all".
        prep_file: Path to the Python file containing the prep class.
        prep_class: Name of the prep class inside that file.

    Returns:
        A ready-to-use FeatureRanker instance.
    """
    return FeatureRanker(
        task=task,
        group=group,
        prep_file=prep_file or "featureCalc.py",
        prep_class=prep_class,
        **kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
