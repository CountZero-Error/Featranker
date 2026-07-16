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
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}


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

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureRanker":
        """Fit configured models using training data only."""
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
            details = json.dumps(self.fit_failures_)
            raise RuntimeError(f"[!] No models fit successfully. Failures: {details}")

        self.feature_names_ = names
        self.n_features_in_ = len(names)
        self.is_fitted_ = True
        return self

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
                    params = model.get("params") or {}
                    if not name or not module_path or not class_name:
                        continue

                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name, None)
                    if model_class is None:
                        raise ValueError(
                            f"[!] Model class '{class_name}' not found in module '{module_path}'"
                        )

                    if not isinstance(params, dict):
                        raise ValueError(
                            f"[!] Params for model '{name}' must be a dict, got {type(params)}"
                        )

                    params = dict(params)
                    # Suppress verbose LightGBM output unless the user overrides it.
                    if module_path == "lightgbm":
                        params.setdefault("verbosity", -1)

                    group_models[name] = model_class(**params)

                models[task_key][group_name] = group_models

        return models

    def run_ML(self) -> Dict[str, Any]:
        """
        Train every initialized model on the feature matrix.

        The feature dict is unpacked into a NumPy array X (features) and a
        vector y (labels), then each model is fitted via its .fit() method.

        Returns:
            Dict mapping model_name → trained model instance.
        """
        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise ValueError("[!] inputs must be a non-empty dict of features.")

        # --- Unpack the feature dict into X and y arrays ---
        feature_names = list(self.features.keys())
        n_samples = len(self.features[feature_names[0]])
        X, y = [], []
        feature_names = []
        for feature_name, values in tqdm(
            self.features.items(), desc="[*] Unpacking features"
        ):
            if len(values) != n_samples:
                raise ValueError(
                    f"[!] Feature '{feature_name}' length {len(values)} does not match {n_samples}."
                )

            if feature_name == "label":
                y = self.features["label"]
            else:
                feature_names.append(np.asarray(values, dtype=float))
                X.append(values)

        X = np.column_stack(X)
        y = np.asarray(y)

        # Encode string labels to integers for models that require numeric targets
        if y.dtype.kind in ("U", "S", "O"):  # Unicode, byte string, or object
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # --- Fit each model in the selected group(s) ---
        group2train = ["linear", "tree"] if self.group == "all" else [self.group]

        trained_models: Dict[str, Any] = {}
        for group in group2train:
            models = self.models.get(self.task, {}).get(group, {})

            for model_name, model in tqdm(
                models.items(),
                desc=f"[*] Training {task_map[self.task]}-{group} models",
            ):
                try:
                    trained_models[model_name] = model.fit(X, y)
                except Exception as e:
                    print(
                        f"[*] Error training {task_map[self.task]}-{group} model '{model_name}': {e}"
                    )
                    continue

        return trained_models

    def rankFeatures(self):
        """
        Compute permutation importance for every trained model and produce
        a per-model ranking plus an overall average ranking.

        Returns:
            Dict mapping each model name to a sorted list of
            {feature_name: score} dicts, plus an "average" entry that
            aggregates scores across all models.
        """
        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise ValueError("[!] inputs must be a non-empty dict of features.")

        if "label" not in self.features:
            raise ValueError("[!] inputs must include a 'label' feature.")

        y = np.asarray(self.features["label"])
        # Re-apply label encoding if labels were encoded during training
        if self.label_encoder is not None:
            y = self.label_encoder.transform(y)
        n_samples = len(y)
        if n_samples == 0:
            raise ValueError("[!] label must be a non-empty sequence.")

        # Separate feature columns from the label.
        feature_names: List[str] = []
        X_cols: List[np.ndarray] = []
        for feature_name, values in self.features.items():
            if feature_name == "label":
                continue

            if len(values) != n_samples:
                raise ValueError(
                    f"[!] Feature '{feature_name}' length {len(values)} does not match {n_samples}."
                )

            feature_names.append(feature_name)
            X_cols.append(np.asarray(values, dtype=float))

        if len(feature_names) == 0:
            raise ValueError("[!] No feature columns found (excluding 'label').")

        X = np.column_stack(X_cols)

        if not isinstance(self.trained_models, dict) or len(self.trained_models) == 0:
            raise ValueError("[!] No trained models available.")

        def _sorted_scores(scores: Dict[str, float]) -> List[Dict[str, float]]:
            """Return a list of single-entry dicts sorted by score descending."""
            return [
                {name: scores[name]}
                for name in sorted(scores, key=scores.get, reverse=True)
            ]

        results: Dict[str, List[Dict[str, float]]] = {}
        score_bank: Dict[str, List[float]] = {name: [] for name in feature_names}
        any_model = False

        for model_name, model in tqdm(
            self.trained_models.items(), desc="[*] Computing permutation importance"
        ):
            try:
                # Silence sklearn warnings about feature-name mismatches.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    perm = permutation_importance(
                        model, X, y, n_repeats=10, random_state=42
                    )
            except Exception as e:
                print(f"[*] Error computing importance for model '{model_name}': {e}")
                continue

            # Round scores to four decimal places for compact output.
            model_scores = {
                fname: float(round(score, 4))
                for fname, score in zip(feature_names, perm.importances_mean)
            }
            for fname, score in model_scores.items():
                score_bank[fname].append(score)

            results[model_name] = _sorted_scores(model_scores)
            any_model = True

        if not any_model:
            raise ValueError("[!] Failed to compute importance for any trained model.")

        # Aggregate a single summary ranking averaged across all models.
        average_scores = {
            fname: float(round(np.mean(scores), 4))
            for fname, scores in score_bank.items()
            if scores
        }
        results["average"] = _sorted_scores(average_scores)

        return results


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
        task=task, group=group, prep_file=prep_file, prep_class=prep_class, **kwargs
    )


if __name__ == "__main__":
    raise SystemExit(main())
