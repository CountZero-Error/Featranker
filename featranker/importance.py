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
from tqdm import tqdm

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}


class FeatureRanker:
    def __init__(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all"] = "all",
        prep_file: Optional[str] = None,
        prep_class: Any = "prepFeature",
    ) -> None:
        """
        Initialize the ranker: load features, build models, and train them.

        Args:
            task: Task type — "clf" for classification or "reg" for regression.
            group: Model family — "linear", "tree", or "all" for both.
            prep_file: Path to a Python file that contains the feature-prep class.
                       Defaults to "featureCalc.py" in the current working directory.
            prep_class: Name of the class to instantiate from prep_file.
        """
        # Resolve the feature-preparation file, falling back to the default.
        if prep_file is None:
            prep_file = "featureCalc.py"

        prep_path = Path(prep_file).resolve()
        if not prep_path.is_file():
            raise FileNotFoundError(
                f"[!] Prep file not found: {prep_path}\n"
                "    Make sure you run featranker from the directory "
                "containing your featureCalc.py, or pass --prep-file."
            )

        # Dynamically load the prep module from the file path.
        spec = importlib.util.spec_from_file_location(prep_path.stem, str(prep_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"[!] Could not load module from: {prep_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[prep_path.stem] = module
        spec.loader.exec_module(module)

        # Instantiate the prep class and compute the feature dict.
        _featureCalc = getattr(module, prep_class)
        featureCalc = _featureCalc()
        self.features = featureCalc._calc_features()

        self.task = task
        self.group = group

        # Initialize model instances from the YAML config.
        self.models = self.init_models()

        # Train every initialized model on the prepared features.
        self.trained_models = self.run_ML()

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
        task=task, group=group, prep_file=prep_file, prep_class=prep_class
    )


if __name__ == "__main__":
    raise SystemExit(main())
