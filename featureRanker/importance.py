"""
Examine Feature Importance
"""

import argparse
import importlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import yaml
try:
    from . import featureCalc
except ImportError:  # pragma: no cover - fallback for direct script execution
    import featureCalc  # type: ignore
from sklearn.inspection import permutation_importance
from tqdm import tqdm

task_map = {
    "classification": "clf",
    "regression": "reg",
    "clf": "classification",
    "reg": "regression",
}


def _resolve_prep_class(class_name: str):
    return _resolve_prep_class_from_module(
        module=featureCalc,
        class_name=class_name,
        source_label="featureRanker/featureCalc.py",
    )


def _resolve_prep_class_from_module(
    module: ModuleType, class_name: str, source_label: str
):
    try:
        prep_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"[!] prepFeature class '{class_name}' not found in {source_label}"
        ) from exc

    if not isinstance(prep_cls, type):
        raise ValueError(f"[!] '{class_name}' is not a class.")

    if not hasattr(prep_cls, "_calc_features"):
        raise ValueError(f"[!] '{class_name}' does not define _calc_features().")

    return prep_cls


def _load_module_from_file(prep_file: str) -> ModuleType:
    prep_path = Path(prep_file).expanduser().resolve()
    if not prep_path.exists():
        raise ValueError(f"[!] prep file does not exist: {prep_path}")
    if not prep_path.is_file():
        raise ValueError(f"[!] prep file is not a file: {prep_path}")

    module_name = f"_featureRanker_user_prep_{abs(hash(str(prep_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, prep_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"[!] Could not load module from prep file: {prep_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_prep_class_with_source(
    class_name: str, prep_file: Optional[str] = None
):
    if prep_file:
        user_module = _load_module_from_file(prep_file)
        return _resolve_prep_class_from_module(
            module=user_module,
            class_name=class_name,
            source_label=f"{Path(prep_file).expanduser().resolve()}",
        )

    return _resolve_prep_class(class_name)


def _build_feature_ranker(prep_class_name: str, prep_file: Optional[str] = None):
    prep_cls = _resolve_prep_class_with_source(prep_class_name, prep_file)

    class FeatureRanker(prep_cls, RankerMixin):
        def __init__(
            self,
            task: Literal["clf", "reg"],
            group: Literal["linear", "tree", "all"] = "all",
        ):
            super().__init__()
            self._init_ranker(task=task, group=group)

    FeatureRanker.__name__ = "FeatureRanker"
    return FeatureRanker


class RankerMixin:
    def _init_ranker(
        self,
        task: Literal["clf", "reg"],
        group: Literal["linear", "tree", "all"] = "all",
    ) -> None:
        """
        Initialize feature ranker, load models from config, and train them.

        Args:
            task: Task type to run ("clf" or "reg").
            group: Model group to train ("linear", "tree", or "all").
        """
        self.task = task
        self.group = group

        # Initrialize models
        self.models = self.init_models()

        # Train models
        self.trained_models = self.run_ML()

    def init_models(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load model definitions from importance_config.yaml and instantiate them.

        Returns:
            Dict[task, Dict[group, Dict[model_name, model_instance]]]
        """
        # Load config bundled in the package.
        config_path = Path(__file__).with_name("importance_config.yaml")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        models: Dict[str, Dict[str, Dict[str, Any]]] = {
            "clf": {"linear": {}, "tree": {}},
            "reg": {"linear": {}, "tree": {}},
        }

        # Only initialize the requested task/group to avoid duplicate work.
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
                    if module_path == "lightgbm":
                        # Silence LightGBM training logs unless overridden in config.
                        params.setdefault("verbosity", -1)

                    group_models[name] = model_class(**params)

                models[task_key][group_name] = group_models

        return models

    def run_ML(self) -> Dict[str, Any]:
        """
        Train all initialized models on the prepared features.

        Returns:
            Dict[model_name, trained_model]
        """
        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise ValueError("[!] inputs must be a non-empty dict of features.")

        """Unpack Features"""
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

        """Train Models"""
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
        Rank feature importance for each trained model and compute an average ranking.

        Returns:
            Dict[model_name, List[Dict[feature_name, score]]] with an "average" key.
        """
        if not isinstance(self.features, dict) or len(self.features) == 0:
            raise ValueError("[!] inputs must be a non-empty dict of features.")

        if "label" not in self.features:
            raise ValueError("[!] inputs must include a 'label' feature.")

        y = np.asarray(self.features["label"])
        n_samples = len(y)
        if n_samples == 0:
            raise ValueError("[!] label must be a non-empty sequence.")

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
                # Suppress sklearn warnings about missing feature names for some estimators.
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

            # Keep output compact with 4 decimal places.
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

        # Average across models for a single summary ranking.
        average_scores = {
            fname: float(round(np.mean(scores), 4))
            for fname, scores in score_bank.items()
            if scores
        }
        results["average"] = _sorted_scores(average_scores)

        return results


# Default FeatureRanker uses the base prepFeature class.
FeatureRanker = _build_feature_ranker("prepFeature")


def build_ranker(
    prep_class: str = "prepFeature", prep_file: Optional[str] = None
):
    """
    Build a FeatureRanker class using a prep class from package or external file.

    Args:
        prep_class: Prep class name to use.
        prep_file: Optional path to a Python file that defines prep classes.

    Returns:
        Dynamically built FeatureRanker class.
    """
    return _build_feature_ranker(prep_class, prep_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank feature importance with ML models defined in importance_config.yaml. "
            "Requires a prepFeature class in featureRanker/featureCalc.py "
            "to provide features and labels."
        )
    )
    parser.add_argument(
        "--task",
        choices=["clf", "reg"],
        help=("Task type to run: `clf` for classification or `reg` for regression."),
    )
    parser.add_argument(
        "--prep-class",
        default="prepFeature",
        help=(
            "Name of the prep class to use. "
            "With --prep-file, this class is loaded from that file. "
            "Without --prep-file, class is loaded from featureRanker/featureCalc.py."
        ),
    )
    parser.add_argument(
        "--prep-file",
        default=None,
        help=(
            "Optional path to user prep Python file. "
            "Use this to avoid editing installed package files."
        ),
    )
    parser.add_argument(
        "--group",
        choices=["linear", "tree", "all"],
        default="all",
        help=(
            "Model family to run: "
            "`linear` (linear models), `tree` (tree/ensemble models), or `all`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write results as JSON to this path. "
            "If no extension is provided, `.json` will be appended. "
            "If omitted, results are printed to stdout."
        ),
    )
    args = parser.parse_args()

    RankerClass = _build_feature_ranker(args.prep_class, args.prep_file)
    ranker = RankerClass(task=args.task, group=args.group)
    results = ranker.rankFeatures()

    output = json.dumps(results, indent=2)

    if args.output:
        # Ensure output path ends with .json when writing to disk.
        output_path = args.output
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
