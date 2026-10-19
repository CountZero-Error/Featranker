"""
Command-line interface: rank features from separate train and evaluation CSVs.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sklearn.metrics import get_scorer

from .evidence import auto_groups
from .importance import FeatureRanker, _float_target, _resolve_feature_groups


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
            row_values = [float(row[index]) for index in feature_indices]
        except ValueError as error:
            raise ValueError(
                f"[!] CSV feature values must be numeric (row {row_number})."
            ) from error
        if not all(math.isfinite(value) for value in row_values):
            raise ValueError(
                f"[!] CSV feature values must be finite (row {row_number}); "
                "impute or drop missing values first."
            )
        features.append(row_values)
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
        default=None,
        help=(
            "Optional YAML mapping of feature-group names to column names, or "
            "'auto' to group correlated training columns (a file named 'auto' "
            "needs './auto')."
        ),
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
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers over models (default: sequential; -1 uses all cores).",
    )
    parser.add_argument(
        "--group",
        choices=["linear", "tree", "all", "fast"],
        default="all",
        help=(
            "Built-in model group: 'linear' (linear models only), "
            "'tree' (tree/ensemble models only), 'all' (both), or "
            "'fast' (one representative per model family)."
        ),
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help=(
            "Add mechanism evidence to the report: train score, overfit gap, "
            "correlated groups, and SHAP when the shap package is installed."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide progress bars (warnings and errors are still shown).",
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


def _load_feature_groups(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as group_file:
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
    if args.feature_groups == "auto":
        feature_groups: Optional[Dict[str, List[str]]] = auto_groups(
            X_train, feature_names=feature_names
        )
    else:
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
    ranker = FeatureRanker(
        task=args.task,
        group=args.group,
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
        evidence=args.evidence,
    )
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
