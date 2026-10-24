"""
Command-line interface: rank features from separate train and evaluation CSVs.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from .evidence import auto_groups, auto_groups_record
from .importance import (
    FeatureRanker,
    _float_target,
    _resolve_feature_groups,
    _resolve_max_samples,
    _resolve_scorers,
)


def _read_csv(path: Path, target: str) -> tuple[List[str], List[tuple[int, List[str]]]]:
    """The validated header and the non-blank (row number, cells) rows of a CSV file."""
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
    return header, rows


def _is_number(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def _infer_categorical(path: Path, target: str) -> List[str]:
    """Feature columns with a non-blank cell that is not a number."""
    header, rows = _read_csv(path, target)
    return [
        name
        for index, name in enumerate(header)
        if name != target
        and any(len(row) == len(header) and row[index].strip() and not _is_number(row[index])
                for _, row in rows)
    ]


def _check_drop(
    header: List[str], target: str, weight: Optional[str], drop: Sequence[str], path: Path
) -> None:
    for name in drop:
        if name not in header:
            raise ValueError(f"[!] --drop names unknown column '{name}' in '{path}'.")
        if name == target:
            raise ValueError(f"[!] --drop names the target column '{name}'.")
        if name == weight:
            raise ValueError(f"[!] --drop names the --sample-weight column '{name}'.")


def _load_csv(
    path: Path,
    target: str,
    allow_nan: bool = False,
    categorical: Optional[List[str]] = None,
    weight: Optional[str] = None,
    drop: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a numeric feature matrix and target column from a CSV file.

    With allow_nan, an empty feature cell is NaN (infinite values still raise).
    Columns named in categorical stay text (an empty cell is None, a missing
    level) and the matrix is then an object array. The weight column is not a
    feature (see _load_weights), and neither are the columns named in drop.
    """
    header, rows = _read_csv(path, target)
    _check_drop(header, target, weight, drop, path)
    text = set(categorical or [])
    unknown = sorted(text - set(header))
    if unknown:
        raise ValueError(f"[!] categorical names unknown column '{unknown[0]}' in '{path}'.")
    target_index = header.index(target)
    feature_indices = [
        index for index in range(len(header))
        if index != target_index and header[index] != weight and header[index] not in drop
    ]
    if drop and not feature_indices:
        raise ValueError("[!] --drop removed every feature column.")
    features: List[List[Any]] = []
    targets: List[str] = []
    for row_number, row in rows:
        if len(row) != len(header):
            raise ValueError(
                f"[!] CSV row {row_number} has {len(row)} columns; "
                f"expected {len(header)}."
            )
        try:
            row_values: List[Any] = [
                (row[index].strip() or None) if header[index] in text
                else math.nan if allow_nan and not row[index].strip()
                else float(row[index])
                for index in feature_indices
            ]
        except ValueError as error:
            blank = any(not row[index].strip() for index in feature_indices)
            hint = "; use --missing impute to read empty cells as missing" if blank else ""
            raise ValueError(
                f"[!] CSV feature values must be numeric (row {row_number}){hint}."
            ) from error
        numeric_values = [
            value for index, value in zip(feature_indices, row_values)
            if header[index] not in text
        ]
        if allow_nan:
            if any(math.isinf(value) for value in numeric_values):
                raise ValueError(
                    f"[!] CSV feature values must not be infinite (row {row_number})."
                )
        elif not all(math.isfinite(value) for value in numeric_values):
            hint = (
                ", or use --missing impute"
                if any(math.isnan(value) for value in numeric_values)
                else ""
            )
            raise ValueError(
                f"[!] CSV feature values must be finite (row {row_number}); "
                f"impute or drop missing values first{hint}."
            )
        features.append(row_values)
        targets.append(row[target_index])

    if any(value == "" for value in targets):
        raise ValueError("[!] Target values must be non-empty.")
    # Raw strings: FeatureRanker types the target by task, so train and eval
    # labels stay comparable even when only one file looks numeric.
    return (
        np.asarray(features, dtype=object if text else float),
        np.asarray(targets),
        [header[index] for index in feature_indices],
    )


def _load_weights(path: Path, target: str, column: str) -> np.ndarray:
    """The sample-weight column of a CSV file (rows were checked by _load_csv)."""
    header, rows = _read_csv(path, target)
    if column == target or column not in header:
        raise ValueError(f"[!] Weight column '{column}' was not found in '{path}'.")
    index = header.index(column)
    try:
        return np.array([float(row[index]) for _, row in rows])
    except ValueError as error:
        raise ValueError(f"[!] CSV weight values must be numeric in '{path}'.") from error


def _max_samples(text: str) -> Any:
    """'50' is 50 rows; '0.5' is half the rows."""
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid max-samples value: {text!r}") from None


def _max_train_samples(text: str) -> Any:
    """As --max-samples, for the training rows."""
    try:
        return _max_samples(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(f"invalid max-train-samples value: {text!r}") from None


def _ablate_ks(text: str) -> List[int]:
    """'1,2,5' is [1, 2, 5]; every entry must be a positive int."""
    try:
        ks = [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid ablate-ks value: {text!r}") from None
    if any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError(f"ablate-ks must be positive ints, got {text!r}")
    return ks


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
        help=(
            "Scikit-learn scorer name, such as neg_mean_absolute_error, or "
            "several separated by commas (the first is primary)."
        ),
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
        "--auto-groups-threshold",
        type=float,
        default=0.7,
        help=(
            "With --feature-groups auto, group columns whose |Spearman rho| is "
            "at least this (default: 0.7)."
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
        "--max-samples",
        type=_max_samples,
        default=None,
        help=(
            "Score on a random subsample of the evaluation rows: an int (rows) "
            "or a float with a decimal point in (0, 1] (share of rows)."
        ),
    )
    parser.add_argument(
        "--max-train-samples",
        type=_max_train_samples,
        default=None,
        help=(
            "Fit the models on a random subsample of the training rows: an int "
            "(rows) or a float with a decimal point in (0, 1] (share of rows). "
            "Drawn with --random-state."
        ),
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help=(
            "Keep at most this many one-hot columns (at least 2) per categorical "
            "column in the linear models; rarer levels share one column."
        ),
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
            "Built-in model group: 'linear' (linear, discriminant, naive Bayes "
            "and kernel models), "
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
        "--missing",
        choices=["error", "impute"],
        default="error",
        help=(
            "Missing feature values (empty CSV cells): 'error' rejects them "
            "(default); 'impute' gives every built-in model a median imputer "
            "fit on training data only."
        ),
    )
    parser.add_argument(
        "--categorical",
        default=None,
        help=(
            "Text feature columns: 'auto' (columns with non-numeric cells in the "
            "training CSV) or comma-separated column names. Levels are learned "
            "from the training rows only; an empty cell is a missing level."
        ),
    )
    parser.add_argument(
        "--sample-weight",
        default=None,
        metavar="COLUMN",
        help=(
            "Weight column present in both CSV files: used as sample weights "
            "for fitting and scoring, and not ranked as a feature."
        ),
    )
    parser.add_argument(
        "--drop",
        default=None,
        metavar="COLUMNS",
        help=(
            "Comma-separated columns to remove from both CSV files before "
            "features are read, such as an ID or timestamp column."
        ),
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help=(
            "Add an ablation curve to the report: refit the voting models on the "
            "top-k consensus groups and score them on the evaluation data."
        ),
    )
    parser.add_argument(
        "--ablate-ks",
        type=_ablate_ks,
        default=None,
        metavar="KS",
        help="With --ablate, comma-separated k values, such as 1,2,5 (default: 1, 2, 3, 5, 10, ...).",
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


def _drop_names(args: argparse.Namespace) -> List[str]:
    if args.drop is None:
        return []
    names = [name.strip() for name in args.drop.split(",")]
    if not all(names):
        raise ValueError("[!] --drop needs comma-separated column names.")
    return names


def _check_groups_keep_dropped_out(
    feature_groups: Optional[Dict[str, List[str]]], drop: Sequence[str]
) -> None:
    """A dropped column cannot be a member of a --feature-groups group."""
    for members in (feature_groups or {}).values():
        if not isinstance(members, list):
            continue  # _resolve_feature_groups names the malformed group
        clash = [name for name in members if name in drop]
        if clash:
            raise ValueError(f"[!] '{clash[0]}' is given to both --drop and --feature-groups.")


def _categorical_names(args: argparse.Namespace) -> List[str]:
    """The --categorical column names; "auto" is read off the training CSV."""
    dropped = _drop_names(args)
    if args.categorical is None:
        return []
    if args.categorical.strip() == "auto":
        inferred = _infer_categorical(args.train_data, args.target)
        return [name for name in inferred if name not in dropped]
    names = [name.strip() for name in args.categorical.split(",")]
    if not all(names):
        raise ValueError("[!] --categorical needs 'auto' or comma-separated column names.")
    clash = [name for name in names if name in dropped]
    if clash:
        raise ValueError(f"[!] '{clash[0]}' is given to both --drop and --categorical.")
    return names


def _scoring_names(args: argparse.Namespace) -> List[str]:
    names = [name.strip() for name in args.scoring.split(",")]
    if not all(names):
        raise ValueError("[!] --scoring has an empty scorer name.")
    return names


def _load_cli_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the CSV files and validate options before any model is fit."""
    allow_nan = args.missing == "impute"
    if args.ablate_ks is not None and not args.ablate:
        raise ValueError("[!] --ablate-ks needs --ablate.")
    drop = _drop_names(args)
    categorical = _categorical_names(args)
    if args.sample_weight == args.target:
        raise ValueError(
            f"[!] --sample-weight '{args.sample_weight}' is the target column."
        )
    if args.sample_weight is not None and args.sample_weight in categorical:
        raise ValueError(
            f"[!] '{args.sample_weight}' is given to both --sample-weight and --categorical."
        )
    X_train, y_train, feature_names = _load_csv(
        args.train_data, args.target, allow_nan, categorical, args.sample_weight, drop
    )
    X_eval, y_eval, eval_names = _load_csv(
        args.eval_data, args.target, allow_nan, categorical, args.sample_weight, drop
    )
    # Only when given, so a ranker without the option still works here.
    train_weight: Dict[str, Any] = {}
    eval_weight: Dict[str, Any] = {}
    if args.sample_weight is not None:
        train_weight["sample_weight"] = _load_weights(
            args.train_data, args.target, args.sample_weight
        )
        eval_weight["sample_weight"] = _load_weights(
            args.eval_data, args.target, args.sample_weight
        )
    if eval_names != feature_names:
        raise ValueError(
            "[!] Evaluation feature names and order must match training data."
        )
    if args.task == "reg":
        y_train, y_eval = _float_target(y_train), _float_target(y_eval)
    if args.feature_groups == "auto":
        feature_groups: Optional[Dict[str, List[str]]] = auto_groups(
            X_train,
            threshold=args.auto_groups_threshold,
            feature_names=feature_names,
            categorical=categorical or None,
        )
    else:
        feature_groups = _load_feature_groups(args.feature_groups)
        _check_groups_keep_dropped_out(feature_groups, drop)
    # Fail fast: rank_features repeats these checks, but only after
    # every model has been fit.
    _resolve_scorers(_scoring_names(args))
    if args.n_repeats < 1:
        raise ValueError("[!] n_repeats must be a positive integer.")
    if args.max_samples is not None:
        _resolve_max_samples(args.max_samples, len(y_eval))
    if args.max_train_samples is not None:
        _resolve_max_samples(args.max_train_samples, len(y_train))
    n_groups = len(_resolve_feature_groups(feature_names, feature_groups))
    if args.ablate_ks is not None and max(args.ablate_ks) > n_groups:
        raise ValueError(
            f"[!] --ablate-ks has {max(args.ablate_ks)}, but there are only "
            f"{n_groups} feature groups."
        )
    return {
        "train": (X_train, y_train, feature_names),
        "eval": (X_eval, y_eval),
        "feature_groups": feature_groups,
        "categorical": categorical,
        "train_weight": train_weight,
        "eval_weight": eval_weight,
        "max_samples": args.max_samples,
    }


def _run_ranking(args: argparse.Namespace) -> Dict[str, Any]:
    inputs = _load_cli_inputs(args)
    ranker = FeatureRanker(
        task=args.task,
        group=args.group,
        n_jobs=args.n_jobs,
        verbose=not args.quiet,
        evidence=args.evidence,
        missing=args.missing,
        # Only when given, so a ranker without the option still works here.
        **({"categorical": inputs["categorical"]} if inputs["categorical"] else {}),
        **({} if args.max_categories is None else {"max_categories": args.max_categories}),
    )
    train_sample: Dict[str, Any] = (
        {}
        if args.max_train_samples is None
        else {"max_samples": args.max_train_samples, "random_state": args.random_state}
    )
    X_train, y_train, feature_names = inputs["train"]
    ranker.fit(
        X_train, y_train, feature_names=feature_names, **inputs["train_weight"], **train_sample
    )
    # Only when given, so a ranker without the option still works here.
    max_samples: Dict[str, Any] = (
        {} if inputs["max_samples"] is None else {"max_samples": inputs["max_samples"]}
    )
    names = _scoring_names(args)
    scoring = names[0] if len(names) == 1 else names
    report = ranker.rank_features(
        *inputs["eval"],
        **{**inputs["eval_weight"], **max_samples},
        scoring=scoring,
        feature_groups=inputs["feature_groups"],
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    if args.feature_groups == "auto":
        report["run"]["auto_groups"] = auto_groups_record(args.auto_groups_threshold)
    if args.ablate and not report["consensus"]:
        print(
            "[!] No model beat the dummy baseline, so there is nothing to ablate; "
            "the report is written without an ablation block.",
            file=sys.stderr,
        )
    elif args.ablate:
        X_train, y_train, _ = inputs["train"]
        X_eval, y_eval = inputs["eval"]
        report = ranker.ablate(
            X_train, y_train, X_eval, y_eval, report, ks=args.ablate_ks,
            sample_weight=inputs["train_weight"].get("sample_weight"),
            eval_sample_weight=inputs["eval_weight"].get("sample_weight"),
        )
    return report


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
