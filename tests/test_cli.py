import json
import sys

import numpy as np
import pytest
from sklearn.base import BaseEstimator

import featranker.cli as cli_module
from featranker import FeatureRanker


class RecordingClassifier(BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def clf_ranker_with_models(monkeypatch, models):
    def init_models(self):
        return {"clf": {"linear": models, "tree": {}}, "reg": {}}

    monkeypatch.setattr(FeatureRanker, "init_models", init_models)
    return FeatureRanker(task="clf", group="linear")


def test_load_csv_returns_features_target_and_names(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "a,b,target\n1,2,low\n3,4,high\n",
        encoding="utf-8",
    )

    X, y, names = cli_module._load_csv(path, "target")

    assert names == ["a", "b"]
    assert np.array_equal(X, [[1.0, 2.0], [3.0, 4.0]])
    assert y.tolist() == ["low", "high"]


def test_load_csv_keeps_raw_target_strings(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("a,target\n1,2.5\n3,4.5\n", encoding="utf-8")

    _, y, _ = cli_module._load_csv(path, "target")

    assert y.tolist() == ["2.5", "4.5"]


@pytest.mark.parametrize(
    ("contents", "target", "message"),
    [
        ("a,a,target\n1,2,3\n", "target", "unique"),
        ("a,b\n1,2\n", "target", "target"),
        ("a,target\n1,2\n3\n", "target", "columns"),
        ("a,target\nnope,2\n", "target", "numeric"),
        ("a,target\nnan,2\n", "target", "finite"),
        ("a,target\n1,2\n-inf,3\n", "target", "finite"),
        ("a,target\n", "target", "data row"),
    ],
)
def test_load_csv_rejects_invalid_input(tmp_path, contents, target, message):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli_module._load_csv(path, target)


def test_modern_cli_routes_train_and_eval_data(monkeypatch, tmp_path, capsys):
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    groups_path = tmp_path / "groups.yaml"
    train_path.write_text(
        "a,b,target\n1,2,3\n4,5,6\n7,8,9\n",
        encoding="utf-8",
    )
    eval_path.write_text(
        "a,b,target\n10,11,12\n13,14,15\n",
        encoding="utf-8",
    )
    groups_path.write_text("pair: [a, b]\n", encoding="utf-8")
    calls = {}

    class RecordingRanker:
        def __init__(self, task, group, n_jobs, verbose, evidence, missing):
            calls["constructor"] = (task, group, n_jobs)
            calls["missing"] = missing
            calls["verbose"] = verbose
            calls["evidence"] = evidence

        def fit(self, X, y, feature_names):
            calls["fit"] = (X.copy(), y.copy(), list(feature_names))
            return self

        def rank_features(self, X, y, **kwargs):
            calls["rank"] = (X.copy(), y.copy(), kwargs)
            return {"evaluation_mode": "held_out"}

    monkeypatch.setattr(cli_module, "FeatureRanker", RecordingRanker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker",
            "--task",
            "reg",
            "--group",
            "tree",
            "--train-data",
            str(train_path),
            "--eval-data",
            str(eval_path),
            "--target",
            "target",
            "--scoring",
            "neg_mean_absolute_error",
            "--feature-groups",
            str(groups_path),
            "--n-repeats",
            "20",
            "--random-state",
            "42",
            "--n-jobs",
            "2",
        ],
    )

    assert cli_module.main() == 0

    assert calls["constructor"] == ("reg", "tree", 2)
    assert calls["verbose"] is True
    assert calls["evidence"] is False
    assert calls["missing"] == "error"
    assert np.array_equal(calls["fit"][0], [[1, 2], [4, 5], [7, 8]])
    assert np.array_equal(calls["fit"][1], [3, 6, 9])
    assert calls["fit"][2] == ["a", "b"]
    assert np.array_equal(calls["rank"][0], [[10, 11], [13, 14]])
    assert np.array_equal(calls["rank"][1], [12, 15])
    assert calls["rank"][2] == {
        "scoring": "neg_mean_absolute_error",
        "feature_groups": {"pair": ["a", "b"]},
        "n_repeats": 20,
        "random_state": 42,
    }
    assert json.loads(capsys.readouterr().out)["evaluation_mode"] == "held_out"


def test_modern_cli_requires_both_data_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker",
            "--task",
            "reg",
            "--train-data",
            str(tmp_path / "train.csv"),
            "--target",
            "target",
            "--scoring",
            "r2",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "--eval-data" in capsys.readouterr().err


def test_unknown_group_is_rejected_with_the_choices(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys, "argv", _modern_argv(tmp_path, "--task", "reg", "--scoring", "r2", "--group", "bogus")
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    error = capsys.readouterr().err
    assert "invalid choice: 'bogus'" in error and "fast" in error


def test_group_help_says_what_linear_contains(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli_module._build_parser().parse_args(["--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "linear, discriminant, naive Bayes and kernel models" in help_text


def test_removed_prep_flags_are_rejected(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys, "argv", _modern_argv(tmp_path, "--task", "reg", "--scoring", "r2", "--prep-file", "prep.py")
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "unrecognized arguments: --prep-file" in capsys.readouterr().err


def test_cli_without_arguments_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["featranker"])

    assert cli_module.main() == 0

    assert "--train-data" in capsys.readouterr().out


def test_cli_end_to_end_with_real_ranker_and_registry(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)

    def write_csv(path, n_rows):
        X = rng.normal(size=(n_rows, 3))
        y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=n_rows)
        lines = ["signal,noise_a,noise_b,target"]
        lines += [",".join(map(str, [*row, target])) for row, target in zip(X, y)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(tmp_path / "train.csv", 60)
    write_csv(tmp_path / "eval.csv", 30)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker", "--task", "reg", "--group", "linear",
            "--train-data", str(tmp_path / "train.csv"),
            "--eval-data", str(tmp_path / "eval.csv"),
            "--target", "target", "--scoring", "r2",
            "--n-repeats", "2", "--random-state", "0",
            "--output", str(output),
        ],
    )

    assert cli_module.main() == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation_mode"] == "held_out"
    assert isinstance(report["baseline_score"], float)
    assert report["consensus"]
    assert report["consensus"][0]["feature_group"] == "signal"
    assert not [f for f in report["failures"] if f["stage"] in {"fit", "ranking"}]


def test_cli_group_fast_reports_the_fast_set(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)

    def write_csv(path, n_rows):
        X = rng.normal(size=(n_rows, 2))
        y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=n_rows)
        lines = ["signal,noise,target"]
        lines += [",".join(map(str, [*row, target])) for row, target in zip(X, y)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(tmp_path / "train.csv", 60)
    write_csv(tmp_path / "eval.csv", 30)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker", "--task", "reg", "--group", "fast", "--quiet",
            "--train-data", str(tmp_path / "train.csv"),
            "--eval-data", str(tmp_path / "eval.csv"),
            "--target", "target", "--scoring", "r2", "--n-repeats", "2",
            "--random-state", "0", "--output", str(output),
        ],
    )

    assert cli_module.main() == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["models"]) == {
        "ridge_regression", "svr_regression", "decision_tree_regressor",
        "random_forest_regressor", "hist_gradient_boosting_regressor",
    }


@pytest.mark.filterwarnings("ignore:shap is not installed:UserWarning")
def test_cli_evidence_flag_writes_evidence(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)

    def write_csv(path, n_rows):
        X = rng.normal(size=(n_rows, 2))
        y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=n_rows)
        lines = ["signal,noise,target"]
        lines += [",".join(map(str, [*row, target])) for row, target in zip(X, y)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(tmp_path / "train.csv", 60)
    write_csv(tmp_path / "eval.csv", 30)
    argv = [
        "featranker", "--task", "reg", "--group", "linear", "--quiet",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--scoring", "r2", "--n-repeats", "2",
        "--random-state", "0",
    ]

    reports = {}
    for flag in ([], ["--evidence"]):
        output = tmp_path / f"report{len(flag)}.json"
        monkeypatch.setattr(sys, "argv", [*argv, *flag, "--output", str(output)])
        assert cli_module.main() == 0
        reports[len(flag)] = json.loads(output.read_text(encoding="utf-8"))

    assert "evidence" not in reports[0]
    assert "correlation" in reports[1]["evidence"]
    assert "train_score" in next(iter(reports[1]["models"].values()))


def test_cli_quiet_writes_report_without_progress(monkeypatch, tmp_path, capsys):
    train, evaluation = tmp_path / "train.csv", tmp_path / "eval.csv"
    rows = "\n".join(f"{i},{i % 3},{2 * i}" for i in range(8))
    train.write_text("a,b,target\n" + rows + "\n", encoding="utf-8")
    evaluation.write_text("a,b,target\n" + rows + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker", "--task", "reg", "--group", "linear",
            "--train-data", str(train), "--eval-data", str(evaluation),
            "--target", "target", "--scoring", "r2", "--n-repeats", "1",
            "--quiet", "--output", str(output),
        ],
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli_module.main() == 0

    err = capsys.readouterr().err
    assert "[*]" not in err and "it/s" not in err
    assert json.loads(output.read_text(encoding="utf-8"))["consensus"]


def test_modern_cli_rejects_mismatched_feature_order(
    monkeypatch, tmp_path, capsys
):
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    train_path.write_text("a,b,target\n1,2,3\n", encoding="utf-8")
    eval_path.write_text("b,a,target\n2,1,3\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker",
            "--task",
            "reg",
            "--train-data",
            str(train_path),
            "--eval-data",
            str(eval_path),
            "--target",
            "target",
            "--scoring",
            "r2",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "names and order" in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore:No model beat the dummy baseline")
def test_modern_cli_accepts_eval_class_subset(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli_module,
        "FeatureRanker",
        lambda task, group, n_jobs, verbose, evidence, missing: clf_ranker_with_models(
            monkeypatch, {"ok": RecordingClassifier()}
        ),
    )
    argv = _modern_argv(
        tmp_path,
        "--task",
        "clf",
        "--scoring",
        "accuracy",
        train="a,target\n1,1\n2,other\n",
        eval_="a,target\n3,1\n4,1\n",
    )
    monkeypatch.setattr(sys, "argv", argv)

    assert cli_module.main() == 0
    assert json.loads(capsys.readouterr().out)["evaluation_mode"] == "held_out"


def test_load_csv_strips_utf8_bom(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_text("target,a\n1,2\n3,4\n", encoding="utf-8-sig")

    _, y, names = cli_module._load_csv(path, "target")

    assert names == ["a"]
    assert y.tolist() == ["1", "3"]


def test_load_csv_skips_blank_lines(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("a,target\n1,2\n\n3,4\n\n", encoding="utf-8")

    X, y, _ = cli_module._load_csv(path, "target")

    assert np.array_equal(X, [[1.0], [3.0]])
    assert y.tolist() == ["2", "4"]


def _modern_argv(tmp_path, *extra, train="a,target\n1,0\n2,1\n", eval_="a,target\n3,0\n4,1\n"):
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    train_path.write_text(train, encoding="utf-8")
    eval_path.write_text(eval_, encoding="utf-8")
    return [
        "featranker",
        "--train-data",
        str(train_path),
        "--eval-data",
        str(eval_path),
        "--target",
        "target",
        *extra,
    ]


class NoFitRanker:
    def __init__(self, task, group, n_jobs, verbose):
        pass

    def fit(self, X, y, feature_names):
        raise AssertionError("fit must not run on invalid CLI input")


def test_modern_cli_rejects_non_numeric_regression_eval_target_before_fit(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli_module, "FeatureRanker", NoFitRanker)
    argv = _modern_argv(
        tmp_path,
        "--task",
        "reg",
        "--scoring",
        "r2",
        eval_="a,target\n3,0\n4,NA\n",
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "target" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "groups", "message"),
    [
        (["--scoring", "neg_mae"], None, "neg_mae"),
        (["--scoring", "accuracy", "--n-repeats", "0"], None, "n_repeats"),
        (["--scoring", "accuracy"], "g: [a, missing]\n", "unknown feature"),
    ],
)
def test_modern_cli_validates_options_before_fit(
    monkeypatch, tmp_path, capsys, extra, groups, message
):
    monkeypatch.setattr(cli_module, "FeatureRanker", NoFitRanker)
    if groups is not None:
        groups_path = tmp_path / "groups.yaml"
        groups_path.write_text(groups, encoding="utf-8")
        extra = [*extra, "--feature-groups", str(groups_path)]
    monkeypatch.setattr(
        sys, "argv", _modern_argv(tmp_path, "--task", "clf", *extra)
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert message in capsys.readouterr().err


def test_modern_cli_requires_task(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys, "argv", _modern_argv(tmp_path, "--scoring", "accuracy")
    )

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "--task" in capsys.readouterr().err


def test_cli_rejects_unseen_longer_eval_label_with_full_name(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        cli_module,
        "FeatureRanker",
        lambda task, group, n_jobs, verbose, evidence, missing: clf_ranker_with_models(
            monkeypatch, {"ok": RecordingClassifier()}
        ),
    )
    argv = _modern_argv(
        tmp_path,
        "--task",
        "clf",
        "--scoring",
        "accuracy",
        train="a,target\n1,yes\n2,no\n",
        eval_="a,target\n3,yesss\n4,no\n",
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "yesss" in capsys.readouterr().err


def test_cli_rejects_non_finite_eval_feature_before_fit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli_module, "FeatureRanker", NoFitRanker)
    argv = _modern_argv(
        tmp_path, "--task", "reg", "--scoring", "r2", eval_="a,target\nnan,0\n4,1\n"
    )
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "finite" in capsys.readouterr().err


def run_cli_with_pair(monkeypatch, tmp_path, groups_argument):
    rng = np.random.default_rng(0)

    def write_csv(path, n_rows):
        e = rng.normal(size=n_rows)
        X = np.column_stack([e, e + 0.05 * rng.normal(size=n_rows), rng.normal(size=n_rows)])
        lines = ["e,f,g,target"]
        lines += [",".join(map(str, [*row, 2 * row[0]])) for row in X]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(tmp_path / "train.csv", 60)
    write_csv(tmp_path / "eval.csv", 30)
    output = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "featranker", "--task", "reg", "--group", "linear", "--quiet",
            "--train-data", "train.csv", "--eval-data", "eval.csv",
            "--target", "target", "--scoring", "r2", "--n-repeats", "2",
            "--random-state", "0", "--feature-groups", groups_argument,
            "--output", str(output),
        ],
    )

    assert cli_module.main() == 0

    return json.loads(output.read_text(encoding="utf-8"))["feature_groups"]


def test_cli_feature_groups_auto_groups_the_pair(monkeypatch, tmp_path):
    groups = run_cli_with_pair(monkeypatch, tmp_path, "auto")

    assert groups == {"e+f": ["e", "f"], "g": ["g"]}


def test_cli_feature_groups_yaml_path_still_works(monkeypatch, tmp_path):
    (tmp_path / "groups.yaml").write_text("pair: [e, g]\n", encoding="utf-8")

    groups = run_cli_with_pair(monkeypatch, tmp_path, "groups.yaml")

    assert groups == {"pair": ["e", "g"], "f": ["f"]}


def test_cli_dot_slash_auto_reads_a_file_named_auto(monkeypatch, tmp_path):
    (tmp_path / "auto").write_text("pair: [f, g]\n", encoding="utf-8")

    groups = run_cli_with_pair(monkeypatch, tmp_path, "./auto")

    assert groups == {"pair": ["f", "g"], "e": ["e"]}
