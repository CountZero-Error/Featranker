"""1.21 memory-limits: lower-peak copies, category cap, training subsample."""

import json
import pickle
import sys
import tracemalloc

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

import featranker.cli as cli_module
import featranker.evidence as evidence_module
from featranker import ConsensusSelector, FeatureRanker
from featranker.importance import _labels_fingerprint, _resolve_splits, _shuffle_index, _with_one_hot, _with_shadows

# -- the previous implementations, kept to show the new ones equal them ---------

MIN_PAIR_ROWS = evidence_module.MIN_PAIR_ROWS


def old_with_shadows(X_values, seed, mode="joint", shuffle="rows", labels=None):
    rng = np.random.default_rng(seed)
    n_rows = len(X_values)
    if mode == "independent":
        shadow = np.column_stack(
            [
                X_values[_shuffle_index(rng, n_rows, shuffle, labels), j]
                for j in range(X_values.shape[1])
            ]
        )
    else:
        shadow = X_values[_shuffle_index(rng, n_rows, shuffle, labels)]
    return np.hstack([X_values, shadow])


def old_spearman(X):
    if np.isnan(X).any():
        present = ~np.isnan(X)
        complete = present.all(axis=0)
        rho = np.full((X.shape[1], X.shape[1]), np.nan)
        if complete.any():
            rho[np.ix_(complete, complete)] = old_spearman(X[:, complete])
        for first in range(X.shape[1]):
            for second in range(first, X.shape[1]):
                if complete[first] and complete[second]:
                    continue
                rows = present[:, first] & present[:, second]
                if rows.sum() >= MIN_PAIR_ROWS:
                    pair = X[rows][:, [first, second]]
                    rho[first, second] = rho[second, first] = old_spearman(pair)[0, 1]
        return rho
    centered = rankdata(X, axis=0) - (X.shape[0] + 1) / 2
    norm = np.sqrt((centered**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(centered.T @ centered / np.outer(norm, norm), -1.0, 1.0)


# -- 1. shadows and Spearman equal the previous construction ---------------------

SHAPES = [(1, 1), (7, 3), (50, 1), (200, 5), (1000, 12), (3000, 2)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", ["joint", "independent"])
@pytest.mark.parametrize("shuffle", ["rows", "within_groups", "circular"])
@pytest.mark.parametrize("seed", [0, 1, 123456])
def test_shadows_equal_previous_construction(shape, mode, shuffle, seed):
    n_rows = shape[0]
    if shuffle == "circular" and n_rows < 2:
        pytest.skip("a circular shift needs two rows")
    X = np.random.default_rng(5).normal(size=shape)
    labels = None
    if shuffle == "within_groups":
        labels = np.random.default_rng(2).integers(0, max(2, n_rows // 5), n_rows)

    new = _with_shadows(X, seed, mode, shuffle, labels)
    old = old_with_shadows(X, seed, mode, shuffle, labels)

    assert new.dtype == old.dtype and new.shape == old.shape
    assert np.array_equal(new, old)
    assert np.array_equal(new[:, : shape[1]], X)


@pytest.mark.parametrize("mode", ["joint", "independent"])
def test_shadows_equal_previous_construction_for_other_dtypes(mode):
    base = np.random.default_rng(1).normal(size=(40, 4))
    arrays = [
        base.astype(np.float32),
        (base * 10).astype(np.int64),
        base.astype(object),
        np.asfortranarray(base),
        base[:, ::2],
    ]
    for X in arrays:
        new, old = _with_shadows(X, 4, mode), old_with_shadows(X, 4, mode)
        assert new.dtype == old.dtype
        assert np.array_equal(new, old)


def test_shadows_do_not_alias_the_input():
    X = np.random.default_rng(1).normal(size=(30, 3))
    before = X.copy()
    out = _with_shadows(X, 0)
    out[:] = 0.0
    assert np.array_equal(X, before)


def _peak(build, *args):
    tracemalloc.start()
    try:
        build(*args)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_shadows_peak_memory_is_below_the_previous_gather_plus_hstack():
    X = np.random.default_rng(1).normal(size=(20000, 50))  # 8 MB

    new, old = _peak(_with_shadows, X, 0), _peak(old_with_shadows, X, 0)

    assert old > 2.9 * X.nbytes  # gather (1X) plus hstack (2X)
    assert new < 2.3 * X.nbytes  # the output (2X) and a small chunk


def _with_missing(X, seed, share):
    X = X.copy()
    X[np.random.default_rng(seed).random(X.shape) < share] = np.nan
    return X


def _spearman_inputs():
    rng = np.random.default_rng(0)
    constant = rng.normal(size=(120, 66))
    constant[:, 0] = 3.0
    constant[:, 65] = -1.0
    correlated = rng.normal(size=(500, 70))
    correlated[:, 1] = correlated[:, 0] * 2 + 0.01 * rng.normal(size=500)
    return {
        "wide": rng.normal(size=(300, 70)),
        "odd-width": rng.normal(size=(257, 129)),
        "one-column": rng.normal(size=(100, 1)),
        "two-columns": rng.normal(size=(100, 2)),
        "long": rng.normal(size=(5000, 9)),
        "ties": rng.integers(0, 4, size=(400, 66)).astype(float),
        "float32": rng.normal(size=(200, 67)).astype(np.float32),
        "int64": rng.integers(-50, 50, size=(300, 65)),
        "constant-columns": constant,
        "correlated": correlated,
        "nan": _with_missing(rng.normal(size=(200, 70)), 1, 0.1),
        "nan-few-columns": _with_missing(rng.normal(size=(150, 5)), 2, 0.3),
        "nan-and-constant": _with_missing(constant, 3, 0.02),
    }


SPEARMAN_INPUTS = _spearman_inputs()


@pytest.mark.parametrize("name", list(SPEARMAN_INPUTS))
def test_spearman_equals_previous_implementation_bit_for_bit(name):
    X = SPEARMAN_INPUTS[name]

    new, old = evidence_module._spearman_matrix(X), old_spearman(X)

    assert new.shape == old.shape
    assert np.array_equal(new, old, equal_nan=True)


@pytest.mark.parametrize("width", [1, 2, 63, 64, 65, 66, 128, 129, 130, 1000])
@pytest.mark.parametrize("n_rows", [10, 20_000, 3_000_000])
def test_spearman_column_blocks_cover_the_columns_within_the_cell_budget(width, n_rows):
    blocks = evidence_module._column_blocks(width, n_rows)

    assert blocks[0][0] == 0 and blocks[-1][1] == width
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))
    assert all(stop > start for start, stop in blocks)
    assert all(
        (stop - start) * n_rows <= max(n_rows, evidence_module.SPEARMAN_BLOCK_CELLS)
        for start, stop in blocks
    )


def test_spearman_ranks_a_narrow_matrix_in_one_block():
    assert evidence_module._column_blocks(10, 20_000) == [(0, 10)]


@pytest.mark.parametrize("shape", [(1_000_000, 3), (400_000, 65)])
def test_spearman_equals_previous_implementation_bit_for_bit_at_large_n(shape):
    X = np.random.default_rng(0).normal(size=shape)
    X[:, 0] = np.round(X[:, 0], 1)  # ties

    assert np.array_equal(evidence_module._spearman_matrix(X), old_spearman(X), equal_nan=True)


def test_spearman_does_not_change_its_input():
    X = np.random.default_rng(0).normal(size=(100, 70))
    before = X.copy()
    evidence_module._spearman_matrix(X)
    assert np.array_equal(X, before)


@pytest.mark.parametrize("width", [10, 64, 500])
def test_spearman_peak_memory_is_never_above_the_previous_implementation(width):
    X = np.random.default_rng(1).normal(size=(20000, width))

    new = _peak(evidence_module._spearman_matrix, X)
    old = _peak(old_spearman, X)

    assert new <= 1.001 * old  # a narrow matrix is one block: the same peak, give or take a list
    if width >= 64:  # several blocks: little beyond the output is full size
        assert new < 0.5 * old


# -- 2. max_categories -----------------------------------------------------------


def cat_frame(levels, n=600, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, levels, n)
    return pd.DataFrame(
        {"x": rng.normal(size=n), "c": codes.astype(str), "z": rng.normal(size=n)}
    ), (codes % 3).astype(float) + 0.1 * rng.normal(size=n)


def cat_ranker(**kwargs):
    return FeatureRanker("reg", "fast", verbose=False, categorical=["c"], **kwargs)


def linear_models(ranker):
    names = set(ranker.models["reg"]["linear"])
    return [model for name, model in ranker.trained_models.items() if name in names]


def encoder_of(model):
    return model.steps[0][1].named_transformers_["cat"]


@pytest.mark.parametrize("value", [1, 0, -3, 2.5, True, "5", float("nan"), [3]])
def test_invalid_max_categories_raises(value):
    with pytest.raises(ValueError, match="max_categories"):
        FeatureRanker("reg", "fast", verbose=False, max_categories=value)


@pytest.mark.parametrize("value", [None, 2, 20, np.int64(5)])
def test_valid_max_categories_is_accepted(value):
    FeatureRanker("reg", "fast", verbose=False, max_categories=value)


def test_a_cardinality_1000_column_is_capped_for_every_builtin_linear_model():
    X, y = cat_frame(1000, n=3000)
    ranker = cat_ranker(max_categories=20).fit(X, y)

    models = linear_models(ranker)

    assert models
    for model in models:
        assert len(encoder_of(model).get_feature_names_out()) == 20
    uncapped = cat_ranker().fit(X, y)
    assert all(len(encoder_of(m).get_feature_names_out()) > 900 for m in linear_models(uncapped))


def test_shadow_categorical_columns_are_capped_too():
    X, y = cat_frame(100)
    ranker = cat_ranker(max_categories=10, shadows=True, shadow_seed=0).fit(X, y)

    for model in linear_models(ranker):
        assert len(encoder_of(model).get_feature_names_out()) == 20  # column and shadow


def test_unseen_levels_fall_into_the_infrequent_column():
    X, y = cat_frame(50)
    ranker = cat_ranker(max_categories=5).fit(X, y)

    encoder = encoder_of(linear_models(ranker)[0])
    row = encoder.transform(np.array([[1e6]])).toarray()

    assert row.sum() == 1 and row[0, -1] == 1  # the infrequent category is last
    assert np.array_equal(
        encoder.transform(np.array([[1e6]])).toarray(), encoder.transform(np.array([[2e6]])).toarray()
    )


def test_unseen_eval_levels_rank_without_failures_when_capped():
    X, y = cat_frame(50)
    ranker = cat_ranker(max_categories=5).fit(X.iloc[:400], y[:400])
    held_out = X.iloc[400:].assign(c="never-seen")

    report = ranker.rank_features(held_out, y[400:], scoring="r2", n_repeats=2, random_state=0)

    assert report["failures"] == []
    assert report["categorical"]["unseen_eval"] == {"c": 200}


def test_default_one_hot_pipeline_is_unchanged():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = _with_one_hot(make_pipeline(StandardScaler(), Ridge()), [1])
    encoder = pipeline.steps[0][1].named_transformers_["cat"] if hasattr(
        pipeline.steps[0][1], "named_transformers_"
    ) else pipeline.steps[0][1].transformers[0][1]

    params = encoder.get_params()
    assert params["handle_unknown"] == "ignore" and params["max_categories"] is None
    capped = _with_one_hot(make_pipeline(StandardScaler(), Ridge()), [1], 7)
    capped_params = capped.steps[0][1].transformers[0][1].get_params()
    assert capped_params["max_categories"] == 7
    assert capped_params["handle_unknown"] == "infrequent_if_exist"


def _stable(report):
    """The scores and rankings, without the model parameters (a cap is one of them)."""
    models = {
        name: {k: v for k, v in model.items() if k != "params"}
        for name, model in report["models"].items()
    }
    return json.dumps(
        {
            "models": models,
            **{k: report[k] for k in ("consensus", "baseline_score", "failures", "categorical")},
        },
        sort_keys=True,
    )


def test_default_and_explicit_none_reports_are_identical_and_have_no_new_keys():
    X, y = cat_frame(8)

    def rank(**kwargs):
        ranker = cat_ranker(**kwargs).fit(X.iloc[:400], y[:400])
        return ranker.rank_features(X.iloc[400:], y[400:], scoring="r2", n_repeats=2, random_state=0)

    default, explicit = rank(), rank(max_categories=None)

    assert _stable(default) == _stable(explicit)
    assert default["run"] == explicit["run"]
    assert "max_categories" not in default["run"] and "n_train_rows" not in default["run"]


def test_a_cap_above_the_level_count_changes_no_score():
    X, y = cat_frame(8)

    def rank(**kwargs):
        ranker = cat_ranker(**kwargs).fit(X.iloc[:400], y[:400])
        return ranker.rank_features(X.iloc[400:], y[400:], scoring="r2", n_repeats=2, random_state=0)

    default, capped = rank(), rank(max_categories=20)

    assert _stable(default) == _stable(capped)  # no infrequent level: unknown stays all zeros
    assert capped["run"]["max_categories"] == 20


def test_run_records_max_categories_only_when_set_and_in_resampled_reports():
    X, y = cat_frame(8)
    ranker = cat_ranker(max_categories=4)

    held_out = ranker.fit(X.iloc[:400], y[:400]).rank_features(
        X.iloc[400:], y[400:], scoring="r2", n_repeats=2, random_state=0
    )
    resampled = ranker.rank_resampled(
        X, y, scoring="r2", cv=2, n_repeats=2, random_state=0
    )

    assert held_out["run"]["max_categories"] == 4
    assert resampled["run"]["max_categories"] == 4
    assert all(s["report"]["run"]["max_categories"] == 4 for s in resampled["splits"])
    assert "max_categories" not in cat_ranker().rank_resampled(
        X, y, scoring="r2", cv=2, n_repeats=2, random_state=0
    )["run"]
    json.dumps(dict(held_out), allow_nan=False)


def test_numpy_integer_max_categories_is_stored_as_a_plain_int():
    X, y = cat_frame(8)
    ranker = cat_ranker(max_categories=np.int64(4)).fit(X.iloc[:400], y[:400])
    report = ranker.rank_features(X.iloc[400:], y[400:], scoring="r2", n_repeats=2, random_state=0)
    assert type(report["run"]["max_categories"]) is int
    json.dumps(dict(report), allow_nan=False)


def test_ablation_refits_with_the_same_cap():
    from featranker.ablation import _sub_ranker

    ranker = cat_ranker(max_categories=6)
    assert _sub_ranker(ranker, ["linear_regression"], ["x", "c"]).max_categories == 6


def test_ranker_pickled_before_max_categories_loads_with_none():
    ranker = cat_ranker()
    state = ranker.__getstate__()
    del state["max_categories"]
    loaded = FeatureRanker.__new__(FeatureRanker)
    loaded.__setstate__(state)
    X, y = cat_frame(8)
    assert loaded.max_categories is None
    assert loaded.fit(X, y).is_fitted_
    assert pickle.loads(pickle.dumps(cat_ranker(max_categories=5))).max_categories == 5


def test_selector_passes_max_categories_and_loads_old_pickles():
    X, y = cat_frame(30)
    selector = ConsensusSelector(
        "reg", "r2", k=1, group="fast", categorical=["c"], cv=2, n_repeats=2,
        random_state=0, max_categories=4,
    )
    assert clone(selector).get_params()["max_categories"] == 4

    selector.fit(X, y)

    assert selector.report_["run"]["max_categories"] == 4
    state = ConsensusSelector("reg", "r2", k=1).__getstate__()
    del state["max_categories"]
    del state["max_train_samples"]
    loaded = ConsensusSelector.__new__(ConsensusSelector)
    loaded.__setstate__(state)
    assert loaded.max_categories is None and loaded.max_train_samples is None


def write_cat_csv(path, n, seed):
    rng = np.random.default_rng(seed)
    lines = ["signal,city,target"]
    for _ in range(n):
        s, level = rng.normal(), rng.integers(0, 8)
        lines.append(f"{s:.5f},L{level},{2 * s + 0.1 * rng.normal():.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli_argv(tmp_path, *extra):
    write_cat_csv(tmp_path / "train.csv", 80, 0)
    write_cat_csv(tmp_path / "eval.csv", 40, 1)
    return [
        "featranker", "--task", "reg", "--group", "fast",
        "--train-data", str(tmp_path / "train.csv"),
        "--eval-data", str(tmp_path / "eval.csv"),
        "--target", "target", "--scoring", "r2", "--n-repeats", "2",
        "--random-state", "0", "--quiet", "--categorical", "city", *extra,
    ]


def run_cli(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)
    assert cli_module.main() == 0
    return json.loads(capsys.readouterr().out)


def test_cli_max_categories_and_max_train_samples(monkeypatch, capsys, tmp_path):
    report = run_cli(
        monkeypatch, capsys,
        cli_argv(tmp_path, "--max-categories", "3", "--max-train-samples", "0.5"),
    )

    assert report["run"]["max_categories"] == 3
    assert report["run"]["n_train_rows"] == 40


def test_cli_train_sample_as_an_int_and_defaults_leave_no_keys(monkeypatch, capsys, tmp_path):
    report = run_cli(monkeypatch, capsys, cli_argv(tmp_path, "--max-train-samples", "30"))
    assert report["run"]["n_train_rows"] == 30

    plain = run_cli(monkeypatch, capsys, cli_argv(tmp_path))
    assert "n_train_rows" not in plain["run"] and "max_categories" not in plain["run"]


@pytest.mark.parametrize(
    "option, text",
    [("--max-categories", "1"), ("--max-categories", "abc"),
     ("--max-train-samples", "0"), ("--max-train-samples", "1.5"),
     ("--max-train-samples", "abc")],
)
def test_cli_rejects_bad_values(monkeypatch, capsys, tmp_path, option, text):
    monkeypatch.setattr(sys, "argv", cli_argv(tmp_path, option, text))

    with pytest.raises(SystemExit, match="2"):
        cli_module.main()

    assert "max" in capsys.readouterr().err.lower()


# -- 3. fit(max_samples=) ---------------------------------------------------------

N = 400
RNG = np.random.default_rng(0)
X = RNG.normal(size=(N, 3))
Y = 3 * X[:, 0] + X[:, 1] + 0.1 * RNG.normal(size=N)  # distinct values: row identity
NAMES = ["a", "b", "c"]
FITS = []


class RecordingRidge(Ridge):
    """Keeps the rows each fit saw, in FITS."""

    def fit(self, X_, y_, sample_weight=None):
        FITS.append((np.array(X_), np.array(y_), sample_weight))
        return super().fit(X_, y_, sample_weight=sample_weight)


@pytest.fixture(autouse=True)
def clear_fits():
    FITS.clear()


def recording_ranker(**kwargs):
    return FeatureRanker("reg", models={"ridge": RecordingRidge()}, verbose=False, **kwargs)


def row_ids(y_values):
    lookup = {float(v): i for i, v in enumerate(Y)}
    return np.array([lookup[float(v)] for v in y_values])


def rank(ranker, **kwargs):
    kwargs = {"n_repeats": 2, "random_state": 0, "scoring": "r2", **kwargs}
    return ranker.rank_features(X[300:], Y[300:], **kwargs)


def test_every_model_is_fit_on_the_share_and_the_run_records_it():
    ranker = recording_ranker().fit(
        X, Y, feature_names=NAMES, max_samples=0.25, random_state=0
    )

    assert [len(y_) for _, y_, _ in FITS] == [100]
    assert rank(ranker)["run"]["n_train_rows"] == 100


def test_a_count_larger_than_the_data_uses_every_row():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=10_000)
    assert [len(y_) for _, y_, _ in FITS] == [N]
    assert rank(ranker)["run"]["n_train_rows"] == N


@pytest.mark.parametrize("value", [0, -1, 1.5, 0.0, True, "5", float("nan"), [3]])
def test_invalid_max_samples_raises_before_any_fit(value):
    ranker = recording_ranker()
    with pytest.raises(ValueError, match="max_samples"):
        ranker.fit(X, Y, feature_names=NAMES, max_samples=value)
    assert FITS == [] and not ranker.is_fitted_


def test_rows_are_the_sorted_draw_of_the_train_stream():
    recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=60, random_state=7)

    expected = np.sort(np.random.default_rng([7, 3]).choice(N, 60, replace=False))
    assert np.array_equal(row_ids(FITS[0][1]), expected)
    assert np.array_equal(FITS[0][0], X[expected])


def test_reproducible_and_seed_dependent():
    def fit_rows(seed):
        FITS.clear()
        recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=60, random_state=seed)
        return row_ids(FITS[0][1])

    assert np.array_equal(fit_rows(3), fit_rows(3))
    assert not np.array_equal(fit_rows(3), fit_rows(4))


def test_without_a_seed_the_draw_uses_fresh_entropy():
    draws = []
    for _ in range(3):
        FITS.clear()
        recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=60)
        draws.append(tuple(row_ids(FITS[0][1])))
        assert len(draws[-1]) == 60
    assert len(set(draws)) > 1


def test_same_seed_gives_equal_reports():
    def report():
        ranker = recording_ranker(shadows=True, shadow_seed=1)
        ranker.fit(X, Y, feature_names=NAMES, max_samples=0.5, random_state=2)
        return rank(ranker)

    first, again = report(), report()
    assert first["models"] == again["models"]
    assert first["run"] == again["run"]
    assert first["consensus"] == again["consensus"]


def test_default_report_is_unchanged_and_has_no_train_key():
    def report(**kwargs):
        ranker = recording_ranker().fit(X, Y, feature_names=NAMES, **kwargs)
        return rank(ranker)

    plain, explicit = report(), report(max_samples=None, random_state=5)

    assert plain["models"] == explicit["models"] and plain["run"] == explicit["run"]
    assert "n_train_rows" not in plain["run"]


def test_within_groups_fits_whole_groups():
    labels = np.repeat(np.arange(40), 10)
    ranker = recording_ranker(shuffle="within_groups", shadows=True, shadow_seed=0)
    ranker.fit(
        X, Y, feature_names=NAMES, shuffle_groups=labels, max_samples=95, random_state=1
    )

    rows = row_ids(FITS[0][1])
    kept, counts = np.unique(labels[rows], return_counts=True)
    assert set(counts) == {10} and len(kept) == 10
    assert list(rows) == sorted(rows)
    assert rank_with_groups(ranker, labels)["run"]["n_train_rows"] == 100


def rank_with_groups(ranker, labels):
    return ranker.rank_features(
        X[300:], Y[300:], scoring="r2", n_repeats=2, random_state=0,
        shuffle_groups=labels[300:],
    )


def test_circular_keeps_row_order():
    ranker = recording_ranker(shuffle="circular", shadows=True, shadow_seed=0)
    ranker.fit(X, Y, feature_names=NAMES, max_samples=80, random_state=1)

    rows = row_ids(FITS[0][1])
    assert len(rows) == 80 and list(rows) == sorted(rows)
    assert rank(ranker)["run"]["n_train_rows"] == 80


def test_sampled_rows_with_zero_total_weight_raise():
    expected = np.sort(np.random.default_rng([5, 3]).choice(N, 40, replace=False))
    weights = np.ones(N)
    weights[expected] = 0.0
    ranker = recording_ranker()

    with pytest.raises(ValueError, match="zero sample_weight"):
        ranker.fit(
            X, Y, feature_names=NAMES, sample_weight=weights, max_samples=40, random_state=5
        )

    assert FITS == [] and not ranker.is_fitted_


def test_sample_weights_follow_the_sampled_rows():
    weights = np.random.default_rng(3).uniform(0.5, 2, N)
    recording_ranker().fit(
        X, Y, feature_names=NAMES, sample_weight=weights, max_samples=50, random_state=9
    )

    rows = row_ids(FITS[0][1])
    assert np.array_equal(FITS[0][2], weights[rows])


def test_evidence_keeps_the_subsample_not_the_full_data():
    ranker = recording_ranker(evidence=True, shadows=True, shadow_seed=0)
    ranker.fit(X, Y, feature_names=NAMES, max_samples=100, random_state=4)

    expected = np.sort(np.random.default_rng([4, 3]).choice(N, 100, replace=False))
    assert ranker.train_X_.shape == (100, 6)
    assert np.array_equal(ranker.train_X_[:, :3], X[expected])
    assert np.array_equal(ranker.train_y_, Y[expected])


def test_fingerprint_covers_the_full_data_passed_to_fit():
    sampled = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=50, random_state=1)
    full = recording_ranker().fit(X, Y, feature_names=NAMES)

    assert rank(sampled)["run"]["train_data_sha256"] == rank(full)["run"]["train_data_sha256"]


def test_weight_and_label_fingerprints_cover_the_full_arrays():
    weights = np.random.default_rng(3).uniform(0.5, 2, N)
    labels = np.repeat(np.arange(40), 10)

    def run(**kwargs):
        ranker = recording_ranker(shuffle="within_groups")
        ranker.fit(X, Y, feature_names=NAMES, sample_weight=weights, shuffle_groups=labels, **kwargs)
        return ranker.rank_features(
            X[300:], Y[300:], scoring="r2", n_repeats=2, random_state=0,
            shuffle_groups=labels[300:],
        )["run"]

    sampled, full = run(max_samples=100, random_state=0), run()
    for key in ("train_weights_sha256", "train_shuffle_groups_sha256", "train_data_sha256"):
        assert sampled[key] == full[key]


def test_ablate_accepts_the_report_and_refits_on_the_same_rows():
    ranker = recording_ranker()
    ranker.fit(X, Y, feature_names=NAMES, max_samples=100, random_state=6)
    report = rank(ranker)
    FITS.clear()

    ablated = ranker.ablate(X, Y, X[300:], Y[300:], report, ks=[1])

    assert ablated["ablation"]["curve"][0]["scores"]["ridge"] is not None
    expected = np.sort(np.random.default_rng([6, 3]).choice(N, 100, replace=False))
    assert FITS and np.array_equal(row_ids(FITS[0][1]), expected)


def test_ablate_still_refuses_other_training_data():
    ranker = recording_ranker()
    ranker.fit(X, Y, feature_names=NAMES, max_samples=100, random_state=6)
    report = rank(ranker)

    with pytest.raises(ValueError, match="training data"):
        ranker.ablate(X[:200], Y[:200], X[300:], Y[300:], report, ks=[1])


def test_fit_resets_the_subsample_when_refit_without_it():
    ranker = recording_ranker()
    ranker.fit(X, Y, feature_names=NAMES, max_samples=100, random_state=6)
    ranker.fit(X, Y, feature_names=NAMES)

    assert "n_train_rows" not in rank(ranker)["run"]


def test_ranker_pickled_before_train_subsample_loads():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES)
    state = ranker.__getstate__()
    for key in ("train_n_rows_", "train_rows_"):
        state.pop(key, None)
    loaded = FeatureRanker.__new__(FeatureRanker)
    loaded.__setstate__(state)

    assert "n_train_rows" not in rank(loaded)["run"]


def test_rank_resampled_subsamples_each_splits_training_rows():
    ranker = recording_ranker()

    report = ranker.rank_resampled(
        X, Y, scoring="r2", cv=4, n_repeats=2, random_state=0, feature_names=NAMES,
        max_train_samples=0.5,
    )

    assert report["max_train_samples"] == 0.5
    assert [split["n_train"] for split in report["splits"]] == [300] * 4
    for split in report["splits"]:
        assert split["report"]["run"]["n_train_rows"] == 150
    assert sorted({len(y_) for _, y_, _ in FITS}) == [150]


def test_rank_resampled_uses_the_splits_seed_for_the_draw():
    ranker = recording_ranker()
    ranker.rank_resampled(
        X, Y, scoring="r2", cv=2, n_repeats=2, random_state=10, feature_names=NAMES,
        max_train_samples=40,
    )

    train = _resolve_splits(2, X, Y, None, "reg", 10)[1][0]  # split 1 seeds with 10 + 1
    expected = train[np.sort(np.random.default_rng([11, 3]).choice(len(train), 40, replace=False))]
    assert np.array_equal(row_ids(FITS[1][1]), expected)
    assert {len(y_) for _, y_, _ in FITS} == {40}


def test_rank_resampled_default_has_no_train_keys():
    report = recording_ranker().rank_resampled(
        X, Y, scoring="r2", cv=2, n_repeats=2, random_state=0, feature_names=NAMES
    )
    assert "max_train_samples" not in report
    assert all("n_train_rows" not in s["report"]["run"] for s in report["splits"])


@pytest.mark.parametrize("value", [0, 1.5, True, "5"])
def test_rank_resampled_rejects_a_bad_train_sample_before_fitting(value):
    with pytest.raises(ValueError, match="max_samples"):
        recording_ranker().rank_resampled(
            X, Y, scoring="r2", cv=2, feature_names=NAMES, max_train_samples=value
        )
    assert FITS == []


def test_numpy_scalar_train_sample_is_stored_as_a_plain_number():
    report = recording_ranker().rank_resampled(
        X, Y, scoring="r2", cv=2, n_repeats=2, random_state=0, feature_names=NAMES,
        max_train_samples=np.float32(0.5),
    )
    assert type(report["max_train_samples"]) is float
    json.dumps(dict(report), allow_nan=False)


def test_consensus_selector_passes_max_train_samples():
    selector = ConsensusSelector(
        "reg", "r2", k=1, models={"ridge": Ridge()}, cv=2, n_repeats=2,
        random_state=0, max_train_samples=0.5,
    )
    assert clone(selector).get_params()["max_train_samples"] == 0.5

    selector.fit(X, Y)

    assert selector.report_["max_train_samples"] == 0.5
    for split in selector.report_["splits"]:
        assert split["report"]["run"]["n_train_rows"] == split["n_train"] // 2


# -- 3b. a subsample must keep every class ----------------------------------------


def clf_data():
    y = np.zeros(N, dtype=int)
    y[N // 2 :] = 1
    y[[10, 300]] = 2  # two rows of a rare class
    return y


def test_a_subsample_that_misses_a_class_raises_naming_it():
    y = clf_data()
    ranker = FeatureRanker("clf", models={"lr": LogisticRegression()}, verbose=False)

    with pytest.raises(ValueError, match=r"max_samples.*\[2\]|\[2\].*max_samples"):
        ranker.fit(X, y, feature_names=NAMES, max_samples=40, random_state=5)

    assert not ranker.is_fitted_


def test_a_subsample_with_every_class_fits():
    y = clf_data()
    ranker = FeatureRanker("clf", models={"lr": LogisticRegression()}, verbose=False)

    ranker.fit(X, y, feature_names=NAMES, max_samples=300, random_state=5)

    assert ranker.train_n_rows_ == 300


# -- 4. the run block identifies the training subsample ---------------------------


def test_run_records_the_training_subsample_settings():
    ranker = recording_ranker().fit(
        X, Y, feature_names=NAMES, max_samples=0.25, random_state=3
    )
    run = rank(ranker)["run"]

    rows = np.sort(np.random.default_rng([3, 3]).choice(N, 100, replace=False))
    assert run["train_rows_sha256"] == _labels_fingerprint(rows)
    assert run["train_max_samples"] == 0.25 and type(run["train_max_samples"]) is float
    assert run["train_random_state"] == 3
    json.dumps(run, allow_nan=False)


def test_run_records_a_numpy_count_as_a_plain_int_and_no_seed_as_no_key():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=np.int64(60))
    run = rank(ranker)["run"]

    assert run["train_max_samples"] == 60 and type(run["train_max_samples"]) is int
    assert "train_random_state" not in run and "train_rows_sha256" in run


def test_run_has_no_subsample_keys_without_max_samples():
    run = rank(recording_ranker().fit(X, Y, feature_names=NAMES, random_state=3))["run"]
    assert not {"train_rows_sha256", "train_max_samples", "train_random_state"} & set(run)


def test_ablate_refuses_a_ranker_refit_on_other_rows():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=100, random_state=6)
    report = rank(ranker)
    other = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=100, random_state=7)

    with pytest.raises(ValueError, match="training rows"):
        other.ablate(X, Y, X[300:], Y[300:], report, ks=[1])
    full = recording_ranker().fit(X, Y, feature_names=NAMES)
    with pytest.raises(ValueError, match="training rows"):
        full.ablate(X, Y, X[300:], Y[300:], report, ks=[1])


def test_ablate_accepts_a_ranker_pickled_without_train_rows():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES)
    report = rank(ranker)
    state = ranker.__getstate__()
    for key in ("train_n_rows_", "train_rows_"):
        state.pop(key)
    loaded = FeatureRanker.__new__(FeatureRanker)
    loaded.__setstate__(state)

    ablated = loaded.ablate(X, Y, X[300:], Y[300:], report, ks=[1])

    assert ablated["ablation"]["curve"][0]["scores"]["ridge"] is not None


def test_ablate_refuses_a_subsampled_ranker_for_a_report_from_before_the_row_hash():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES, max_samples=100, random_state=6)
    report = rank(ranker)
    del report["run"]["train_rows_sha256"]
    del report["run"]["fingerprint_scheme"]  # float columns hash the same in both schemes

    with pytest.raises(ValueError, match="training rows"):
        ranker.ablate(X, Y, X[300:], Y[300:], report, ks=[1])


def test_ablate_accepts_a_full_ranker_for_a_report_from_before_the_row_hash():
    ranker = recording_ranker().fit(X, Y, feature_names=NAMES)
    report = rank(ranker)
    assert "train_rows_sha256" not in report["run"]
    del report["run"]["fingerprint_scheme"]

    ranker.ablate(X, Y, X[300:], Y[300:], report, ks=[1])


# -- 5/6. levels and counts come from the subsample --------------------------------


def test_a_cap_above_the_level_count_leaves_unseen_levels_all_zero():
    X_, y_ = cat_frame(8)
    ranker = cat_ranker(max_categories=20).fit(X_, y_)

    encoder = encoder_of(linear_models(ranker)[0])

    assert encoder.transform(np.array([[1e6]])).toarray().sum() == 0


def test_levels_and_missing_counts_are_counted_on_the_subsample():
    frame = pd.DataFrame({"x": np.arange(200.0), "c": ["a"] * 190 + [None] * 5 + ["rare"] * 5})
    target = np.arange(200.0)
    ranker = FeatureRanker(
        "reg", "fast", verbose=False, categorical=["c"], missing="impute"
    ).fit(frame, target, max_samples=30, random_state=0)
    rows = np.sort(np.random.default_rng([0, 3]).choice(200, 30, replace=False))

    assert ranker.train_missing_counts_["c"] == int(frame["c"].iloc[rows].isna().sum())
    assert set(ranker.categorical_levels_["c"]) - {None} <= {"a", "rare"}
    assert "rare" in ranker.categorical_levels_["c"] or not (rows >= 195).any()
