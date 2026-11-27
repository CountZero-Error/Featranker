"""Regressions verified during the 2.0 release audit."""
import os
import stat
import subprocess
import sys

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from featranker import ConsensusSelector, FeatureRanker
from featranker.cli import _write_report


X = np.random.default_rng(8).normal(size=(60, 2))
Y = X[:, 0] + X[:, 1]
NAMES = ["a", "b"]


def fitted(model=None, **options):
    return FeatureRanker("reg", models={"m": Ridge() if model is None else model},
                         verbose=False, **options).fit(X[:40], Y[:40], feature_names=NAMES)


def rank(ranker, **options):
    return ranker.rank_features(X[40:], Y[40:], scoring="r2", random_state=0,
                                n_repeats=1, **options)


@pytest.mark.parametrize("labels", [[0, "0"] * 10, np.array([0, "0"] * 10, dtype=object),
                                    [2**53 + 1, complex(2**53)] * 10,
                                    [2**53 + 1, np.clongdouble(2**53)] * 10])
def test_group_labels_keep_distinct_types(labels):
    r = FeatureRanker("reg", models={"m": Ridge()}, shuffle="within_groups", verbose=False)
    r.fit(X[:40], Y[:40], feature_names=NAMES, shuffle_groups=[0, 1] * 20)
    with pytest.raises(ValueError, match="sortable"):
        rank(r, shuffle_groups=labels)


def test_refit_drops_previous_training_evidence():
    r = fitted(evidence=True)
    r.evidence = False
    r.fit(X[:40], -Y[:40], feature_names=NAMES)
    assert not hasattr(r, "train_X_")
    r.evidence = True
    with pytest.raises(ValueError, match="evidence"):
        rank(r)


class DropsShadows(FunctionTransformer):
    def transform(self, X):
        return X[:, :2]


class DropsWithPredict(DropsShadows):
    def predict(self, X):
        return X[:, 0]


@pytest.mark.parametrize("transformer", [DropsShadows(), DropsWithPredict()])
def test_custom_transform_cannot_vote_on_dropped_shadows(transformer):
    report = rank(fitted(make_pipeline(transformer, Ridge()), shadows=True, shadow_seed=1))
    assert any(f["stage"] == "shadows" for f in report["failures"])
    assert "shadow_max" not in report["models"]["m"]


def test_ablation_distinguishes_model_classes_in_different_modules():
    first = type("Regressor", (Ridge,), {"__module__": "first"})
    second = type("Regressor", (Ridge,), {"__module__": "second"})
    old = rank(fitted(first()))
    with pytest.raises(ValueError, match="Regressor"):
        fitted(second()).ablate(X[:40], Y[:40], X[40:], Y[40:], old, ks=[2])


@pytest.mark.parametrize("bad_seed", [True, np.timedelta64(1, "ns")])
def test_assigned_shadow_seed_validated_before_ranking(bad_seed):
    r = fitted(shadows=True, shadow_seed=1)
    r.shadow_seed = bad_seed
    with pytest.raises(TypeError, match="shadow_seed"):
        rank(r)


@pytest.mark.parametrize("as_list", [False, True])
def test_object_date_column_rejected_before_numeric_cast(as_list):
    dates = np.array([[np.datetime64("2024-01-01")]] * len(Y), dtype=object)
    r = FeatureRanker("reg", models={"m": Ridge()}, verbose=False)
    with pytest.raises(ValueError, match="datetime"):
        r.fit(dates.tolist() if as_list else dates, Y, feature_names=["date"])


@pytest.mark.parametrize("task", ["reg", "clf"])
def test_integer_cv_accepts_full_seed_in_selector(task):
    from sklearn.linear_model import LogisticRegression
    s = ConsensusSelector(task, "r2" if task == "reg" else "accuracy", cv=2, k=1,
                          models={"m": Ridge() if task == "reg" else LogisticRegression()},
                          random_state=2**64 - 1, n_repeats=1)
    s.fit(X, Y if task == "reg" else (Y > 0))
    assert s.get_support().sum() == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_private_output_stays_private(tmp_path):
    target = tmp_path / "report.json"
    target.touch(mode=0o600)
    old = os.umask(0o022)
    try:
        _write_report({"new": True}, target)
    finally:
        os.umask(old)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACL")
def test_windows_output_keeps_protected_dacl(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{}")
    subprocess.run(["icacls", str(target), "/inheritance:d"], check=True)

    # A PowerShell 7 parent (the GitHub Actions shell) leaks its PSModulePath, and
    # Windows PowerShell 5.1 then cannot autoload Get-Acl; let it use its own path.
    env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}

    def security_descriptor():
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-Acl -LiteralPath $env:REPORT_PATH).Sddl"],
            env={**env, "REPORT_PATH": str(target)}, text=True,
        ).strip()

    before = security_descriptor()
    assert "D:P" in before
    _write_report({"updated": True}, target)
    assert security_descriptor() == before


@pytest.mark.skipif(sys.platform != "linux", reason="Linux POSIX ACL")
def test_linux_output_keeps_access_acl(tmp_path):
    import errno
    import struct

    target = tmp_path / "report.json"
    target.write_text("{}")
    # Linux POSIX ACL v2: owner rw, named user read, group/other none, mask read.
    no_id = 2**32 - 1
    entries = [(1, 6, no_id), (2, 4, 12345), (4, 0, no_id), (16, 4, no_id), (32, 0, no_id)]
    acl = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)
    key = "system.posix_acl_access"
    try:
        os.setxattr(target, key, acl)
    except OSError as exc:
        if exc.errno == errno.ENOTSUP:
            pytest.skip("filesystem lacks POSIX ACLs")
        raise
    _write_report({"updated": True}, target)
    assert os.getxattr(target, key) == acl


@pytest.mark.parametrize("float_type", [np.float64, np.longdouble])
@pytest.mark.parametrize("as_array", [False, True])
def test_mixed_numeric_groups_remain_exact(float_type, as_array):
    labels = [2**53 + 1, float_type(2**53)] * 20
    if as_array:
        labels = np.array(labels, dtype=object)
    feature = np.tile([0., 1.], 20).reshape(-1, 1)
    r = FeatureRanker("reg", models={"m": Ridge()}, shuffle="within_groups", verbose=False)
    r.fit(feature, feature[:, 0], feature_names=["a"], shuffle_groups=labels)
    report = r.rank_features(feature, feature[:, 0], scoring="r2", n_repeats=2,
                             random_state=0, shuffle_groups=labels)
    assert report["constant_within_groups"] == ["a"]
    assert report["models"]["m"]["importance"]["a"]["mean"] == 0.0


def test_wrapped_transformer_cannot_vote_on_shadows():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import GridSearchCV

    pipeline = make_pipeline(DropsWithPredict(), Ridge())
    for model in (TransformedTargetRegressor(regressor=pipeline),
                  GridSearchCV(pipeline, {"ridge__alpha": [1.0]}, cv=2)):
        result = rank(fitted(model, shadows=True, shadow_seed=1))
        assert any(f["stage"] == "shadows" for f in result["failures"])
        assert "shadow_max" not in result["models"]["m"]


def test_group_splitter_uses_same_exact_labels_as_permutations():
    from sklearn.model_selection import GroupKFold

    r = FeatureRanker("reg", models={"m": Ridge()}, shuffle="within_groups", verbose=False)
    report = r.rank_resampled(X, Y, feature_names=NAMES, scoring="r2", cv=GroupKFold(2),
                              groups=[2**53 + 1, np.float64(2**53)] * 30,
                              n_repeats=1, random_state=0)
    assert len(report["splits"]) == 2


@pytest.mark.parametrize("labels", [["a", "a\0"] * 20, [b"a", b"a\0"] * 20])
def test_text_groups_preserve_trailing_nuls(labels):
    feature = np.tile([0., 1.], 20).reshape(-1, 1)
    r = FeatureRanker("reg", models={"m": Ridge()}, shuffle="within_groups", verbose=False)
    r.fit(feature, feature[:, 0], feature_names=["a"], shuffle_groups=labels)
    report = r.rank_features(feature, feature[:, 0], scoring="r2", n_repeats=2,
                             random_state=0, shuffle_groups=labels)
    assert report["constant_within_groups"] == ["a"]
    assert report["models"]["m"]["importance"]["a"]["mean"] == 0.0


class PredictorWithTransform(Ridge):
    def transform(self, X):
        return X[:, :2]


def test_wrapped_final_predictor_keeps_shadow_vote():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), PredictorWithTransform())
    for model in (TransformedTargetRegressor(regressor=PredictorWithTransform()),
                  GridSearchCV(pipeline, {"predictorwithtransform__alpha": [1.0]}, cv=2)):
        result = rank(fitted(model, shadows=True, shadow_seed=1))
        assert not any(f["stage"] == "shadows" for f in result["failures"])
        assert "shadow_max" in result["models"]["m"]


def test_grid_search_inspects_the_fitted_transformer():
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), Ridge())
    model = GridSearchCV(pipeline, {"standardscaler": [DropsWithPredict()]}, cv=2)
    result = rank(fitted(model, shadows=True, shadow_seed=1))
    assert any(f["stage"] == "shadows" for f in result["failures"])
    assert "shadow_max" not in result["models"]["m"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
@pytest.mark.parametrize("mode", [0o200, 0o2700, 0o4700])
def test_report_replacement_preserves_posix_modes(tmp_path, mode):
    target = tmp_path / "report.json"
    target.write_text("{}")
    target.chmod(mode)
    if stat.S_IMODE(target.stat().st_mode) != mode:
        pytest.skip("filesystem or sandbox does not allow these mode bits")
    original_inode = target.stat().st_ino

    _write_report({"updated": True}, target)

    assert target.stat().st_ino != original_inode
    assert stat.S_IMODE(target.stat().st_mode) == mode
    target.chmod(0o600)
    assert '"updated": true' in target.read_text()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL")
def test_write_only_report_replacement_preserves_macos_acl(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("{}")
    target.chmod(0o200)
    subprocess.run(["/bin/chmod", "+a", "everyone deny execute", str(target)], check=True)

    def access_entries():
        return subprocess.check_output(["/bin/ls", "-le", str(target)], text=True).splitlines()[1:]

    before = access_entries()
    assert before
    _write_report({"updated": True}, target)
    assert access_entries() == before
    assert stat.S_IMODE(target.stat().st_mode) == 0o200


def test_grid_search_as_intermediate_transformer_cannot_vote_on_dropped_shadows():
    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(PredictorWithTransform(), {"alpha": [1.0]}, cv=2)
    result = rank(fitted(make_pipeline(search, Ridge()), shadows=True, shadow_seed=1))
    assert any(f["stage"] == "shadows" for f in result["failures"])
    assert "shadow_max" not in result["models"]["m"]


def test_frozen_intermediate_transformer_cannot_vote_on_dropped_shadows():
    FrozenEstimator = pytest.importorskip("sklearn.frozen").FrozenEstimator

    fitted_transformer = PredictorWithTransform().fit(np.column_stack([X, X]), Y)
    model = make_pipeline(FrozenEstimator(fitted_transformer), Ridge())
    result = rank(fitted(model, shadows=True, shadow_seed=1))
    assert any(f["stage"] == "shadows" for f in result["failures"])
    assert "shadow_max" not in result["models"]["m"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ACL inheritance")
def test_report_staging_does_not_inherit_public_access(tmp_path, monkeypatch):
    from featranker import cli

    target = tmp_path / "report.json"
    target.touch(mode=0o600)
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow list,search,read,file_inherit,directory_inherit",
         str(tmp_path)], check=True,
    )

    def access_entries(path):
        return subprocess.check_output(["/bin/ls", "-led", str(path)], text=True).splitlines()[1:]

    # Confirm the fixture really grants access on a newly created child.
    probe = tmp_path / "probe"
    probe.mkdir(mode=0o700)
    assert access_entries(probe)
    preserve_report_access = cli._preserve_report_access

    def check_staged_access(destination, staged, original):
        assert not access_entries(staged.parent)
        assert not access_entries(staged)
        preserve_report_access(destination, staged, original)

    monkeypatch.setattr(cli, "_preserve_report_access", check_staged_access)
    cli._write_report({"updated": True}, target)
    assert not access_entries(target)


def test_voting_member_that_drops_shadows_cannot_supply_evidence():
    from sklearn.ensemble import VotingRegressor
    from sklearn.feature_selection import SelectKBest, f_regression

    pipeline = make_pipeline(SelectKBest(f_regression, k=2), Ridge())
    r = fitted(VotingRegressor([("member", pipeline)]), shadows=True, shadow_seed=1)
    selector = r.trained_models["m"].estimators_[0][0]
    assert set(selector.get_support(indices=True)) == {0, 1}

    report = rank(r)

    assert any(failure["stage"] == "shadows" for failure in report["failures"])
    assert "shadow_max" not in report["models"]["m"]


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="unprivileged POSIX access")
def test_read_only_output_refuses_overwrite(tmp_path):
    target = tmp_path / "report.json"
    target.write_text('{"original":true}')
    target.chmod(0o400)

    with pytest.raises(PermissionError):
        _write_report({"updated": True}, target)

    assert target.read_text() == '{"original":true}'


@pytest.mark.parametrize("refit", ["targets", "weights"])
def test_ablation_requires_the_latest_training_data_and_weights(refit):
    r = fitted()
    report = rank(r)
    settings = {"sample_weight": np.arange(1, 41)} if refit == "weights" else {}
    targets = -Y[:40] if refit == "targets" else Y[:40]
    r.fit(X[:40], targets, feature_names=NAMES, **settings)

    with pytest.raises(ValueError, match="latest fit"):
        r.ablate(X[:40], Y[:40], X[40:], Y[40:], report, ks=[2])


def test_ablation_checks_random_state_parameter_values():
    from sklearn.ensemble import RandomForestRegressor

    first = fitted(RandomForestRegressor(n_estimators=4, random_state=np.random.RandomState(0)))
    second = fitted(RandomForestRegressor(n_estimators=4, random_state=np.random.RandomState(10)))
    report = rank(first)

    assert first.ablate(X[:40], Y[:40], X[40:], Y[40:], report, ks=[2])
    with pytest.raises(ValueError, match="parameters"):
        second.ablate(X[:40], Y[:40], X[40:], Y[40:], report, ks=[2])


def test_stacking_final_selector_cannot_certify_shadow_evidence():
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import StackingRegressor
    from sklearn.feature_selection import SelectKBest, f_regression

    final = make_pipeline(SelectKBest(f_regression, k=1), Ridge())
    model = StackingRegressor([("member", DummyRegressor())], final_estimator=final,
                              passthrough=True, cv=2)
    r = fitted(model, shadows=True, shadow_seed=1)
    selector = r.trained_models["m"].final_estimator_[0]
    assert set(selector.get_support(indices=True)) <= {1, 2}

    report = rank(r)

    assert any(failure["stage"] == "shadows" for failure in report["failures"])
    assert "shadow_max" not in report["models"]["m"]
