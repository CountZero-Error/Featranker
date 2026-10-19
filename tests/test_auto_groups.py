import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from featranker import FeatureRanker, auto_groups

RNG = np.random.default_rng(0)
N = 300
E = RNG.normal(size=N)
F = E + 0.05 * RNG.normal(size=N)  # near-copy of E
G = RNG.normal(size=N)
H = RNG.normal(size=N)


def columns(*arrays):
    return np.column_stack(arrays)


def test_near_copy_pair_is_grouped():
    frame = pd.DataFrame({"e": E, "f": F, "g": G, "h": H})

    assert auto_groups(frame) == {"e+f": ["e", "f"]}


def test_independent_columns_give_no_groups():
    assert auto_groups(columns(E, G, H)) == {}


def test_single_column_gives_no_groups():
    assert auto_groups(columns(E)) == {}


def test_complete_linkage_keeps_weakly_related_ends_apart():
    # a-b and b-c about 0.87 correlated, a-c about 0.5: chaining would join all three.
    a, c = RNG.normal(size=(2, 2000))
    c = 0.5 * a + np.sqrt(0.75) * c
    b = a + c
    frame = pd.DataFrame({"a": a, "b": b, "c": c})

    groups = auto_groups(frame)

    assert len(groups) == 1
    assert not {"a", "c"} <= set(next(iter(groups.values())))


def test_constant_column_is_never_grouped():
    zero = np.zeros(N)

    assert auto_groups(columns(zero, zero, E, F)) == {"x2+x3": ["x2", "x3"]}


def test_names_come_from_dataframe_feature_names_or_default():
    X = columns(E, F, G)

    assert auto_groups(pd.DataFrame(X, columns=["a", "b", "c"])) == {"a+b": ["a", "b"]}
    assert auto_groups(X, feature_names=["p", "q", "r"]) == {"p+q": ["p", "q"]}
    assert auto_groups(X) == {"x0+x1": ["x0", "x1"]}


def test_groups_follow_first_member_and_members_follow_column_order():
    X = columns(E, G, G + 0.05 * H, F)  # pairs (x0, x3) and (x1, x2)

    assert list(auto_groups(X).items()) == [
        ("x0+x3", ["x0", "x3"]),
        ("x1+x2", ["x1", "x2"]),
    ]


def test_threshold_moves_the_cut():
    weak = E + 1.0 * RNG.normal(size=N)  # rho with E near 0.7 to 0.8
    X = columns(E, weak)

    assert auto_groups(X, threshold=0.5) == {"x0+x1": ["x0", "x1"]}
    assert auto_groups(X, threshold=0.99) == {}


@pytest.mark.parametrize(
    "threshold", [0, -0.5, 1.5, "0.7", None, True, float("nan")]
)
def test_bad_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        auto_groups(columns(E, F), threshold=threshold)


def test_name_collision_with_a_column_raises():
    frame = pd.DataFrame({"e": E, "f": F, "e+f": G})

    with pytest.raises(ValueError, match=r"e\+f"):
        auto_groups(frame)


@pytest.mark.parametrize(
    "X",
    [np.ones((1, 3)), np.ones(3), np.array([[1.0, np.nan], [2.0, 3.0]])],
)
def test_invalid_x_raises(X):
    with pytest.raises(ValueError, match=r"\[!\]"):
        auto_groups(X)


def test_x_is_not_modified():
    X = columns(E, F, G)
    frame = pd.DataFrame(X.copy(), columns=["a", "b", "c"])
    before = X.copy()

    auto_groups(X)
    auto_groups(frame)

    np.testing.assert_array_equal(X, before)
    pd.testing.assert_frame_equal(frame, pd.DataFrame(before, columns=["a", "b", "c"]))


def test_result_is_usable_as_feature_groups():
    frame = pd.DataFrame({"e": E, "f": F, "g": G})
    y = E + 0.1 * RNG.normal(size=N)
    groups = auto_groups(frame)
    ranker = FeatureRanker("reg", models={"lin": LinearRegression()}, verbose=False)
    ranker.fit(frame[:200], y[:200])

    report = ranker.rank_features(
        frame[200:], y[200:], scoring="r2", feature_groups=groups, n_repeats=2, random_state=0
    )

    assert report["feature_groups"]["e+f"] == ["e", "f"]


@pytest.mark.parametrize("n_rows", [*range(3, 40), 100, 200])
def test_exact_monotone_copies_group_at_threshold_one(n_rows):
    a = np.random.default_rng(n_rows).normal(size=n_rows)

    assert auto_groups(columns(a, np.exp(a), 3 * a + 1, -a), threshold=1.0) == {
        "x0+x1+x2+x3": ["x0", "x1", "x2", "x3"]
    }


def test_constant_column_is_never_grouped_at_tiny_threshold():
    X = columns(np.zeros(N), E, G)

    assert auto_groups(X, threshold=1e-17) == {"x1+x2": ["x1", "x2"]}


def test_duplicate_generated_group_names_raise():
    # a~"b+c" and "a+b"~c both generate the name "a+b+c".
    frame = pd.DataFrame({"a": E, "b+c": E + 0.01 * F, "a+b": G, "c": G + 0.01 * H})

    with pytest.raises(ValueError, match=r"a\+b\+c"):
        auto_groups(frame)
