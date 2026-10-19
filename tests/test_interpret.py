import copy
import hashlib
import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from featranker import FeatureRanker, explain, summarize, to_prompt

# to_prompt() and explain() warn since 1.1; the tests below cover that warning.
pytestmark = pytest.mark.filterwarnings("ignore:.*deprecated since 1.1:FutureWarning")

RNG = np.random.default_rng(0)
X = RNG.normal(size=(60, 3))
Y = 3.0 * X[:, 0] + 1.0 * X[:, 1] + 0.1 * RNG.normal(size=60)
NAMES = ["bmi", "serum", "fit"]  # "fit" collides with a failure stage name


class FailFit(LinearRegression):
    def fit(self, X, y):
        raise ValueError("boom")


def make_ranker(**kwargs):
    models = {"lin": LinearRegression(), "ridge": Ridge(), "bad": FailFit()}
    return FeatureRanker(task="reg", models=models, verbose=False, **kwargs)


@pytest.fixture(scope="module")
def report():
    ranker = make_ranker().fit(X[:40], Y[:40], NAMES)
    return ranker.rank_features(X[40:], Y[40:], scoring="r2", n_repeats=3, random_state=0)


@pytest.fixture(scope="module")
def resampled():
    return make_ranker().rank_resampled(X, Y, feature_names=NAMES, scoring="r2", cv=3,
                                        n_repeats=2, random_state=0)


def data_section(prompt):
    start = prompt.index("DATA:\n") + len("DATA:\n")
    end = prompt.index("\n\nREPLY FORMAT")
    return json.loads(prompt[start:end])


def test_prompt_has_rules_findings_and_groups(report):
    prompt = to_prompt(report)

    assert "not causal" in prompt
    for finding in summarize(report):
        assert finding["code"] in prompt
    for row in report["consensus"]:
        assert row["feature_group"] in prompt
    assert prompt.index("RULES") < prompt.index("FINDINGS") < prompt.index("DATA:") \
        < prompt.index("REPLY FORMAT")


def test_prompt_is_deterministic_compact_and_pure(report):
    before = copy.deepcopy(report)

    prompt = to_prompt(report)

    assert prompt == to_prompt(report)
    assert report == before
    data = data_section(prompt)
    text = json.dumps(data)
    assert '"values"' not in text and '"params"' not in text
    assert data["consensus"] == report["consensus"]


def test_resampled_report_prompt(resampled):
    data = data_section(to_prompt(resampled))

    assert data["n_splits"] == 3
    assert data["stability"] == resampled["stability"]
    assert "splits" not in data or all("report" not in s for s in data["splits"])


def test_non_dict_raises_type_error():
    with pytest.raises(TypeError):
        to_prompt(["not", "a", "report"])


def all_strings(obj):
    if isinstance(obj, dict):
        return set(obj) | {s for v in obj.values() for s in all_strings(v)}
    if isinstance(obj, list):
        return {s for v in obj for s in all_strings(v)}
    return {obj} if isinstance(obj, str) else set()


def test_anonymized_prompt_hides_names(report):
    prompt = to_prompt(report, anonymize=True)

    assert "bmi" not in prompt and "serum" not in prompt
    data = data_section(prompt)
    assert data["feature_groups"] == {"g1": ["c1"], "g2": ["c2"], "g3": ["c3"]}
    assert {row["feature_group"] for row in data["consensus"]} == {"g1", "g2", "g3"}


def test_anonymization_is_positional_not_text_replacement(report):
    data = data_section(to_prompt(report, anonymize=True))

    # column "fit" is renamed, but the failure stage "fit" is untouched
    assert "fit" not in data["feature_groups"]
    assert {f["stage"] for f in data["failures"]} == {"fit"}
    # custom model names are caller-chosen identifiers, so they are aliased too
    assert set(data["models"]) == {"m1", "m2"}
    assert {f["model"] for f in data["failures"]} == {"m3"}
    assert set(data["models"]["m1"]["importance"]) == {"g1", "g2", "g3"}


def test_anonymized_resampled_prompt(resampled):
    prompt = to_prompt(resampled, anonymize=True)

    assert "bmi" not in prompt and "serum" not in prompt
    assert {row["feature_group"] for row in data_section(prompt)["stability"]} == {
        "g1", "g2", "g3"
    }


def test_anonymized_findings_use_tokens(report):
    tied = copy.deepcopy(report)
    for row in tied["consensus"]:
        row["median_rank"] = 1.0  # force a tied_groups finding naming every group

    prompt = to_prompt(tied, anonymize=True)

    assert "tied_groups" in prompt and "bmi" not in prompt



def reply(claims, summary="A short summary."):
    return json.dumps({"summary": summary, "claims": claims})


def fake(text):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return text

    llm.calls = calls
    return llm


def test_llm_called_once_with_the_prompt(report):
    llm = fake(reply([]))

    explain(report, llm)

    assert llm.calls == [to_prompt(report)]


@pytest.mark.parametrize("llm", [None, "claude", lambda prompt: 42])
def test_bad_llm_raises_type_error(report, llm):
    with pytest.raises(TypeError):
        explain(report, llm)


def test_fenced_json_is_parsed(report):
    result = explain(report, fake("```json\n" + reply([]) + "\n```"))

    assert result["summary"] == "A short summary." and result["claims"] == []


@pytest.mark.parametrize(
    "text",
    ["The features look good.", json.dumps({"claims": []}),
     json.dumps({"summary": "s", "claims": [{"text": "t"}]}),
     json.dumps({"summary": "s", "claims": [{"text": "t", "evidence": [{"value": 1}]}]})],
)
def test_malformed_reply_raises_value_error_with_excerpt(report, text):
    with pytest.raises(ValueError) as caught:
        explain(report, fake(text))

    assert text[:20] in str(caught.value)


def test_citations_are_verified(report):
    first = report["consensus"][0]
    good = {"path": "consensus[0].median_rank", "value": round(first["median_rank"], 2)}
    group = {"path": "consensus[0].feature_group", "value": first["feature_group"]}
    wrong = {"path": "consensus[0].median_rank", "value": first["median_rank"] + 2}
    missing = {"path": "consensus[99].median_rank", "value": 1.0}

    result = explain(report, fake(reply([
        {"text": "top group", "evidence": [good, group]},
        {"text": "wrong number", "evidence": [wrong]},
        {"text": "missing path", "evidence": [missing]},
        {"text": "uncited", "evidence": []},
    ])))

    assert [c["verified"] for c in result["claims"]] == [True, False, False, False]
    assert [e["verified"] for e in result["claims"][0]["evidence"]] == [True, True]
    assert result["all_verified"] is False


def test_all_verified_when_every_claim_checks_out(report):
    path = "models.lin.evaluation_score"
    value = report["models"]["lin"]["evaluation_score"]

    result = explain(report, fake(reply([{"text": "fit", "evidence": [{"path": path, "value": value}]}])))

    assert result["all_verified"] is True


def test_anonymized_explain_restores_marked_names_only(report):
    data = data_section(to_prompt(report, anonymize=True))
    top = data["consensus"][0]["feature_group"]          # e.g. "g1"
    original = report["consensus"][0]["feature_group"]   # e.g. "bmi"
    evidence = [{"path": "consensus[0].feature_group", "value": top}]

    result = explain(report, fake(reply(
        [{"text": f"[[{top}]] ranks first; plain {top} and [[{top}0]] stay", "evidence": evidence}],
        summary=f"[[{top}]] leads.",
    )), anonymize=True)

    assert result["summary"] == f"{original} leads."
    assert result["claims"][0]["text"] == f"{original} ranks first; plain {top} and [[{top}0]] stay"
    assert result["claims"][0]["evidence"][0] == {**evidence[0], "verified": True}


@pytest.mark.parametrize(
    "path", ["consensus..0", ".baseline_score", "consensus[0]..median_rank",
             "consensus[x].median_rank", "consensus[0]median_rank", ""],
)
def test_malformed_paths_are_not_verified(report, path):
    result = explain(report, fake(reply([{"text": "t", "evidence": [{"path": path, "value": 1}]}])))

    assert result["claims"][0]["verified"] is False


# --- Signal round-1 findings -------------------------------------------------

def leaky_report():
    """A report whose dummy scoring failed with a message naming a feature and a row."""
    def scorer(model, X, y):
        if type(model).__name__.startswith("Dummy"):
            raise ValueError("bad row patient_hiv=1 age=57.3")
        return model.score(X, y)

    ranker = FeatureRanker(task="reg", models={"patient_secret": Ridge()}, verbose=False)
    ranker.fit(X[:40], Y[:40], ["patient_hiv", "age", "patient_secret"])
    return ranker.rank_features(X[40:], Y[40:], scoring=scorer, n_repeats=1)


def test_failure_messages_never_reach_the_prompt():
    report = leaky_report()
    assert any(f["model"] == "dummy_baseline" for f in report["failures"])

    plain = to_prompt(report)
    hidden = to_prompt(report, anonymize=True)

    assert "57.3" not in plain and "bad row" not in plain
    assert "57.3" not in hidden and "patient_hiv" not in hidden
    assert report["failures"][0]["message"] == "bad row patient_hiv=1 age=57.3"


def test_anonymize_aliases_custom_model_family_and_scorer_names():
    prompt = to_prompt(leaky_report(), anonymize=True)

    assert "patient_secret" not in prompt
    data = data_section(prompt)
    assert set(data["models"]) == {"m1"} and data["models"]["m1"]["family"] == "f1"
    assert data["scoring"] == "custom_scorer"


def test_anonymize_keeps_builtin_model_and_scorer_names():
    report = FeatureRanker(task="reg", group="linear", verbose=False).fit(
        X[:40], Y[:40], NAMES
    ).rank_features(X[40:], Y[40:], scoring="r2", n_repeats=1)

    data = data_section(to_prompt(report, anonymize=True))

    assert "ridge_regression" in data["models"] and data["scoring"] == "r2"
    assert data["models"]["ridge_regression"]["family"] == "linear"


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, float("nan"), float("inf")])
def test_non_scalar_or_non_finite_values_are_rejected(report, value):
    text = json.dumps({"summary": "s", "claims": [
        {"text": "t", "evidence": [{"path": "consensus", "value": value}]}]})

    with pytest.raises(ValueError):
        explain(report, fake(text))


def test_booleans_and_numbers_do_not_coerce(report):
    n_models = report["consensus"][0]["n_models"]
    result = explain(report, fake(reply([
        {"text": "a", "evidence": [{"path": "models.lin.in_consensus", "value": 1}]},
        {"text": "b", "evidence": [{"path": "consensus[0].n_models", "value": True}]},
        {"text": "c", "evidence": [{"path": "consensus[0].n_models", "value": float(n_models)}]},
    ])))

    assert [c["verified"] for c in result["claims"]] == [False, False, True]


def test_singleton_group_and_column_both_restore(report):
    result = explain(report, fake(reply([], summary="[[c1]] and [[g1]]")), anonymize=True)

    assert result["summary"] == "bmi and bmi"


def test_json_pointer_paths_reach_keys_with_dots_and_brackets():
    ranker = FeatureRanker(task="reg", models={"model.v1": LinearRegression()}, verbose=False)
    ranker.fit(X[:40], Y[:40], ["lab[0]", "a/b", "c"])
    report = ranker.rank_features(X[40:], Y[40:], scoring="r2", n_repeats=1)
    lab = report["models"]["model.v1"]["importance"]["lab[0]"]["mean"]
    slash = report["models"]["model.v1"]["importance"]["a/b"]["mean"]

    result = explain(report, fake(reply([
        {"text": "a", "evidence": [{"path": "/models/model.v1/importance/lab[0]/mean", "value": lab}]},
        {"text": "b", "evidence": [{"path": "/models/model.v1/importance/a~1b/mean", "value": slash}]},
        {"text": "c", "evidence": [{"path": "/consensus/0/n_models", "value": 1}]},
        {"text": "d", "evidence": [{"path": "/consensus/01/n_models", "value": 1}]},
    ])))

    assert [c["verified"] for c in result["claims"]] == [True, True, True, False]


def test_verification_uses_a_snapshot_taken_before_the_llm_call(report):
    mutable = copy.deepcopy(report)

    def tampering_llm(prompt):
        mutable["consensus"][0]["median_rank"] = 999
        return reply([{"text": "t", "evidence": [{"path": "consensus[0].median_rank", "value": 999}]}])

    assert explain(mutable, tampering_llm)["claims"][0]["verified"] is False


def test_prompt_tie_rule_is_qualified(report):
    prompt = to_prompt(report)

    assert "Equal median_rank ties the primary score only" in prompt
    assert "mean_rank" in prompt and "top_k_share" in prompt


def test_anonymized_prompt_asks_for_markers(report):
    assert "[[g1]]" in to_prompt(report, anonymize=True)
    assert "[[g1]]" not in to_prompt(report)


def test_oversized_number_is_unverified_not_a_crash(report):
    text = '{"summary": "s", "claims": [{"text": "t", "evidence": [{"path": "consensus[0].n_models", "value": 1' + "0" * 400 + '}]}]}'

    assert explain(report, fake(text))["claims"][0]["verified"] is False


# --- Signal round-2 findings -------------------------------------------------

def test_custom_exception_names_are_replaced():
    secret_error = type("patient_hiv_1_age_57_3", (ValueError,), {})

    def scorer(model, X, y):
        if type(model).__name__.startswith("Dummy"):
            raise secret_error("x")
        return model.score(X, y)

    ranker = FeatureRanker(task="reg", models={"lin": LinearRegression(), "bad": FailFit()},
                           verbose=False).fit(X[:40], Y[:40], NAMES)
    report = ranker.rank_features(X[40:], Y[40:], scoring=scorer, n_repeats=1)

    for anonymize in (False, True):
        prompt = to_prompt(report, anonymize=anonymize)
        assert "patient_hiv" not in prompt
        types = {f["error_type"] for f in data_section(prompt)["failures"]}
        assert types == {"CustomError", "ValueError"}  # builtin names stay


def test_invalid_pointer_escape_is_not_verified():
    ranker = FeatureRanker(task="reg", models={"r": LinearRegression()}, verbose=False)
    ranker.fit(X[:40], Y[:40], ["a~2b", "b", "c"])
    report = ranker.rank_features(X[40:], Y[40:], scoring="r2", n_repeats=1)
    mean = report["models"]["r"]["importance"]["a~2b"]["mean"]

    result = explain(report, fake(reply([
        {"text": "bad", "evidence": [{"path": "/models/r/importance/a~2b/mean", "value": mean}]},
        {"text": "good", "evidence": [{"path": "/models/r/importance/a~02b/mean", "value": mean}]},
    ])))

    assert [c["verified"] for c in result["claims"]] == [False, True]


def test_marker_instruction_keeps_builtin_names():
    prompt = to_prompt(FeatureRanker(task="reg", group="linear", verbose=False).fit(
        X[:40], Y[:40], NAMES).rank_features(X[40:], Y[40:], scoring="r2", n_repeats=1),
        anonymize=True)

    assert "Keep unaliased built-in names unchanged" in prompt
    assert "every feature group, column, model, and family name" not in prompt


def test_anonymized_findings_hide_names_in_correlated_pairs(report):
    correlated = copy.deepcopy(report)
    correlated["evidence"] = {
        "correlation": {"pairs": [{"groups": ["bmi", "serum"], "rho": 0.9}]}
    }

    prompt = to_prompt(correlated, anonymize=True)

    assert "correlated_groups" in prompt
    assert "bmi" not in prompt and "serum" not in prompt
    assert "g1 and g2" in prompt


@pytest.mark.parametrize("call", ["to_prompt", "explain"])
def test_deprecated_functions_warn_once_and_name_2_0(report, call):
    def run():
        if call == "to_prompt":
            return to_prompt(report)
        return explain(report, fake(reply([])))

    with pytest.warns(FutureWarning, match=r"deprecated since 1\.1.*removed in 2\.0") as caught:
        result = run()

    assert len(caught) == 1
    assert caught[0].filename == __file__  # points at the caller
    assert call in str(caught[0].message)
    assert "deprecated" in (to_prompt.__doc__ if call == "to_prompt" else explain.__doc__).lower()
    if call == "explain":
        assert set(result) == {"summary", "claims", "all_verified"}


def test_explain_still_calls_the_llm_once(report):
    llm = fake(reply([]))

    with pytest.warns(FutureWarning):
        explain(report, llm)

    assert len(llm.calls) == 1


def golden_report():
    """A small hand-written 1.0-shaped report; no model is trained."""
    def model(family, evaluation_score, means):
        return {
            "family": family, "evaluation_score": evaluation_score,
            "in_consensus": True, "params": {"alpha": 1.0},
            "importance": {
                name: {"values": [mean], "mean": mean, "std": 0.05, "rank": rank}
                for rank, (name, mean) in enumerate(means.items(), start=1)
            },
        }

    means = {"bmi": 0.5, "serum": 0.2, "fit": 0.01}
    row = lambda name, rank: {  # noqa: E731
        "feature_group": name, "median_rank": rank, "mean_rank": rank,
        "rank_std": 0.0, "n_models": 2, "n_families": 2, "family_ranks": {},
    }
    return {
        "evaluation_mode": "held_out", "scoring": "r2", "n_repeats": 3,
        "random_state": 0,
        "feature_groups": {"bmi": ["bmi"], "serum": ["serum"], "fit": ["fit"]},
        "baseline_score": -0.01,
        "models": {"ridge": model("ridge", 0.8, means), "lasso": model("lasso", 0.7, means)},
        "consensus": [row("bmi", 1.0), row("serum", 2.0), row("fit", 3.0)],
        "failures": [],
    }


# sha256 of the 1.0 prompts for golden_report(), generated from the 1.0.0 package
GOLDEN_PROMPT_SHA256 = {
    False: "8f5200ad99eab140f6611afc2af65f0248fa05c3978919b96e0b5363e5026bd6",
    True: "0c20e476a04ba07c8146c4ce2f9b43cca309e466541046d75e9f7c0b97665ee2",
}


@pytest.mark.parametrize("anonymize", [False, True])
def test_prompt_is_unchanged_from_1_0(anonymize):
    prompt = to_prompt(golden_report(), anonymize=anonymize)

    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == GOLDEN_PROMPT_SHA256[anonymize]
