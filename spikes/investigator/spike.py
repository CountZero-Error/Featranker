"""
Spike: does an LLM that runs preregistered follow-up experiments find more
planted mechanisms than one that only reads the report?

Throwaway and never shipped. Design and decision rule:
openspec/changes/spike-investigator-value/design.md (D1-D8). No network here;
run.py supplies the real LLM.
"""

import json
import math
import operator
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from featranker import FeatureRanker
from featranker.interpret import _data_view, _resolve

N_ROWS, N_TRAIN, N_FEATURES = 1000, 700, 10
MODEL_NAMES = ["ridge", "random_forest", "hist_gb", "decision_tree"]
TREE_MODELS = {"random_forest", "hist_gb", "decision_tree"}
MECHANISMS = {"linear", "nonlinear", "interaction", "correlated", "overfit", "noise"}
REFIT_PARAMS = {
    "ridge": {"alpha"},
    "random_forest": {"max_depth", "min_samples_leaf"},
    "hist_gb": {"max_depth", "min_samples_leaf"},
    "decision_tree": {"max_depth", "min_samples_leaf"},
}
ARMS = ("narrator", "evidence", "investigator")
MAX_FOLLOWUPS = 3
OVERFIT_GAP = 0.1  # train R² minus eval R² below this: calling the model overfit is false
CORRELATION_MIN = 0.5
INTERACTION_ROWS = 200  # interaction values are O(features²) per row
# D8: features a finding must name, by mechanism; any other count is a false finding.
ARITY = {"linear": 1, "nonlinear": 1, "noise": 1, "interaction": 2, "correlated": 2, "overfit": 0}
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}

# D6, fixed before any real run.
GAP_MIN = 0.2  # one more of the 5 mechanisms found
FALSE_MAX = 0.5
CONFIRMED_FALSE_MAX = 1
WIN_SHARE = 0.8  # 4 of 5 seeds
BORDER = 0.05
EPS = 1e-9  # 0.6 - 0.4 must count as 0.2

TASK = """TASK
You are analysing a regression benchmark. Features are f0-f9; models are ridge, random_forest, hist_gb and decision_tree, scored with R² (higher is better). Identify the mechanisms behind the target:
- linear: a feature with a straight-line effect
- nonlinear: a feature with a curved or threshold effect
- interaction: two features whose effect depends on each other
- correlated: two features carrying the same information
- overfit: a model that fits its training data much better than the evaluation data
- noise: a feature with no effect
Report only findings the data supports."""

FINAL_FORMAT = """FINAL REPLY FORMAT
Reply with JSON only:
{"findings": [{"features": ["f1"], "model": "<only for overfit>", "mechanism": "linear|nonlinear|interaction|correlated|overfit|noise", "reason": "<one sentence>"}]}
Each finding names exactly the features its mechanism needs: one for linear, nonlinear and noise; two for interaction and correlated; none ("features": []) for overfit. A finding with any other number of features counts as false."""

EVIDENCE_LEGEND = """mean_abs: mean |SHAP value| of the feature on evaluation data. direction: Spearman correlation between the feature's value and its SHAP value. correlated_pairs: feature pairs with |Pearson r| >= 0.5 on evaluation data."""

ACTIONS = """ACTIONS
Before your final reply you may run up to 3 experiments, one per reply. State what you expect before it runs:
{"hypothesis": "<one sentence>", "claim": <one finding in the final format>, "prediction": {"path": "<path into the action result, e.g. linearity_r2 or bins[0].mean_shap>", "op": ">" | ">=" | "<" | "<=", "value": <number>}, "action": <one action>}
Code runs the action and decides the verdict (confirmed, refuted, or invalid); you see it in HISTORY. An invalid experiment ends experimenting.
Actions:
- {"name": "shap_dependence", "model": M, "feature": F} -> {"bins": [{"x_mid", "mean_shap"} for each feature decile], "linearity_r2": R² of a straight-line fit of SHAP on the feature}
- {"name": "shap_interaction", "model": <random_forest|hist_gb|decision_tree>, "features": [F1, F2]} -> {"strength": mean |SHAP interaction value|, "share": its share of total attribution}
- {"name": "refit", "model": M, "params": {...}} -> {"train_r2", "eval_r2"}; params: ridge "alpha" (> 0); tree models "max_depth", "min_samples_leaf" (integers >= 1)
- {"name": "merge_rank", "features": [F1, F2, ...]} -> FeatRanker re-rank with those features as one group: {"merged_median_rank", "merged_position", "member_median_ranks"}"""


# --- benchmark (D2) ---------------------------------------------------------


def make_benchmark(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_ROWS, N_FEATURES))
    order = rng.permutation(N_FEATURES)  # roles move between seeds
    a, b, c, d, e, f = order[:6]
    X[:, f] = X[:, e] + 0.1 * rng.standard_normal(N_ROWS)
    y = (
        2 * X[:, a] + 3 * (X[:, b] > 0) + 2 * X[:, c] * X[:, d] + 1.5 * X[:, e]
        + rng.standard_normal(N_ROWS)
    )
    names = [f"f{i}" for i in range(N_FEATURES)]
    roles = {
        "linear": names[a],
        "nonlinear": names[b],
        "interaction": [names[c], names[d]],
        "correlated": [names[e], names[f]],  # e has the effect, f copies it
        "noise": sorted(names[i] for i in order[6:]),
    }
    return {
        "seed": seed, "names": names, "roles": roles,
        "X_train": X[:N_TRAIN], "X_eval": X[N_TRAIN:],
        "y_train": y[:N_TRAIN], "y_eval": y[N_TRAIN:],
    }


def _key(finding: Dict[str, Any]) -> str:
    if finding["mechanism"] == "overfit":
        return f"overfit:{finding.get('model')}"
    return f"{finding['mechanism']}:{'+'.join(sorted(set(finding['features'])))}"


def scored_items(bench: Dict[str, Any]) -> List[str]:
    r = bench["roles"]
    return sorted(_key(item) for item in (
        {"features": [r["linear"]], "mechanism": "linear"},
        {"features": [r["nonlinear"]], "mechanism": "nonlinear"},
        {"features": r["interaction"], "mechanism": "interaction"},
        {"features": r["correlated"], "mechanism": "correlated"},
        {"features": [], "mechanism": "overfit", "model": "decision_tree"},
    ))


def is_false(finding: Dict[str, Any], bench: Dict[str, Any], gaps: Dict[str, float]) -> bool:
    """D2: contradicts ground truth. Unlisted cases (e.g. linear on the copy f) are neutral."""
    r, mechanism, features = bench["roles"], finding["mechanism"], set(finding["features"])
    if len(features) != ARITY[mechanism]:
        return True
    if mechanism == "overfit":
        return gaps.get(finding.get("model"), -math.inf) < OVERFIT_GAP
    signal = {r["linear"], r["nonlinear"], *r["interaction"], r["correlated"][0]}
    if mechanism == "noise":
        return bool(features & signal)
    if features & set(r["noise"]):
        return True
    if mechanism == "linear":
        return bool(features & {r["nonlinear"], *r["interaction"]})
    if mechanism == "nonlinear":
        return bool(features & {r["linear"], r["correlated"][0]})
    if mechanism == "interaction":
        return features != set(r["interaction"])
    return features != set(r["correlated"])


def score(findings: List[Dict[str, Any]], bench: Dict[str, Any], gaps: Dict[str, float]) -> Dict[str, Any]:
    truth, matched, false = set(scored_items(bench)), set(), 0
    for finding in findings:
        if is_false(finding, bench, gaps):
            false += 1
        elif _key(finding) in truth:
            matched.add(_key(finding))
    return {"recall": len(matched) / len(truth), "matched": sorted(matched), "false_findings": false}


# --- replies ----------------------------------------------------------------


def _json(text: str) -> Any:
    """The outermost {...} in a reply; fences and surrounding prose are ignored."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _valid_finding(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("mechanism") not in MECHANISMS:
        return False
    features = item.get("features")
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        return False
    if item["mechanism"] == "overfit":
        return isinstance(item.get("model"), str)
    return bool(features)


def parse_findings(text: str) -> Optional[List[Dict[str, Any]]]:
    """Valid findings, or None when the reply is not a findings object."""
    data = _json(text)
    if not isinstance(data, dict) or not isinstance(data.get("findings"), list):
        return None
    return [item for item in data["findings"] if _valid_finding(item)]


# --- lab: fitted models, SHAP, evidence battery (D3, D4) --------------------


def _models() -> Dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=0),
        "hist_gb": HistGradientBoostingRegressor(random_state=0),
        "decision_tree": DecisionTreeRegressor(random_state=0),
    }


def _r2(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    return round(float(r2_score(y, model.predict(X))), 4)


def _spearman(x: np.ndarray, s: np.ndarray) -> float:
    if np.ptp(s) == 0:  # a feature the model never uses
        return 0.0
    return round(float(spearmanr(x, s).statistic), 4)


def build_lab(seed: int) -> Dict[str, Any]:
    bench = make_benchmark(seed)
    X_train, X_eval = bench["X_train"], bench["X_eval"]
    ranker = FeatureRanker(task="reg", models=_models(), verbose=False)
    ranker.fit(X_train, bench["y_train"], feature_names=bench["names"])
    report = ranker.rank_features(X_eval, bench["y_eval"], scoring="r2", random_state=0)
    models = ranker.trained_models
    values = {
        # Full background: the default subsamples 100 rows, which would break replay.
        name: (shap.LinearExplainer(model, shap.maskers.Independent(X_train, max_samples=len(X_train)))
               if name == "ridge"
               else shap.TreeExplainer(model)).shap_values(X_eval)
        for name, model in models.items()
    }
    scores = {
        name: {"train_r2": _r2(m, X_train, bench["y_train"]), "eval_r2": _r2(m, X_eval, bench["y_eval"])}
        for name, m in models.items()
    }
    corr = np.corrcoef(X_eval, rowvar=False)
    names = bench["names"]
    battery = {
        "scores": scores,
        "shap": {
            name: {
                feature: {
                    "mean_abs": round(float(np.abs(values[name][:, i]).mean()), 4),
                    "direction": _spearman(X_eval[:, i], values[name][:, i]),
                }
                for i, feature in enumerate(names)
            }
            for name in models
        },
        "correlated_pairs": [
            {"features": [names[i], names[j]], "r": round(float(corr[i, j]), 4)}
            for i in range(N_FEATURES) for j in range(i + 1, N_FEATURES)
            if abs(corr[i, j]) >= CORRELATION_MIN
        ],
    }
    return {
        "bench": bench, "ranker": ranker, "report": report, "models": models,
        "shap": values, "interactions": {}, "battery": battery,
        "gaps": {name: s["train_r2"] - s["eval_r2"] for name, s in scores.items()},
    }


# --- actions (D4) -----------------------------------------------------------


def _model(action: Dict[str, Any], lab: Dict[str, Any], tree: bool = False) -> str:
    name = action.get("model")
    if name not in lab["models"]:
        raise ValueError(f"unknown model {name!r}")
    if tree and name not in TREE_MODELS:
        raise ValueError(f"{name} is not a tree model")
    return name


def _features(value: Any, lab: Dict[str, Any], minimum: int) -> List[int]:
    names = lab["bench"]["names"]
    if not isinstance(value, list) or len(value) < minimum or len(set(value)) != len(value):
        raise ValueError(f"need a list of at least {minimum} distinct features")
    for feature in value:
        if feature not in names:
            raise ValueError(f"unknown feature {feature!r}")
    return [names.index(feature) for feature in value]


def _shap_dependence(action: Dict[str, Any], lab: Dict[str, Any]) -> Dict[str, Any]:
    model = _model(action, lab)
    (i,) = _features([action.get("feature")], lab, 1)
    x, s = lab["bench"]["X_eval"][:, i], lab["shap"][model][:, i]
    edges = np.quantile(x, np.linspace(0, 1, 11))
    decile = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    bins = [
        {"x_mid": round(float(x[decile == k].mean()), 4),
         "mean_shap": round(float(s[decile == k].mean()), 4)}
        for k in range(10)
    ]
    r2 = 0.0 if np.ptp(s) == 0 else float(np.corrcoef(x, s)[0, 1] ** 2)
    return {"bins": bins, "linearity_r2": round(r2, 4)}


def _shap_interaction(action: Dict[str, Any], lab: Dict[str, Any]) -> Dict[str, Any]:
    model = _model(action, lab, tree=True)
    features = action.get("features")
    if not isinstance(features, list) or len(features) != 2:
        raise ValueError("shap_interaction needs exactly two features")
    i, j = _features(features, lab, 2)
    if model not in lab["interactions"]:  # deterministic cache, so replay matches
        rows = lab["bench"]["X_eval"][:INTERACTION_ROWS]
        lab["interactions"][model] = shap.TreeExplainer(lab["models"][model]).shap_interaction_values(rows)
    values = lab["interactions"][model]
    strength = float(np.abs(values[:, i, j]).mean() * 2)  # both off-diagonal cells
    total = float(np.abs(values).sum(axis=(1, 2)).mean())
    return {"strength": round(strength, 4), "share": round(strength / total if total else 0.0, 4)}


def _refit(action: Dict[str, Any], lab: Dict[str, Any]) -> Dict[str, Any]:
    model = _model(action, lab)
    params = action.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("refit needs a non-empty params object")
    for key, value in params.items():
        if key not in REFIT_PARAMS[model]:
            raise ValueError(f"{model} does not accept {key!r}")
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "alpha" and not (is_number and value > 0):
            raise ValueError("alpha must be a number > 0")
        if key != "alpha" and not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            raise ValueError(f"{key} must be an integer >= 1")
    bench = lab["bench"]
    fitted = clone(lab["models"][model]).set_params(**params).fit(bench["X_train"], bench["y_train"])
    return {"train_r2": _r2(fitted, bench["X_train"], bench["y_train"]),
            "eval_r2": _r2(fitted, bench["X_eval"], bench["y_eval"])}


def _merge_rank(action: Dict[str, Any], lab: Dict[str, Any]) -> Dict[str, Any]:
    features = action.get("features")
    _features(features, lab, 2)
    group = "+".join(features)
    bench = lab["bench"]
    report = lab["ranker"].rank_features(
        bench["X_eval"], bench["y_eval"], scoring="r2",
        feature_groups={group: features}, random_state=0,
    )
    position, merged = next(
        ((k, row) for k, row in enumerate(report["consensus"], 1) if row["feature_group"] == group),
        (None, {"median_rank": None}),
    )
    before = {row["feature_group"]: row["median_rank"] for row in lab["report"]["consensus"]}
    return {"merged_median_rank": merged["median_rank"], "merged_position": position,
            "member_median_ranks": {feature: before.get(feature) for feature in features}}


ACTION_HANDLERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "shap_dependence": _shap_dependence,
    "shap_interaction": _shap_interaction,
    "refit": _refit,
    "merge_rank": _merge_rank,
}


def run_action(action: Any, lab: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict) or action.get("name") not in ACTION_HANDLERS:
        raise ValueError(f"unknown action {action!r:.80}")
    return ACTION_HANDLERS[action["name"]](action, lab)


# --- arms, loop, trail (D4, D5) ---------------------------------------------


def evaluate(prediction: Any, result: Dict[str, Any]) -> str:
    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if not isinstance(prediction, dict) or prediction.get("op") not in OPS:
        return "invalid"
    if not isinstance(prediction.get("path"), str) or not is_number(prediction.get("value")):
        return "invalid"
    found, actual = _resolve(result, prediction["path"])
    if not found or not is_number(actual):
        return "invalid"
    return "confirmed" if OPS[prediction["op"]](actual, prediction["value"]) else "refuted"


def _run_step(data: Any, lab: Dict[str, Any], turn: int) -> Dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    entry = {"turn": turn, **{k: data.get(k) for k in ("hypothesis", "claim", "prediction", "action")}}
    if not _valid_finding(data.get("claim")):
        return {**entry, "result": {"error": "claim missing or malformed"}, "verdict": "invalid"}
    try:
        result = run_action(data.get("action"), lab)
    except Exception as error:  # an LLM-chosen action must never crash the run
        return {**entry, "result": {"error": f"{type(error).__name__}: {error}"}, "verdict": "invalid"}
    return {**entry, "result": result, "verdict": evaluate(data.get("prediction"), result)}


def _prompt(arm: str, lab: Dict[str, Any]) -> str:
    parts = [TASK, "DATA (FeatRanker report)\n" + json.dumps(_data_view(lab["report"]))]
    if arm in ("evidence", "investigator"):
        parts.append("EVIDENCE (computed by code)\n" + EVIDENCE_LEGEND + "\n" + json.dumps(lab["battery"]))
    if arm == "investigator":
        parts.append(ACTIONS)
    parts.append(FINAL_FORMAT)
    return "\n\n".join(parts)


def _turn(trail: List[Dict[str, Any]], final: bool) -> str:
    history = "\n\nHISTORY\n" + json.dumps(trail) if trail else ""
    if final:
        return history + "\n\nNo experiments left. Reply with your final findings now."
    left = MAX_FOLLOWUPS - len(trail)
    return history + f"\n\nExperiments left: {left}. Reply with one experiment or your final findings."


def run_arm(arm: str, lab: Dict[str, Any], llm: Callable[[str], str]) -> Dict[str, Any]:
    base = _prompt(arm, lab)
    if arm != "investigator":
        return {"findings": parse_findings(llm(base)), "trail": [], "calls": 1}
    trail: List[Dict[str, Any]] = []
    calls = 0
    while True:
        final = len(trail) >= MAX_FOLLOWUPS or bool(trail and trail[-1]["verdict"] == "invalid")
        reply = llm(base + _turn(trail, final))
        calls += 1
        data = _json(reply)
        if final or (isinstance(data, dict) and "findings" in data):
            return {"findings": parse_findings(reply), "trail": trail, "calls": calls}
        trail.append(_run_step(data, lab, len(trail) + 1))


def replay(trail: List[Dict[str, Any]], lab: Dict[str, Any]) -> List[tuple]:
    """Recompute every result and verdict from the recorded steps, without an LLM."""
    return [
        (step["result"], step["verdict"])
        for step in (_run_step(entry, lab, entry["turn"]) for entry in trail)
    ]


def confirmed_false(trail: List[Dict[str, Any]], bench: Dict[str, Any], gaps: Dict[str, float]) -> int:
    return sum(
        1 for entry in trail
        if entry["verdict"] == "confirmed" and _valid_finding(entry.get("claim"))
        and is_false(entry["claim"], bench, gaps)
    )


# --- runs and the decision rule (D6) ----------------------------------------


def run_seed(seed: int, llm: Callable[[str], str], arms: tuple = ARMS) -> Dict[str, Any]:
    lab = build_lab(seed)
    results = {}
    for arm in arms:
        run = run_arm(arm, lab, llm)
        scored = score(run["findings"] or [], lab["bench"], lab["gaps"])
        scored["invalid_final"] = run["findings"] is None
        if arm == "investigator":
            scored["confirmed_false"] = confirmed_false(run["trail"], lab["bench"], lab["gaps"])
        results[arm] = {**run, **scored}
    return {"seed": seed, "roles": lab["bench"]["roles"], "scores": lab["battery"]["scores"], "arms": results}


def decide(rows: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """D6. rows: one {arm: {recall, false_findings[, confirmed_false]}} per seed."""
    recall = {arm: float(np.mean([r[arm]["recall"] for r in rows])) for arm in ARMS}
    false = {arm: float(np.mean([r[arm]["false_findings"] for r in rows])) for arm in ARMS}
    wins = sum(r["investigator"]["recall"] > r["evidence"]["recall"] + EPS for r in rows)
    confirmed = sum(r["investigator"].get("confirmed_false", 0) for r in rows)
    gaps = {
        "investigator_vs_evidence": recall["investigator"] - recall["evidence"],
        "investigator_vs_narrator": recall["investigator"] - recall["narrator"],
        "evidence_vs_narrator": recall["evidence"] - recall["narrator"],
    }
    loop_ok = (
        gaps["investigator_vs_evidence"] >= GAP_MIN - EPS
        and wins >= math.ceil(WIN_SHARE * len(rows))
        and gaps["investigator_vs_narrator"] >= GAP_MIN - EPS
        and false["investigator"] <= FALSE_MAX + EPS
        and confirmed <= CONFIRMED_FALSE_MAX
    )
    evidence_ok = gaps["evidence_vs_narrator"] >= GAP_MIN - EPS and false["evidence"] <= FALSE_MAX + EPS
    outcome = "go_1.1_1.2" if loop_ok else "go_1.1" if evidence_ok else "no_go"
    borderline = any(abs(gap - GAP_MIN) <= BORDER + EPS for gap in gaps.values()) or any(
        abs(false[arm] - FALSE_MAX) <= BORDER + EPS for arm in ("evidence", "investigator")
    )
    return {"outcome": outcome, "borderline": borderline, "mean_recall": recall, "mean_false": false,
            "gaps": gaps, "investigator_wins": wins, "confirmed_false": confirmed}
