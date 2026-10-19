"""
Provider-neutral LLM interpretation of FeatRanker reports.

FeatRanker never calls a model, holds a key, or imports an SDK: to_prompt()
builds text, and explain() calls a function the user supplies.
"""

import builtins
import copy
import json
import math
import re
import warnings
from typing import Any, Callable, Dict, List

from sklearn.metrics import get_scorer_names

from .importance import BASELINE_NAME, _registry_families
from .summary import summarize

RULES = """RULES
You are interpreting a FeatRanker feature-ranking report. Follow these rules:
1. Rankings describe association with model performance on this evaluation data; they are not causal effects. Never use causal language such as "causes", "drives", or "leads to".
2. Equal median_rank ties the primary score only. Single-split consensus uses mean_rank next; stability uses top_k_share. Small gaps alone do not establish statistical significance.
3. Importance near zero, or a mean no greater than its std, is not proof that a feature is unused; correlated features share credit.
4. n_models measures coverage, not independent confirmation; n_families counts more independent views.
5. above_shadow_share and stability shares such as top_k_share are descriptive, not p-values or significance tests.
6. State only numbers that appear in the DATA section, and cite the path and value of every number you use."""

REPLY_FORMAT = """REPLY FORMAT
Reply with JSON only, in this form:
{"summary": "<2-4 sentences>", "claims": [{"text": "<one claim>", "evidence": [{"path": "<path into DATA, e.g. consensus[0].median_rank>", "value": <the value at that path>}]}]}
Paths use dot keys and [i] list indices. For keys containing ".", "[", or "/", use a JSON Pointer instead, e.g. /models/model.v1/importance/lab[0]/mean (write "~1" for "/" and "~0" for "~")."""

ANONYMIZED_NAMES = """Names are anonymized. In summary and claim text, wrap aliases present in DATA as [[g1]], [[c1]], [[m1]], or [[f1]]. Keep unaliased built-in names unchanged. Do not invent aliases; keep evidence paths and values exactly as shown in DATA."""

# Exception class names are caller-controlled text too; only public ones pass.
PUBLIC_ERROR_TYPES = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
) | {"NotFittedError"}


def _model_view(model: Dict[str, Any]) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        key: model[key]
        for key in ("family", "evaluation_score", "in_consensus", "shadow_max")
        if key in model
    }
    # Per-repeat values and fitted params are noise for interpretation.
    view["importance"] = {
        group: {key: value for key, value in entry.items() if key != "values"}
        for group, entry in model["importance"].items()
    }
    return view


def _failure_view(failure: Dict[str, Any]) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        key: failure[key]
        for key in ("split", "model", "stage", "error_type")
        if key in failure
    }
    if "error_type" in view and view["error_type"] not in PUBLIC_ERROR_TYPES:
        view["error_type"] = "CustomError"
    return view


def _data_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """The compact JSON the model sees; evidence paths are checked against it."""
    shared: Dict[str, Any] = {
        key: report[key]
        for key in ("evaluation_mode", "scoring", "n_repeats", "feature_groups", "shadows")
        if key in report
    }
    failures = [_failure_view(f) for f in report.get("failures", [])]
    if report.get("evaluation_mode") == "resampled":
        return {
            **shared,
            "n_splits": report["n_splits"],
            "top_k": report["top_k"],
            "splits": [
                {
                    "split": entry["split"],
                    "n_train": entry["n_train"],
                    "n_eval": entry["n_eval"],
                    "baseline_score": entry["report"]["baseline_score"],
                }
                for entry in report["splits"]
            ],
            "stability": report["stability"],
            "failures": failures,
        }
    return {
        **shared,
        "baseline_score": report["baseline_score"],
        "models": {name: _model_view(m) for name, m in report["models"].items()},
        "consensus": report["consensus"],
        "failures": failures,
    }


def _sanitized(report: Dict[str, Any]) -> Dict[str, Any]:
    """A copy whose failures carry no messages: those can hold names and data rows."""
    clean = copy.deepcopy(report)

    def strip(part: Dict[str, Any]) -> None:
        if "failures" in part:
            part["failures"] = [_failure_view(f) for f in part["failures"]]
        for entry in part.get("splits", []):
            strip(entry["report"])

    strip(clean)
    return clean


def _findings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    # summarize() reads single-split reports; resampled ones rely on stability.
    if report.get("evaluation_mode") == "resampled":
        return []
    return summarize(report)


def _all_parts(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The report and, for resampled reports, every split report."""
    return [report] + [entry["report"] for entry in report.get("splits", [])]


def _name_maps(report: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Aliases for caller-chosen identifiers: groups g1.., columns c1.., and
    custom models m1.. and families f1... Built-in registry model and family
    names (and the dummy) are fixed vocabulary and stay readable.
    """
    groups = report.get("feature_groups", {})
    maps: Dict[str, Dict[str, str]] = {
        "group": {group: f"g{i}" for i, group in enumerate(groups, start=1)},
        "column": {},
        "model": {},
        "family": {},
    }
    for members in groups.values():
        for column in members:
            maps["column"].setdefault(column, f"c{len(maps['column']) + 1}")
    registry = _registry_families()
    builtin_models = set(registry) | {BASELINE_NAME}
    builtin_families = set(registry.values())
    for part in _all_parts(report):
        names = list(part.get("models", {})) + [
            f.get("model") for f in part.get("failures", []) if f.get("model")
        ]
        for name in names:
            if name not in builtin_models:
                maps["model"].setdefault(name, f"m{len(maps['model']) + 1}")
        for model in part.get("models", {}).values():
            family = model.get("family")
            if family is not None and family not in builtin_families:
                maps["family"].setdefault(family, f"f{len(maps['family']) + 1}")
    return maps


def _anonymize(report: Dict[str, Any], maps: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Rename identifiers where reports store them, never by text search
    (a column named "fit" must not rename the "fit" failure stage)."""
    renamed = copy.deepcopy(report)
    group, column = maps["group"], maps["column"]

    def alias(kind: str, name: str) -> str:
        return maps[kind].get(name, name)

    def rename(part: Dict[str, Any]) -> None:
        if "feature_groups" in part:
            part["feature_groups"] = {
                group[g]: [column[c] for c in members]
                for g, members in part["feature_groups"].items()
            }
        if isinstance(part.get("scoring"), str) and part["scoring"] not in get_scorer_names():
            part["scoring"] = "custom_scorer"  # a callable's name is caller-chosen
        if "models" in part:
            part["models"] = {alias("model", n): m for n, m in part["models"].items()}
        for model in part.get("models", {}).values():
            if "family" in model:
                model["family"] = alias("family", model["family"])
            for key in ("importance", "shadow_importance"):
                if key in model:
                    model[key] = {group[g]: v for g, v in model[key].items()}
        for key in ("consensus", "stability"):
            for row in part.get(key, []):
                row["feature_group"] = group[row["feature_group"]]
                if "family_ranks" in row:
                    row["family_ranks"] = {
                        alias("family", f): r for f, r in row["family_ranks"].items()
                    }
        for failure in part.get("failures", []):
            if failure.get("model"):
                failure["model"] = alias("model", failure["model"])
        # summarize() names these groups in its correlated_groups finding.
        for pair in part.get("evidence", {}).get("correlation", {}).get("pairs", []):
            pair["groups"] = [group[g] for g in pair["groups"]]
        for entry in part.get("splits", []):
            rename(entry["report"])

    rename(renamed)
    return renamed


def _prepare(
    report: Any, anonymize: bool
) -> tuple[str, Dict[str, Any], Dict[str, str]]:
    """Prompt, data view, and token -> original name (empty unless anonymized)."""
    if not isinstance(report, dict):
        raise TypeError("[!] Expected a FeatRanker report dict.")
    clean = _sanitized(report)
    if not anonymize:
        prompt, view = _render(clean, anonymized=False)
        return prompt, view, {}
    maps = _name_maps(clean)
    prompt, view = _render(_anonymize(clean, maps), anonymized=True)
    # Tokens are distinct per kind (g1 vs c1), so every alias restores.
    reverse = {token: name for kind in maps.values() for name, token in kind.items()}
    return prompt, view, reverse


def _render(report: Dict[str, Any], *, anonymized: bool) -> tuple[str, Dict[str, Any]]:
    # A private snapshot: what is verified is exactly what the prompt showed,
    # even if the caller's report changes during the llm call.
    view = copy.deepcopy(_data_view(report))
    findings = _findings(report)
    findings_text = (
        json.dumps(findings, sort_keys=True, indent=1)
        if findings
        else "none (see DATA; resampled reports use the stability table)"
    )
    prompt = "\n\n".join(
        [
            RULES,
            "FINDINGS (from featranker.summarize)\n" + findings_text,
            "DATA:\n" + json.dumps(view, sort_keys=True, indent=1),
            REPLY_FORMAT + ("\n" + ANONYMIZED_NAMES if anonymized else ""),
        ]
    )
    return prompt, view


def to_prompt(report: Any, *, anonymize: bool = False) -> str:
    """
    Deprecated since 1.1, to be removed in 2.0.

    Build a deterministic LLM prompt for a FeatRanker report.

    The prompt carries FeatRanker's interpretation rules, the summarize()
    findings, a compact JSON view of the report, and a JSON reply format in
    which every claim cites the data it uses. No network call is made.
    With anonymize=True, group and column names become g1.. and c1...
    """
    warnings.warn(
        "to_prompt() is deprecated since 1.1 and will be removed in 2.0.",
        FutureWarning,
        stacklevel=2,
    )
    prompt, _, _ = _prepare(report, anonymize)
    return prompt


_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.S)
_PATH = re.compile(r"[^.\[\]]+(?:\.[^.\[\]]+|\[\d+\])*")
_PATH_TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")
_MARKER = re.compile(r"\[\[([gcmf]\d+)\]\]")
_POINTER_INDEX = re.compile(r"0|[1-9]\d*")
_REL_TOL = _ABS_TOL = 0.01  # two-decimal rounding passes; rank 3 vs 3.25 fails


def _parse_reply(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1)
    try:
        reply = json.loads(stripped)
    except json.JSONDecodeError:
        reply = None
    if not _valid_reply(reply):
        raise ValueError(
            "[!] The LLM reply is not the expected JSON "
            f"({{summary, claims[{{text, evidence[{{path, value}}]}}]}}): {text[:200]!r}"
        )
    return reply


def _valid_reply(reply: Any) -> bool:
    if not isinstance(reply, dict) or not isinstance(reply.get("summary"), str):
        return False
    if not isinstance(reply.get("claims"), list):
        return False
    for claim in reply["claims"]:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            return False
        if not isinstance(claim.get("evidence"), list):
            return False
        for item in claim["evidence"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                return False
            if "value" not in item or not _is_scalar(item["value"]):
                return False
    return True


def _is_scalar(value: Any) -> bool:
    """number (finite), string, bool, or null: never a list or object."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _resolve_pointer(data: Any, path: str) -> tuple[bool, Any]:
    """RFC 6901 JSON Pointer, resolved structurally (never evaluated)."""
    current = data
    for part in path[1:].split("/"):
        if re.search(r"~(?![01])", part):  # RFC 6901 allows only ~0 and ~1
            return False, None
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not _POINTER_INDEX.fullmatch(key) or int(key) >= len(current):
                return False, None
            current = current[int(key)]
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False, None
    return True, current


def _resolve(data: Any, path: str) -> tuple[bool, Any]:
    """Follow a dotted path or a JSON Pointer; (False, None) when it is not there."""
    if path.startswith("/"):
        return _resolve_pointer(data, path)
    # Only well-formed paths: key(.key | [i])*, e.g. consensus[0].median_rank.
    if not _PATH.fullmatch(path):
        return False, None
    tokens = _PATH_TOKEN.findall(path)
    current = data
    for token in tokens:
        if token.startswith("["):
            index = int(token[1:-1])
            if not isinstance(current, list) or index >= len(current):
                return False, None
            current = current[index]
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            return False, None
    return True, current


def _matches(actual: Any, cited: Any) -> bool:
    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if is_number(actual) and is_number(cited):
        try:
            return math.isclose(actual, cited, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
        except OverflowError:  # e.g. a cited 10**400
            return False
    # No cross-type equality: True must not match 1, nor "1" match 1.
    return type(actual) is type(cited) and actual == cited


def explain(
    report: Any, llm: Callable[[str], str], *, anonymize: bool = False
) -> Dict[str, Any]:
    """
    Deprecated since 1.1, to be removed in 2.0.

    Ask your LLM to interpret a report, then check every number it cites.

    llm is any function taking the prompt string and returning the reply
    string; FeatRanker makes no network call and needs no key. Returns the
    summary and claims, each evidence item and claim with a "verified" flag,
    and "all_verified". Verified numbers do not guarantee a sound reading.
    """
    warnings.warn(
        "explain() is deprecated since 1.1 and will be removed in 2.0.",
        FutureWarning,
        stacklevel=2,
    )
    if not callable(llm):
        raise TypeError("[!] llm must be a function taking a prompt string.")
    prompt, view, reverse = _prepare(report, anonymize)
    text = llm(prompt)
    if not isinstance(text, str):
        raise TypeError("[!] llm must return the reply as a string.")
    reply = _parse_reply(text)

    def restore(value: str) -> str:
        # Only explicit [[g1]]-style markers; plain text such as "c1" is untouched.
        return _MARKER.sub(lambda m: reverse.get(m.group(1), m.group(0)), value)

    claims = []
    for claim in reply["claims"]:
        evidence = []
        for item in claim["evidence"]:
            found, actual = _resolve(view, item["path"])
            evidence.append(
                {
                    "path": item["path"],
                    "value": item["value"],
                    "verified": found and _matches(actual, item["value"]),
                }
            )
        claims.append(
            {
                "text": restore(claim["text"]),
                "evidence": evidence,
                "verified": bool(evidence) and all(e["verified"] for e in evidence),
            }
        )
    return {
        "summary": restore(reply["summary"]),
        "claims": claims,
        "all_verified": all(claim["verified"] for claim in claims),
    }
