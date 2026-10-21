"""
scikit-learn feature selector built on FeatureRanker.rank_resampled().
"""

import math
import numbers
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .evidence import auto_groups, auto_groups_record
from .importance import FeatureRanker


FITTED_ATTRIBUTES = (
    "support_", "selected_groups_", "report_", "n_features_in_", "feature_names_in_",
)


def _string_columns(X: Any) -> Any:
    """A DataFrame with str column labels (caller's frame untouched); others as is."""
    if not hasattr(X, "columns"):
        return X
    names = [str(column) for column in X.columns]
    if len(set(names)) != len(names):
        raise ValueError("[!] Feature names must be unique.")
    return X.set_axis(names, axis=1)


class ConsensusSelector(SelectorMixin, BaseEstimator):
    """
    Select feature groups by resampled consensus ranking, inside fit().

    Use it as a Pipeline step: under cross_val_score or GridSearchCV the
    ranking is redone on each training fold, so selection never sees the
    rows it is scored on.

    Args:
        task: "clf" or "reg".
        scoring: scikit-learn scorer name or callable.
        k: Keep the first k groups in the stability order.
        min_shadow_share: Keep groups whose above_shadow_share is at least
            this (turns shadow features on). With k too, filter then take k.
        max_shadow_q: Keep groups whose shadow_q_value (Benjamini-Hochberg
            adjusted shadow hits test across splits) is at most this, in
            (0, 1]. A heuristic filter: the values are uncalibrated reference
            values, not significance levels. Turns shadow features on;
            combines with the other rules like min_shadow_share. Needs about
            10 or more splits to have power.
        min_shadow_families: Keep groups for which at least this many model
            families (a positive int) are above their shadows in a majority
            of splits (stability shadow_families; splits where a family did
            not vote count as not above). 1 asks whether any one family
            finds signal; 2 is a heuristic starting point, since one family
            fitting noise can pass alone. Its false-selection rate is not
            yet measured (see the calibration study). Turns shadow features
            on; combines with the other rules.
        group, models, n_jobs, shadow_seed, shadow_mode, missing, categorical, model_input:
            As in FeatureRanker. With missing="impute", NaN is accepted in
            fit() and transform() (the selector only picks columns), so put an
            imputer after it in a Pipeline for the final model. With
            categorical, selected raw columns come back as given (use an
            encoder after it in a Pipeline for the final model).
        feature_groups, cv, n_repeats, random_state: As in rank_resampled().
            feature_groups="auto" groups correlated columns with auto_groups()
            on the X given to fit(), so inside a Pipeline from the training
            fold only; the groups used are in report_["feature_groups"] and
            the threshold in report_["run"]["auto_groups"].
        auto_groups_threshold: The |rho| threshold of feature_groups="auto"
            (see auto_groups()); ignored for other feature_groups.
        verbose: Show progress bars (off by default; pipelines fit often).
    """

    def __init__(
        self,
        task: Literal["clf", "reg"],
        scoring: Any,
        *,
        k: Optional[Union[int, "np.integer[Any]"]] = None,
        min_shadow_share: Optional[float] = None,
        max_shadow_q: Optional[float] = None,
        min_shadow_families: Optional[int] = None,
        group: Literal["linear", "tree", "all", "fast"] = "all",
        models: Optional[Dict[str, Any]] = None,
        feature_groups: Union[Dict[str, List[str]], Literal["auto"], None] = None,
        auto_groups_threshold: float = 0.7,
        cv: Any = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        shadow_seed: Optional[int] = None,
        shadow_mode: Literal["joint", "independent"] = "joint",
        missing: Literal["error", "impute"] = "error",
        categorical: Union[List[str], Literal["auto"], None] = None,
        model_input: Literal["array", "dataframe"] = "array",
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # scikit-learn convention: store raw, validate in fit().
        self.task = task
        self.scoring = scoring
        self.k = k
        self.min_shadow_share = min_shadow_share
        self.max_shadow_q = max_shadow_q
        self.min_shadow_families = min_shadow_families
        self.group = group
        self.models = models
        self.feature_groups = feature_groups
        self.auto_groups_threshold = auto_groups_threshold
        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.shadow_seed = shadow_seed
        self.shadow_mode = shadow_mode
        self.missing = missing
        self.categorical = categorical
        self.model_input = model_input
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Selectors pickled by 1.3 lack the 1.4 max_shadow_q parameter, and
        # those pickled before 1.6 lack min_shadow_families and shadow_mode, and
        # those pickled before 1.7 lack missing, those before 1.9 lack categorical, those before 1.10 lack model_input, and those before 1.12 lack auto_groups_threshold.
        defaults = {
            "max_shadow_q": None, "min_shadow_families": None, "shadow_mode": "joint",
            "missing": "error", "categorical": None, "model_input": "array",
            "auto_groups_threshold": 0.7,
        }
        super().__setstate__({**defaults, **state})

    def _validate_rule(self) -> None:
        k, share, q = self.k, self.min_shadow_share, self.max_shadow_q
        families = self.min_shadow_families
        if k is None and share is None and q is None and families is None:
            raise ValueError(
                "[!] Set k, min_shadow_share, max_shadow_q, or min_shadow_families."
            )
        if k is not None and (
            not isinstance(k, numbers.Integral) or isinstance(k, bool) or k < 1
        ):
            raise ValueError("[!] k must be a positive integer.")
        if share is not None and (
            not isinstance(share, numbers.Real)
            or isinstance(share, bool)
            or math.isnan(share)
            or not 0 <= share <= 1
        ):
            raise ValueError("[!] min_shadow_share must be a number in [0, 1].")
        if q is not None and (
            not isinstance(q, numbers.Real)
            or isinstance(q, bool)
            or math.isnan(q)
            or not 0 < q <= 1
        ):
            raise ValueError("[!] max_shadow_q must be a number in (0, 1].")
        if families is not None and (
            not isinstance(families, numbers.Integral)
            or isinstance(families, bool)
            or families < 1
        ):
            raise ValueError("[!] min_shadow_families must be a positive integer.")

    def fit(
        self, X: Any, y: Any, groups: Any = None, sample_weight: Any = None
    ) -> "ConsensusSelector":
        """
        Rank on X/y only (resampled) and keep the groups the rule selects.

        groups: row groups for a group-aware cv (e.g. GroupKFold); in a
        Pipeline pass it as select__groups. sample_weight: row weights, as in
        rank_resampled(); in a Pipeline, enable scikit-learn metadata routing
        and call set_fit_request(sample_weight=True) on the selector.
        """
        # A failed refit must not leave the previous fit's state behind.
        for attribute in FITTED_ATTRIBUTES:
            self.__dict__.pop(attribute, None)
        self._validate_rule()
        X = _string_columns(X)
        if hasattr(X, "columns"):
            names = list(X.columns)
            feature_names = None
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("[!] X must be a two-dimensional feature matrix.")
            # Same default names scikit-learn uses in get_feature_names_out().
            names = [f"x{index}" for index in range(X.shape[1])]
            feature_names = names

        feature_groups = self.feature_groups
        is_auto = isinstance(feature_groups, str) and feature_groups == "auto"
        if is_auto:
            feature_groups = auto_groups(
                X,
                threshold=self.auto_groups_threshold,
                feature_names=names,
                categorical=self.categorical,
            )
        elif feature_groups is not None and not isinstance(feature_groups, dict):
            error = ValueError if isinstance(feature_groups, str) else TypeError
            raise error("[!] feature_groups must be a dict, None, or 'auto'.")

        ranker = FeatureRanker(
            self.task,
            self.group,
            models=self.models,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            shadows=any(
                rule is not None
                for rule in (self.min_shadow_share, self.max_shadow_q, self.min_shadow_families)
            ),
            shadow_seed=self.shadow_seed,
            shadow_mode=self.shadow_mode,
            missing=self.missing,
            categorical=self.categorical,
            model_input=self.model_input,
        )
        report = ranker.rank_resampled(
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            groups=groups,
            feature_names=feature_names,
            feature_groups=feature_groups,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            # k may be a NumPy integer (e.g. from a GridSearchCV grid).
            top_k=int(self.k) if self.k is not None else 5,
            sample_weight=sample_weight,
        )

        if is_auto:
            report["run"]["auto_groups"] = auto_groups_record(self.auto_groups_threshold)
        silent = report["n_splits"] - sum(1 for s in report["splits"] if s["report"]["consensus"])
        if silent:
            warnings.warn(
                f"{silent} of {report['n_splits']} splits did not vote (failed or "
                "produced no consensus); their shares count them as misses.",
                UserWarning,
                stacklevel=2,
            )

        rows = report["stability"]
        if self.min_shadow_share is not None:
            rows = [r for r in rows if r["above_shadow_share"] >= self.min_shadow_share]
        if self.max_shadow_q is not None:
            rows = [r for r in rows if r["shadow_q_value"] <= self.max_shadow_q]
        if self.min_shadow_families is not None:
            rows = [r for r in rows if r["shadow_families"] >= self.min_shadow_families]
        if self.k is not None:
            rows = rows[: int(self.k)]
        selected = [row["feature_group"] for row in rows]
        kept = {
            column for group in selected for column in report["feature_groups"][group]
        }
        # Only now, after ranking and selection succeeded, set fitted state.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.selected_groups_ = selected
        self.support_ = np.array([name in kept for name in names])
        self.n_features_in_ = len(names)
        self.report_ = report
        if not selected:
            message = "No feature group was selected; transform() will return no columns."
            if self.max_shadow_q is not None:
                message += (
                    f" With {report['n_splits']} splits the smallest possible p-value"
                    f" is {report['shadow_test']['min_p_value']:.3f}; use more splits,"
                    " e.g. RepeatedKFold."
                )
            warnings.warn(
                message,
                UserWarning,
                stacklevel=2,
            )
        return self

    def transform(self, X: Any) -> Any:
        # Match fit(): DataFrame column labels are compared as strings.
        return super().transform(_string_columns(X))

    def _more_tags(self) -> Dict[str, Any]:  # scikit-learn < 1.6
        return {"requires_y": True, "allow_nan": self._allows_nan()}

    def __sklearn_tags__(self) -> Any:  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = self._allows_nan()
        return tags

    def _allows_nan(self) -> bool:
        # A missing categorical value is a level, so transform() must let it pass.
        return self.missing == "impute" or self.categorical is not None

    def _get_support_mask(self) -> Any:
        check_is_fitted(self, "support_")
        return self.support_
