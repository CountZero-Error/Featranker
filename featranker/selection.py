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
        group, models, n_jobs, shadow_seed: As in FeatureRanker.
        feature_groups, cv, n_repeats, random_state: As in rank_resampled().
        verbose: Show progress bars (off by default; pipelines fit often).
    """

    def __init__(
        self,
        task: Literal["clf", "reg"],
        scoring: Any,
        *,
        k: Optional[Union[int, "np.integer[Any]"]] = None,
        min_shadow_share: Optional[float] = None,
        group: Literal["linear", "tree", "all", "fast"] = "all",
        models: Optional[Dict[str, Any]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        cv: Any = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        shadow_seed: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # scikit-learn convention: store raw, validate in fit().
        self.task = task
        self.scoring = scoring
        self.k = k
        self.min_shadow_share = min_shadow_share
        self.group = group
        self.models = models
        self.feature_groups = feature_groups
        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.shadow_seed = shadow_seed
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _validate_rule(self) -> None:
        k, share = self.k, self.min_shadow_share
        if k is None and share is None:
            raise ValueError("[!] Set k, min_shadow_share, or both.")
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

    def fit(self, X: Any, y: Any, groups: Any = None) -> "ConsensusSelector":
        """
        Rank on X/y only (resampled) and keep the groups the rule selects.

        groups: row groups for a group-aware cv (e.g. GroupKFold); in a
        Pipeline pass it as select__groups.
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

        ranker = FeatureRanker(
            self.task,
            self.group,
            models=self.models,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            shadows=self.min_shadow_share is not None,
            shadow_seed=self.shadow_seed,
        )
        report = ranker.rank_resampled(
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            groups=groups,
            feature_names=feature_names,
            feature_groups=self.feature_groups,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            # k may be a NumPy integer (e.g. from a GridSearchCV grid).
            top_k=int(self.k) if self.k is not None else 5,
        )

        rows = report["stability"]
        if self.min_shadow_share is not None:
            rows = [r for r in rows if r["above_shadow_share"] >= self.min_shadow_share]
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
            warnings.warn(
                "No feature group was selected; transform() will return no columns.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def transform(self, X: Any) -> Any:
        # Match fit(): DataFrame column labels are compared as strings.
        return super().transform(_string_columns(X))

    def _more_tags(self) -> Dict[str, Any]:  # scikit-learn < 1.6
        return {"requires_y": True}

    def __sklearn_tags__(self) -> Any:  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def _get_support_mask(self) -> Any:
        check_is_fitted(self, "support_")
        return self.support_
