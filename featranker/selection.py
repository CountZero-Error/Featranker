"""
scikit-learn feature selector built on FeatureRanker.rank_resampled().
"""

import math
import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .evidence import _check_threshold, auto_groups, auto_groups_record
from .importance import (
    N_REPEATS_MESSAGE,
    FeatureRanker,
    MAX_COUNT,
    _check_cv,
    _is_int,
    _is_positive_int,
    _is_real,
    _plain_int,
    _resolve_max_samples,
    _resolve_scorers,
    _seed,
)


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
    Select feature groups by a resampled consensus ranking, inside ``fit()``.

    Use it as a ``Pipeline`` step. Under ``cross_val_score`` or
    ``GridSearchCV`` the ranking is redone on each training fold, so the
    selection never sees the rows it is scored on. Set at least one of ``k``,
    ``min_shadow_share``, ``max_shadow_q`` and ``min_shadow_families``; the
    shadow filters apply first, then the first ``k`` groups that pass.
    Invalid rules raise ``ValueError`` in ``fit()``, not here, following the
    scikit-learn convention that ``__init__`` only stores parameters.

    Parameters
    ----------
    task : {"clf", "reg"}
        As in ``FeatureRanker``. Required.
    scoring : str, callable or list
        As in ``FeatureRanker.rank_features()``. With a list, the first scorer
        selects and the rest are reported in ``report_``. Required.
    k : int, optional
        Keep the first ``k`` groups in stability order (all if fewer).
    min_shadow_share : float, optional
        Keep groups whose stability ``above_shadow_share`` is at least this
        (0 to 1). Turns shadow features on.
    max_shadow_q : float, optional
        Keep groups whose ``shadow_q_value`` is at most this (above 0, at most
        1). A heuristic filter on uncalibrated values; experimental, so
        ``fit()`` warns. Turns shadows on. Use 10 or more splits.
    min_shadow_families : int, optional
        Keep groups that at least this many model families put above their
        shadows in most splits (a positive integer). A heuristic. Turns
        shadows on.
    group : {"all", "linear", "tree", "fast"}, default "all"
        As in ``FeatureRanker``.
    models : dict, optional
        As in ``FeatureRanker``.
    feature_groups : dict or "auto", optional
        As in ``rank_resampled()``. ``"auto"`` groups correlated columns with
        ``auto_groups()`` on the data given to ``fit()``; the groups used are
        in ``report_["feature_groups"]``. Any other string raises
        ``ValueError`` in ``fit()``.
    auto_groups_threshold : float, default 0.7
        The |rho| cut of ``feature_groups="auto"`` (above 0, at most 1).
        Ignored for other ``feature_groups``.
    cv : int, splitter or iterable, default 5
        As in ``rank_resampled()``.
    n_repeats : int, default 10
        As in ``rank_resampled()``.
    random_state : int, optional
        As in ``rank_resampled()``.
    max_samples : int or float, optional
        As in ``rank_resampled()``.
    shadow_seed : int, optional
        As in ``FeatureRanker``.
    shadow_mode : {"joint", "independent"}, default "joint"
        As in ``FeatureRanker``.
    missing : {"error", "impute"}, default "error"
        As in ``FeatureRanker``. With ``"impute"`` NaN is accepted in ``fit()``
        and ``transform()``; put an imputer after the selector in a pipeline.
    categorical : "auto" or list of str, optional
        As in ``FeatureRanker``. Selected raw columns come back as given; put
        an encoder after the selector in a pipeline.
    model_input : {"array", "dataframe"}, default "array"
        As in ``FeatureRanker``.
    shuffle : {"rows", "within_groups", "circular"}, default "rows"
        As in ``FeatureRanker``. With ``"within_groups"`` the ``groups`` given
        to ``fit()`` are the shuffle groups and are required.
    families : dict, optional
        As in ``FeatureRanker``.
    max_categories : int, optional
        As in ``FeatureRanker``.
    max_train_samples : int or float, optional
        As in ``rank_resampled()``: each split's models are fit on a random
        subsample of its training rows.
    n_jobs : int, optional
        As in ``FeatureRanker``.
    verbose : bool, default False
        Show progress bars. Off by default because pipelines fit often.

    Attributes
    ----------
    selected_groups_ : list of str
        The groups kept, in stability order.
    report_ : Report
        The resampled report of the last ``fit()``.
    n_features_in_ : int
        Number of columns seen by ``fit()``.
    feature_names_in_ : ndarray
        Column names, for DataFrame input.

    See Also
    --------
    FeatureRanker.rank_resampled : the ranking this selector runs.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.pipeline import Pipeline
    >>> from featranker import ConsensusSelector
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> pipe = Pipeline([
    ...     ("select", ConsensusSelector("reg", "r2", k=3, group="fast", cv=3,
    ...                                  n_repeats=3, random_state=0)),
    ...     ("model", Ridge()),
    ... ]).fit(X, y)
    >>> len(pipe.named_steps["select"].selected_groups_)
    3
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
        max_samples: Any = None,
        shadow_seed: Optional[int] = None,
        shadow_mode: Literal["joint", "independent"] = "joint",
        missing: Literal["error", "impute"] = "error",
        categorical: Union[List[str], Literal["auto"], None] = None,
        model_input: Literal["array", "dataframe"] = "array",
        shuffle: Literal["rows", "within_groups", "circular"] = "rows",
        families: Optional[Mapping[str, str]] = None,
        max_categories: Optional[int] = None,
        max_train_samples: Any = None,
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
        self.max_samples = max_samples
        self.shadow_seed = shadow_seed
        self.shadow_mode = shadow_mode
        self.missing = missing
        self.categorical = categorical
        self.model_input = model_input
        self.shuffle = shuffle
        self.families = families
        self.max_categories = max_categories
        self.max_train_samples = max_train_samples
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Selectors pickled by 1.3 lack the 1.4 max_shadow_q parameter, and
        # those pickled before 1.6 lack min_shadow_families and shadow_mode, and
        # those pickled before 1.7 lack missing, those before 1.9 lack categorical, those before 1.10 lack model_input, those before 1.12 lack auto_groups_threshold, those before 1.13 lack max_samples, and those before 1.15 lack shuffle, those before 1.18 lack families, and those before 1.21 lack max_categories and max_train_samples.
        defaults = {
            "max_shadow_q": None, "min_shadow_families": None, "shadow_mode": "joint",
            "missing": "error", "categorical": None, "model_input": "array",
            "auto_groups_threshold": 0.7, "max_samples": None, "shuffle": "rows",
            "families": None, "max_categories": None, "max_train_samples": None,
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
            not _is_positive_int(_plain_int(k))
        ):
            raise ValueError("[!] k must be a positive integer below 2**63.")
        if share is not None and (
            not _is_real(share)
            or math.isnan(share)
            or not 0 <= share <= 1
        ):
            raise ValueError("[!] min_shadow_share must be a number in [0, 1].")
        if q is not None and (
            not _is_real(q)
            or math.isnan(q)
            or not 0 < q <= 1
        ):
            raise ValueError("[!] max_shadow_q must be a number in (0, 1].")
        if families is not None and (
            not _is_int(families)
            or not 1 <= families < MAX_COUNT
        ):
            raise ValueError("[!] min_shadow_families must be a positive integer below 2**63.")

    def fit(
        self, X: Any, y: Any, groups: Any = None, sample_weight: Any = None
    ) -> "ConsensusSelector":
        """
        Rank on ``X`` and ``y`` only (resampled) and keep the selected groups.

        Parameters
        ----------
        X : array-like or DataFrame
            Training features. Feature names come from DataFrame columns; for
            arrays they are ``x0``, ``x1``, and so on.
        y : array-like
            Training target.
        groups : array-like, optional
            Row groups for a group-aware ``cv`` such as ``GroupKFold``. In a
            pipeline pass ``select__groups=...``. With
            ``shuffle="within_groups"`` they also confine the shuffles.
        sample_weight : array-like, optional
            Row weights, as in ``rank_resampled()``. In a pipeline, enable
            scikit-learn metadata routing and call
            ``set_fit_request(sample_weight=True)`` on the selector.

        Returns
        -------
        ConsensusSelector
            The fitted selector. A failed ``fit()`` clears earlier fitted
            state.

        Raises
        ------
        ValueError
            For an invalid rule or option, or if every model dropped the
            shadow columns in every split while a shadow rule is set.
        """
        # A failed refit must not leave the previous fit's state behind.
        for attribute in FITTED_ATTRIBUTES:
            self.__dict__.pop(attribute, None)
        self._validate_rule()
        _seed(self.random_state, "random_state")
        _seed(self.shadow_seed, "shadow_seed")
        if not _is_positive_int(_plain_int(self.n_repeats)):
            raise ValueError(N_REPEATS_MESSAGE)
        for option in (self.max_samples, self.max_train_samples):
            if option is not None:
                _resolve_max_samples(option, 1)
        if self.max_shadow_q is not None:
            warnings.warn(
                "max_shadow_q is experimental: shadow q-values are uncalibrated "
                "reference values, not significance levels.",
                UserWarning,
                stacklevel=2,
            )
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
            shuffle=self.shuffle,
            families=self.families,
            max_categories=self.max_categories,
        )
        _check_cv(self.cv)
        _resolve_scorers(self.scoring)
        if isinstance(self.feature_groups, str) and self.feature_groups == "auto":
            _check_threshold(self.auto_groups_threshold)
        elif self.feature_groups is not None and not isinstance(self.feature_groups, dict):
            error = ValueError if isinstance(self.feature_groups, str) else TypeError
            raise error("[!] feature_groups must be a dict, None, or 'auto'.")
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

        report = ranker.rank_resampled(
            X,
            y,
            scoring=self.scoring,
            cv=self.cv,
            groups=groups,
            feature_names=feature_names,
            feature_groups=cast(Optional[Dict[str, List[str]]], feature_groups),  # "auto" resolved above
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            # k may be a NumPy integer (e.g. from a GridSearchCV grid).
            top_k=int(self.k) if self.k is not None else 5,
            sample_weight=sample_weight,
            max_samples=self.max_samples,
            max_train_samples=self.max_train_samples,
        )

        ungated = [s for s in report["splits"] if "baseline_score" in s["report"] and s["report"]["baseline_score"] is None]
        if ungated:
            reasons = {
                f"{f['error_type']}: {f['message']}"
                for s in ungated
                for f in s["report"]["failures"]
                if f["model"] == "dummy_baseline"
            }
            raise ValueError(
                "[!] The dummy baseline could not be scored in split(s) "
                f"{[s['split'] for s in ungated]}, so the consensus gate is off and "
                "nothing can be selected safely. Dummy failure: "
                f"{'; '.join(sorted(reasons)) or 'no dummy model was built'}."
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
        for rule, field in (
            ("min_shadow_share", "above_shadow_share"),
            ("max_shadow_q", "shadow_q_value"),
            ("min_shadow_families", "shadow_families"),
        ):
            # Models that drop or cannot be verified for the shadows add no fields.
            if getattr(self, rule) is not None and rows and not any(field in r for r in rows):
                raise ValueError(
                    f"[!] {rule} needs shadow evidence, but no model kept the shadow "
                    "columns in any split (a model that drops them, or whose use of them "
                    "cannot be verified, gives no shadow evidence)."
                )
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
        """
        Keep the columns of the selected groups, in their original order.

        Parameters
        ----------
        X : array-like or DataFrame
            Data with the same columns as ``fit()``. DataFrame column labels
            are compared as strings.

        Returns
        -------
        array-like
            ``X`` with only the selected columns. If no group was selected,
            zero columns (``fit()`` warned).
        """
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
