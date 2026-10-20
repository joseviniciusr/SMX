"""
Predicate metric strategy classes.

Each class implements the ``compute(bags_dict) → dict[str, DataFrame]``
interface so they can be swapped transparently in the SMX pipeline.

Available metrics
-----------------
* :class:`CovarianceMetric` — covariance (or mutual information) between
  zone scores and model predictions within each predicate bag.
* :class:`PerturbationMetric` — perturbation-based importance: replace the
  spectral zone of each predicate with a constant/statistic value and measure
  the impact on model predictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BasePredicateMetric(ABC):
    """Strategy interface for predicate importance metrics.

    Subclasses implement :meth:`compute`, which accepts a bags dictionary
    (as returned by :class:`smx.predicates.bagging.PredicateBagger`) and
    returns a dictionary mapping bag name → DataFrame with columns
    ``['Predicate', <MetricName>]``.
    """

    @abstractmethod
    def compute(self, bags_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Compute metric for every predicate in every bag.

        Parameters
        ----------
        bags_dict : dict
            ``{'Bag_1': {rule: DataFrame(['Zone_Sum', 'Predicted_Y', ...]), ...}, ...}``

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{'Bag_1': DataFrame(['Predicate', MetricName]), ...}``
            Each inner DataFrame is sorted descending by the metric column.
        """


# ---------------------------------------------------------------------------
# Covariance / Mutual Information metric
# ---------------------------------------------------------------------------


class CovarianceMetric(BasePredicateMetric):
    """Association metric between zone scores and model predictions.

    Supports two association measures:

    * ``'covariance'`` — absolute covariance between zone aggregation values
      and continuous model predictions (linear dependency).
    * ``'mutual_info'`` — mutual information (captures non-linear dependencies,
      requires ``scikit-learn``).

    Parameters
    ----------
    metric : {'covariance', 'mutual_info'}, default 'covariance'
        Association measure to compute.
    threshold : float, default 0.01
        Predicates with metric value ≤ threshold are excluded from the result.
    n_neighbors : int, default 10
        Number of nearest neighbours for mutual information estimation.
        Ignored when ``metric='covariance'``.
    """

    _METRIC_COL_NAMES = {
        "covariance": "Covariance",
        "mutual_info": "Mutual_Info",
    }

    def __init__(
        self,
        metric: Literal["covariance", "mutual_info"] = "covariance",
        threshold: float = 0.01,
        n_neighbors: int = 10,
    ) -> None:
        if metric not in self._METRIC_COL_NAMES:
            raise ValueError(
                f"metric must be one of {list(self._METRIC_COL_NAMES)}. Got '{metric}'."
            )
        self.metric = metric
        self.threshold = threshold
        self.n_neighbors = n_neighbors

    @property
    def metric_column(self) -> str:
        return self._METRIC_COL_NAMES[self.metric]

    def compute(self, bags_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Compute the association metric for each predicate in each bag.

        Parameters
        ----------
        bags_dict : dict
            Bags as returned by :class:`smx.predicates.bagging.PredicateBagger`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys = bag names.  Each DataFrame has columns
            ``['Predicate', 'Covariance']`` (or ``'Mutual_Info'``),
            sorted descending by the metric, filtered by *threshold*.
        """
        if self.metric == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression

        results: Dict[str, pd.DataFrame] = {}

        for bag_name, predicates_dict in bags_dict.items():
            if not predicates_dict:
                continue

            metrics: Dict[str, float] = {}
            for rule, df_info in predicates_dict.items():
                X_zone = df_info["Zone_Sum"].values.reshape(-1, 1)
                y_pred = df_info["Predicted_Y"].values

                if len(X_zone) < 2:
                    metrics[rule] = 0.0
                    continue

                if self.metric == "covariance":
                    cov_mat = np.cov(X_zone.flatten(), y_pred)
                    metrics[rule] = float(np.abs(cov_mat[0, 1]))
                else:  # mutual_info
                    mi = mutual_info_regression(
                        X_zone,
                        y_pred,
                        discrete_features=False,
                        n_neighbors=self.n_neighbors,
                        random_state=42,
                    )
                    metrics[rule] = float(mi[0])

            metrics_df = (
                pd.DataFrame.from_dict(metrics, orient="index", columns=[self.metric_column])
                .rename_axis(None)
                .reset_index()
                .rename(columns={"index": "Predicate"})
            )
            metrics_df = metrics_df.sort_values(self.metric_column, ascending=False).reset_index(drop=True)
            metrics_df = metrics_df[metrics_df[self.metric_column] > self.threshold].reset_index(drop=True)
            results[bag_name] = metrics_df

        return results


# ---------------------------------------------------------------------------
# Perturbation metric
# ---------------------------------------------------------------------------

def _get_zone_columns(
    predicate_rule: str,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple],
    dataset_columns: pd.Index,
) -> Tuple[List[str], Optional[float], Optional[float]]:
    """Return (zone_cols, zone_start, zone_end) for a predicate rule.

    Handles 2-element ``(start, end)``, 3-element ``(name, start, end)``, and
    4-element ``(name, start, end, group)`` cuts.  When the zone name matches a
    *group*, all cuts belonging to that group are collected and their column
    ranges unioned.
    """
    mask = predicates_df["rule"] == predicate_rule
    if not mask.any():
        raise KeyError(f"Predicate '{predicate_rule}' not found in predicates_df.")
    zone_name = predicates_df.loc[mask, "zone"].values[0]

    col_numeric = pd.to_numeric(dataset_columns.astype(str), errors="coerce")

    # ── Collect column ranges that belong to this zone ───────────────────
    # A zone_name may come from:
    #  (a) a direct 2- or 3-element cut whose name matches, OR
    #  (b) a 4-element cut whose *group* field matches (grouped zone)
    col_mask = np.zeros(len(dataset_columns), dtype=bool)
    found = False

    for cut in spectral_cuts:
        if isinstance(cut, dict):
            cut_name = cut.get("name", f"{cut.get('start')}-{cut.get('end')}")
            cut_start = cut.get("start")
            cut_end = cut.get("end")
            cut_group = cut.get("group", None)
        elif isinstance(cut, (list, tuple)):
            if len(cut) == 2:
                cut_start, cut_end = cut
                cut_name = f"{cut_start}-{cut_end}"
                cut_group = None
            elif len(cut) == 3:
                cut_name, cut_start, cut_end = cut
                cut_group = None
            elif len(cut) == 4:
                cut_name, cut_start, cut_end, cut_group = cut
            else:
                continue
        else:
            continue

        # Direct name match (ungrouped cut) or group match (grouped cut)
        if cut_name == zone_name or cut_group == zone_name:
            try:
                s, e = float(cut_start), float(cut_end)
            except Exception:
                continue
            if s > e:
                s, e = e, s
            range_mask = (~np.isnan(col_numeric)) & (col_numeric >= s) & (col_numeric <= e)
            col_mask |= range_mask
            found = True

    if not found:
        return [], None, None

    zone_cols = list(dataset_columns[col_mask])

    # For zone_start/zone_end return the overall span (used for logging only)
    matched_vals = col_numeric[col_mask]
    zone_start = float(matched_vals.min()) if len(matched_vals) else None
    zone_end = float(matched_vals.max()) if len(matched_vals) else None

    return zone_cols, zone_start, zone_end


class PerturbationMetric(BasePredicateMetric):
    """Spectral-perturbation importance metric.

    For each predicate, the spectral zone is temporarily replaced by a
    fixed value (or a per-column statistic) and the change in model
    prediction is measured.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Trained model with a ``predict()`` method.
    Xcalclass_prep : pd.DataFrame
        Pre-processed calibration dataset (samples × features).
    predicates_df : pd.DataFrame
        Predicate catalogue with columns ``'rule'`` and ``'zone'``.
    spectral_cuts : list of (name, start, end) tuples
        Defines every spectral zone boundary.
    perturbation_value : float, default 0
        Constant replacement value when ``perturbation_mode='constant'``.
    perturbation_mode : {'constant', 'mean', 'median', 'min', 'max'}, default 'constant'
        How to replace the zone.
    stats_source : {'full', 'predicate'}, default 'full'
        Data source for computing per-column statistics.
    metric : str, default 'mean_abs_diff'
        Importance metric. See :class:`smx.predicates.metrics.PerturbationMetric`
        docstring for available options per *aim*.
    normalize_by_zone_size : bool, default False
        Divide raw importance by the number of zone features (raised to
        *zone_size_exponent*) to compensate for wide-zone bias.
    zone_size_exponent : float, default 1.0
        Exponent applied to zone size for normalisation.
    verbose : bool, default False
        Print per-predicate progress.
    save_detailed_results : bool, default True
        Attach a ``'__detailed_perturbation_results__'`` key to the result.
    cache_predictions : bool, default True
        Reuse model outputs across predicates, bags and :meth:`compute` calls.
        When the perturbed zone does not depend on the predicate's samples
        (``perturbation_mode='constant'`` or ``stats_source='full'``), the
        perturbed copy of a calibration sample is the same for every
        predicate of its zone. The model is then evaluated once on the full
        calibration set and once per perturbed zone (Z + 1 calls), and each
        predicate indexes its samples' rows. All metrics are per-sample
        averages, so the result matches the uncached computation; models
        evaluated through BLAS matrix products (e.g. PLS, MLP) can differ in
        the last bits, because a full-batch product sums in a different
        order than a subset. Ignored for ``stats_source='predicate'`` with a
        non-constant mode.
    """

    _REGRESSION_METRICS = {"mean_abs_diff", "mean_diff", "mean_relative_dev"}
    _CLASSIFICATION_METRICS = {
        "prediction_change_rate",
        "accuracy_drop",
        "f1_drop",
        "probability_shift",
        "decision_function_shift",
    }

    def __init__(
        self,
        estimator: Any,
        Xcalclass_prep: pd.DataFrame,
        predicates_df: pd.DataFrame,
        spectral_cuts: List[Tuple[str, float, float]],
        perturbation_value: float = 0,
        perturbation_mode: Literal["constant", "mean", "median", "min", "max"] = "constant",
        stats_source: Literal["full", "predicate"] = "full",
        metric: str = "mean_abs_diff",
        normalize_by_zone_size: bool = False,
        zone_size_exponent: float = 1.0,
        verbose: bool = False,
        save_detailed_results: bool = True,
        cache_predictions: bool = True,
    ) -> None:
        aim = "classification" if metric in self._CLASSIFICATION_METRICS else "regression" if metric in self._REGRESSION_METRICS else None

        if aim is None:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {self._REGRESSION_METRICS} or {self._CLASSIFICATION_METRICS}.")
        if metric == "probability_shift" and not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "'probability_shift' requires an estimator with predict_proba(). "
                "For SVC, use SVC(probability=True)."
            )
        if metric == "decision_function_shift" and not hasattr(estimator, "decision_function"):
            raise ValueError(
                "'decision_function_shift' requires an estimator with decision_function()."
            )

        self.estimator = estimator
        self.Xcalclass_prep = Xcalclass_prep
        self.predicates_df = predicates_df
        self.spectral_cuts = spectral_cuts
        self.aim = aim
        self.perturbation_value = perturbation_value
        self.perturbation_mode = perturbation_mode
        self.stats_source = stats_source
        self.metric = metric
        self.normalize_by_zone_size = normalize_by_zone_size
        self.zone_size_exponent = zone_size_exponent
        self.verbose = verbose
        self.save_detailed_results = save_detailed_results
        self.cache_predictions = cache_predictions

        # Model outputs on the full calibration set: key None = unperturbed,
        # key <zone name> = that zone replaced. Filled lazily by _cached_outputs.
        self._output_cache: Dict[Optional[str], np.ndarray] = {}
        self._zone_cols_cache: Dict[str, Tuple[List[str], Optional[float], Optional[float]]] = {}
        self._rule_zone: Dict[str, str] = dict(zip(predicates_df["rule"], predicates_df["zone"]))

    @property
    def metric_column(self) -> str:
        return "Perturbation"

    @property
    def uses_cache(self) -> bool:
        """Whether model outputs are shared across predicates and bags."""
        return self.cache_predictions and (
            self.perturbation_mode == "constant" or self.stats_source == "full"
        )

    def warm_cache(self) -> None:
        """Evaluate the model on the calibration set and on every perturbed zone.

        Optional: :meth:`compute` fills the cache on demand. Calling this first
        lets copies of the object (e.g. in parallel workers) share the outputs.
        """
        if not self.uses_cache:
            return
        for rule in self.predicates_df["rule"].unique():
            zone_cols, _, _ = self._zone_columns(rule)
            if zone_cols:
                self._cached_outputs(self._zone_of(rule), zone_cols)

    def compute(self, bags_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Compute perturbation importance for each predicate in each bag.

        Parameters
        ----------
        bags_dict : dict
            Bags as returned by :class:`smx.predicates.bagging.PredicateBagger`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys = bag names.  Each DataFrame has columns
            ``['Predicate', 'Perturbation']``, sorted descending.
            When ``save_detailed_results=True`` the key
            ``'__detailed_perturbation_results__'`` is also included.
        """
        from sklearn.metrics import accuracy_score, f1_score

        results: Dict[str, pd.DataFrame] = {}
        detailed_results: Dict[str, Dict] = {}

        for fold_name, predicates_dict in bags_dict.items():
            if not predicates_dict:
                results[fold_name] = pd.DataFrame({"Predicate": [], "Perturbation": []})
                continue

            fold_metrics: Dict[str, float] = {}
            fold_detailed: Dict[str, Dict] = {}

            for pred_rule, df_info in predicates_dict.items():
                sample_indices = df_info["Sample_Index"].values.tolist()
                n_samples = len(sample_indices)

                if n_samples == 0:
                    fold_metrics[pred_rule] = 0.0
                    fold_detailed[pred_rule] = {"importance": 0.0, "skip_reason": "n_samples=0"}
                    continue

                # ── Get zone columns ──────────────────────────────────────
                try:
                    zone_cols, zone_start, zone_end = self._zone_columns(pred_rule)
                except (KeyError, ValueError) as exc:
                    fold_metrics[pred_rule] = 0.0
                    fold_detailed[pred_rule] = {"importance": 0.0, "skip_reason": str(exc)}
                    continue

                if not zone_cols:
                    fold_metrics[pred_rule] = 0.0
                    fold_detailed[pred_rule] = {"importance": 0.0, "skip_reason": "empty zone"}
                    continue

                # ── Compute importance ────────────────────────────────────
                try:
                    if self.uses_cache:
                        rows = np.asarray(sample_indices, dtype=int)
                        out_orig = self._cached_outputs(None, None)[rows]
                        out_pert = self._cached_outputs(self._zone_of(pred_rule), zone_cols)[rows]
                    else:
                        X_eval = self.Xcalclass_prep.iloc[sample_indices]
                        X_perturbed = self._perturb(X_eval.copy(), zone_cols, X_eval)
                        out_orig = self._model_output(X_eval)
                        out_pert = self._model_output(X_perturbed)
                    importance, importance_for_ranking = self._importance_from_outputs(
                        out_orig, out_pert, accuracy_score, f1_score
                    )
                except (TypeError, ValueError) as exc:
                    try:
                        y_sample = np.array(
                            self.estimator.predict(self.Xcalclass_prep.iloc[sample_indices[:1]])
                        ).flatten()
                        pred_dtype = y_sample.dtype
                        is_numeric = np.issubdtype(pred_dtype, np.number)
                    except Exception:
                        pred_dtype = "unknown"
                        is_numeric = None

                    if self.aim == "regression" and is_numeric is False:
                        hint = (
                            f"Metric '{self.metric}' requires numeric predictions, but the "
                            f"estimator returned dtype '{pred_dtype}' (e.g. class labels). "
                            f"Switch to a classification metric such as 'prediction_change_rate'."
                        )
                    elif self.aim == "classification" and is_numeric is True:
                        hint = (
                            f"Metric '{self.metric}' is a classification metric, but the "
                            f"estimator appears to return numeric values (dtype '{pred_dtype}'). "
                            f"Switch to a regression metric such as 'mean_abs_diff'."
                        )
                    else:
                        hint = (
                            f"Metric '{self.metric}' is incompatible with this estimator "
                            f"(prediction dtype: '{pred_dtype}'). "
                            f"Original error: {exc}"
                        )
                    raise TypeError(hint) from exc

                # ── Zone-size normalisation ───────────────────────────────
                n_zone_features = len(zone_cols)
                if self.normalize_by_zone_size and n_zone_features > 0:
                    importance_for_ranking /= n_zone_features ** self.zone_size_exponent

                fold_metrics[pred_rule] = float(importance_for_ranking)
                fold_detailed[pred_rule] = {
                    "importance": float(importance),
                    "importance_normalized": float(importance_for_ranking),
                    "n_samples": n_samples,
                    "n_zone_features": n_zone_features,
                }

                if self.verbose:
                    print(f"  {pred_rule} (n={n_samples}): {importance_for_ranking:.6f}")

            metrics_df = (
                pd.DataFrame.from_dict(fold_metrics, orient="index", columns=["Perturbation"])
                .rename_axis(None)
                .reset_index()
                .rename(columns={"index": "Predicate"})
                .sort_values("Perturbation", ascending=False)
                .reset_index(drop=True)
            )
            results[fold_name] = metrics_df
            detailed_results[fold_name] = fold_detailed

        if self.save_detailed_results:
            rows = [
                {"fold": fold, "predicate": rule, **data}
                for fold, fold_data in detailed_results.items()
                for rule, data in fold_data.items()
            ]
            results["__detailed_perturbation_results__"] = pd.DataFrame(rows)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _zone_of(self, rule: str) -> str:
        return self._rule_zone[rule]

    def _zone_columns(self, rule: str) -> Tuple[List[str], Optional[float], Optional[float]]:
        if rule not in self._zone_cols_cache:
            self._zone_cols_cache[rule] = _get_zone_columns(
                rule, self.predicates_df, self.spectral_cuts, self.Xcalclass_prep.columns
            )
        return self._zone_cols_cache[rule]

    def _perturb(self, X: pd.DataFrame, zone_cols: List[str], X_eval: pd.DataFrame) -> pd.DataFrame:
        """Replace *zone_cols* of *X* in place; *X_eval* is the predicate's samples."""
        if self.perturbation_mode == "constant":
            X[zone_cols] = self.perturbation_value
        else:
            src = (
                self.Xcalclass_prep[zone_cols]
                if self.stats_source == "full"
                else X_eval[zone_cols]
            )
            col_stats = getattr(src, self.perturbation_mode)(axis=0)
            for col in zone_cols:
                X[col] = col_stats[col]
        return X

    def _cached_outputs(self, zone_name: Optional[str], zone_cols: Optional[List[str]]) -> np.ndarray:
        """Model outputs on the full calibration set, with *zone_name* perturbed (None = original)."""
        if zone_name not in self._output_cache:
            X = self.Xcalclass_prep
            if zone_name is not None:
                X = self._perturb(X.copy(), zone_cols, X)
            self._output_cache[zone_name] = self._model_output(X)
        return self._output_cache[zone_name]

    def _model_output(self, X: pd.DataFrame) -> np.ndarray:
        """The estimator output the metric compares (probabilities, scores or predictions)."""
        if self.metric == "probability_shift":
            return np.asarray(self.estimator.predict_proba(X))
        if self.metric == "decision_function_shift":
            return np.asarray(self.estimator.decision_function(X))
        return np.asarray(self.estimator.predict(X))

    def _importance_from_outputs(
        self,
        out_orig: np.ndarray,
        out_pert: np.ndarray,
        accuracy_score,
        f1_score,
    ) -> Tuple[float, float]:
        """Return (raw_importance, importance_for_ranking)."""
        if self.aim == "regression":
            y_orig = out_orig.flatten()
            y_pert = out_pert.flatten()

            if self.metric == "mean_abs_diff":
                imp = float(np.mean(np.abs(y_orig - y_pert)))
                return imp, imp
            elif self.metric == "mean_diff":
                imp = float(np.mean(y_orig - y_pert))
                return imp, float(np.abs(imp))
            else:  # mean_relative_dev
                y_safe = np.where(y_orig == 0, np.nan, y_orig)
                rel = (y_pert - y_orig) / y_safe
                imp = float(np.nanmean(rel))
                return imp, float(np.abs(imp))

        # ── Classification ────────────────────────────────────────────────
        if self.metric == "prediction_change_rate":
            imp = float(np.mean(out_orig.flatten() != out_pert.flatten()))
            return imp, imp

        elif self.metric == "accuracy_drop":
            imp = float(1.0 - accuracy_score(out_orig.flatten(), out_pert.flatten()))
            return imp, imp

        elif self.metric == "f1_drop":
            imp = float(1.0 - f1_score(out_orig.flatten(), out_pert.flatten(), average="weighted"))
            return imp, imp

        elif self.metric == "probability_shift":
            shift = np.mean(np.sum(np.abs(out_orig - out_pert), axis=1) / 2.0)
            return float(shift), float(shift)

        else:  # decision_function_shift
            imp = float(np.mean(np.abs(out_orig - out_pert)))
            return imp, imp
