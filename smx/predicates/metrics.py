"""
Predicate metric strategy classes.

Each class implements the ``compute(bags_dict) → dict[str, DataFrame]``
interface so they can be swapped transparently in the SMX pipeline.

Available metrics
-----------------
* :class:`PerturbationMetric` — perturbation-based importance: replace the
  spectral zone of each predicate with a constant/statistic value and measure
  the impact on model predictions. Supports:

  - **Classification** (binary and multi-class K ≥ 2) — five metrics including
    ``probability_shift`` (Total Variation Distance, the multi-class default),
    ``prediction_change_rate``, ``accuracy_drop``, ``f1_drop`` and
    ``decision_function_shift``.
  - **Regression** — three metrics: ``mean_abs_diff`` (the regression
    default), ``mean_diff`` and ``mean_relative_dev``.
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
        Importance metric. Choice depends on the estimator type and the
        desired sensitivity:

        **Classification estimators** (with ``predict_proba``):

        - ``'probability_shift'`` — Mean Total Variation Distance between
          pre- and post-perturbation class probability vectors. Works
          natively with any number of classes K ≥ 2. Requires ``predict_proba()``.
        - ``'prediction_change_rate'`` — Fraction of samples whose predicted
          class label changes after perturbation.
        - ``'accuracy_drop'`` — Drop in accuracy when perturbed predictions
          are compared to original predictions.
        - ``'f1_drop'`` — Weighted F1-score decrease after perturbation.
        - ``'decision_function_shift'`` — Mean absolute change in decision
          function values. Requires ``decision_function()``.

        **Regression estimators** (with ``predict`` returning continuous values):

        - ``'mean_abs_diff'`` — Mean absolute difference between original
          and perturbed predictions.
        - ``'mean_diff'`` — Mean signed difference (bias direction). Positive
          values indicate perturbation increases predictions, negative decreases.
        - ``'mean_relative_dev'`` — Mean relative deviation, normalized by
          original prediction magnitude. Treats zero predictions as NaN.
    normalize_by_zone_size : bool, default False
        Divide raw importance by the number of zone features (raised to
        *zone_size_exponent*) to compensate for wide-zone bias.
    zone_size_exponent : float, default 1.0
        Exponent applied to zone size for normalisation.
    verbose : bool, default False
        Print per-predicate progress.
    save_detailed_results : bool, default True
        Attach a ``'__detailed_perturbation_results__'`` key to the result.
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

    @property
    def metric_column(self) -> str:
        return "Perturbation"

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
                    zone_cols, zone_start, zone_end = _get_zone_columns(
                        pred_rule,
                        self.predicates_df,
                        self.spectral_cuts,
                        self.Xcalclass_prep.columns,
                    )
                except (KeyError, ValueError) as exc:
                    fold_metrics[pred_rule] = 0.0
                    fold_detailed[pred_rule] = {"importance": 0.0, "skip_reason": str(exc)}
                    continue

                if not zone_cols:
                    fold_metrics[pred_rule] = 0.0
                    fold_detailed[pred_rule] = {"importance": 0.0, "skip_reason": "empty zone"}
                    continue

                X_eval = self.Xcalclass_prep.iloc[sample_indices].copy()

                # ── Perturb zone ──────────────────────────────────────────
                X_perturbed = X_eval.copy()
                if self.perturbation_mode == "constant":
                    X_perturbed[zone_cols] = self.perturbation_value
                else:
                    src = (
                        self.Xcalclass_prep[zone_cols]
                        if self.stats_source == "full"
                        else X_eval[zone_cols]
                    )
                    col_stats = getattr(src, self.perturbation_mode)(axis=0)
                    for col in zone_cols:
                        X_perturbed[col] = col_stats[col]

                # ── Compute importance ────────────────────────────────────
                try:
                    importance, importance_for_ranking = self._compute_importance(
                        X_eval, X_perturbed, accuracy_score, f1_score
                    )
                except (TypeError, ValueError) as exc:
                    try:
                        y_sample = np.array(self.estimator.predict(X_eval.iloc[:1])).flatten()
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

    def _compute_importance(
        self,
        X_eval: pd.DataFrame,
        X_perturbed: pd.DataFrame,
        accuracy_score,
        f1_score,
    ) -> Tuple[float, float]:
        """Return (raw_importance, importance_for_ranking)."""
        if self.aim == "regression":
            y_orig = np.array(self.estimator.predict(X_eval)).flatten()
            y_pert = np.array(self.estimator.predict(X_perturbed)).flatten()

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
            y_orig = np.array(self.estimator.predict(X_eval)).flatten()
            y_pert = np.array(self.estimator.predict(X_perturbed)).flatten()
            imp = float(np.mean(y_orig != y_pert))
            return imp, imp

        elif self.metric == "accuracy_drop":
            y_orig = np.array(self.estimator.predict(X_eval)).flatten()
            y_pert = np.array(self.estimator.predict(X_perturbed)).flatten()
            imp = float(1.0 - accuracy_score(y_orig, y_pert))
            return imp, imp

        elif self.metric == "f1_drop":
            y_orig = np.array(self.estimator.predict(X_eval)).flatten()
            y_pert = np.array(self.estimator.predict(X_perturbed)).flatten()
            imp = float(1.0 - f1_score(y_orig, y_pert, average="weighted"))
            return imp, imp

        elif self.metric == "probability_shift":
            prob_orig = self.estimator.predict_proba(X_eval)
            prob_pert = self.estimator.predict_proba(X_perturbed)
            shift = np.mean(np.sum(np.abs(prob_orig - prob_pert), axis=1) / 2.0)
            return float(shift), float(shift)

        else:  # decision_function_shift
            df_orig = np.array(self.estimator.decision_function(X_eval))
            df_pert = np.array(self.estimator.decision_function(X_perturbed))
            if df_orig.ndim == 1:
                df_orig = df_orig.flatten()
                df_pert = df_pert.flatten()
            imp = float(np.mean(np.abs(df_orig - df_pert)))
            return imp, imp
