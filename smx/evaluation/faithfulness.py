"""
Faithfulness evaluation utilities for SMX explanations.

This module implements the progressive top-k masking protocol described in the
SMX paper: spectral zones are masked following the explainer ranking and the
resulting change in model output is summarized by the area under the masking
curve (AUC).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from smx.zones.extraction import extract_spectral_zones


MaskingStrategy = Literal["zero", "constant", "mean", "median", "min", "max"]
FaithfulnessMetric = Literal["auto", "probability_shift", "mean_abs_diff"]


def _prepare_zone_ranking(
    ranking_df: pd.DataFrame,
    *,
    sort_column: str = "Local_Reaching_Centrality",
) -> pd.DataFrame:
    """Return a zone-deduplicated ranking table sorted by descending importance."""
    if ranking_df is None or ranking_df.empty:
        raise ValueError("ranking_df must be a non-empty DataFrame.")
    if "Zone" not in ranking_df.columns:
        raise ValueError("ranking_df must contain a 'Zone' column.")
    if sort_column not in ranking_df.columns:
        raise ValueError(f"ranking_df must contain '{sort_column}'.")

    zone_ranking_df = (
        ranking_df[ranking_df["Zone"].notna()]
        .copy()
        .sort_values(sort_column, ascending=False)
        .drop_duplicates(subset="Zone", keep="first")
        .reset_index(drop=True)
    )

    if zone_ranking_df.empty:
        raise ValueError("ranking_df does not contain any valid spectral zones.")

    return zone_ranking_df


def _infer_metric(metric: FaithfulnessMetric, estimator: Any) -> str:
    """Infer the masking score metric from the estimator interface."""
    if metric != "auto":
        return metric
    if hasattr(estimator, "predict_proba"):
        return "probability_shift"
    return "mean_abs_diff"


def _compute_reference_fill_values(
    X_ref: pd.DataFrame,
    strategy: MaskingStrategy,
    constant_value: float,
) -> pd.Series:
    """Return replacement values for every column under the chosen masking strategy."""
    if strategy == "zero":
        return pd.Series(0.0, index=X_ref.columns, dtype=float)
    if strategy == "constant":
        return pd.Series(float(constant_value), index=X_ref.columns, dtype=float)
    if strategy == "mean":
        return X_ref.mean(axis=0)
    if strategy == "median":
        return X_ref.median(axis=0)
    if strategy == "min":
        return X_ref.min(axis=0)
    if strategy == "max":
        return X_ref.max(axis=0)
    raise ValueError(f"Unsupported masking strategy '{strategy}'.")


def _integrate_auc(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Compute trapezoidal AUC with NumPy-version compatibility."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def _score_prediction_shift(
    estimator: Any,
    X_original: pd.DataFrame,
    X_masked: pd.DataFrame,
    metric: str,
) -> float:
    """Return the prediction shift induced by masking."""
    if metric == "probability_shift":
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "Faithfulness metric 'probability_shift' requires an estimator "
                "with predict_proba()."
            )
        y_orig = np.asarray(estimator.predict_proba(X_original), dtype=float)
        y_masked = np.asarray(estimator.predict_proba(X_masked), dtype=float)
        if y_orig.ndim != 2:
            raise ValueError("predict_proba() must return a 2D array.")
        return float(np.mean(np.abs(y_orig - y_masked)))

    if metric == "mean_abs_diff":
        y_orig = np.asarray(estimator.predict(X_original), dtype=float).reshape(-1)
        y_masked = np.asarray(estimator.predict(X_masked), dtype=float).reshape(-1)
        return float(np.mean(np.abs(y_orig - y_masked)))

    raise ValueError(f"Unsupported faithfulness metric '{metric}'.")


def progressive_masking_faithfulness(
    estimator: Any,
    X_eval: pd.DataFrame,
    spectral_cuts: Sequence,
    ranking_df: pd.DataFrame,
    *,
    X_reference: Optional[pd.DataFrame] = None,
    metric: FaithfulnessMetric = "auto",
    masking_strategy: MaskingStrategy = "zero",
    constant_value: float = 0.0,
    max_k: Optional[int] = None,
    n_random_rankings: int = 100,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """Evaluate faithfulness by progressive top-k masking over ranked zones.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Fitted model used to score masked inputs.
    X_eval : pd.DataFrame
        Evaluation spectra to be progressively masked.
    spectral_cuts : sequence
        Spectral zone definitions accepted by :func:`extract_spectral_zones`.
    ranking_df : pd.DataFrame
        Ranking table containing at least ``Zone`` and
        ``Local_Reaching_Centrality``.
    X_reference : pd.DataFrame, optional
        Reference dataset used to compute masking replacement values for
        non-zero strategies. When ``None``, *X_eval* is used.
    metric : {'auto', 'probability_shift', 'mean_abs_diff'}, default 'auto'
        Scoring function applied to original vs masked predictions.
    masking_strategy : {'zero', 'constant', 'mean', 'median', 'min', 'max'}, default 'zero'
        How the masked zone values are replaced.
    constant_value : float, default 0.0
        Constant used when ``masking_strategy='constant'``.
    max_k : int, optional
        Maximum number of ranked zones to mask. When ``None``, all ranked
        zones present in *X_eval* are used.
    n_random_rankings : int, default 100
        Number of random rankings used to contextualize the observed AUC.
    random_state : int, optional
        Seed for the random baseline.

    Returns
    -------
    dict
        Dictionary containing the masking curve, AUC summary, and a random
        baseline used to derive a categorical faithfulness level.
    """
    if not isinstance(X_eval, pd.DataFrame):
        raise TypeError("X_eval must be a pandas DataFrame.")

    zone_ranking_df = _prepare_zone_ranking(ranking_df)
    zone_dict = extract_spectral_zones(X_eval, list(spectral_cuts))
    available_zones = [z for z in zone_ranking_df["Zone"].tolist() if z in zone_dict]

    if not available_zones:
        raise ValueError("No ranked zones overlap the provided evaluation dataset.")

    if max_k is None:
        max_k = len(available_zones)
    max_k = max(1, min(int(max_k), len(available_zones)))
    available_zones = available_zones[:max_k]

    metric_resolved = _infer_metric(metric, estimator)
    X_reference = X_eval if X_reference is None else X_reference
    fill_values = _compute_reference_fill_values(
        X_reference, masking_strategy, constant_value
    )

    def _mask_curve(zone_order: List[str]) -> pd.DataFrame:
        masked_columns: List[str] = []
        curve_rows: List[Dict[str, Any]] = []
        for k, zone_name in enumerate(zone_order, start=1):
            zone_cols = [c for c in zone_dict[zone_name].columns if c in X_eval.columns]
            new_cols = [c for c in zone_cols if c not in masked_columns]
            masked_columns.extend(new_cols)

            X_masked = X_eval.copy()
            if masked_columns:
                X_masked.loc[:, masked_columns] = fill_values.loc[masked_columns].values

            score = _score_prediction_shift(
                estimator=estimator,
                X_original=X_eval,
                X_masked=X_masked,
                metric=metric_resolved,
            )
            curve_rows.append({
                "k": k,
                "masked_zone": zone_name,
                "masked_zones": tuple(zone_order[:k]),
                "score": score,
            })
        return pd.DataFrame(curve_rows)

    curve_df = _mask_curve(available_zones)
    auc = _integrate_auc(
        curve_df["score"].to_numpy(dtype=float),
        curve_df["k"].to_numpy(dtype=float),
    )
    auc_normalized = float(auc / max_k) if max_k > 0 else 0.0

    rng = np.random.default_rng(random_state)
    null_aucs: List[float] = []
    for _ in range(max(0, int(n_random_rankings))):
        shuffled = list(rng.permutation(available_zones))
        null_curve = _mask_curve(shuffled)
        null_auc = _integrate_auc(
            null_curve["score"].to_numpy(dtype=float),
            null_curve["k"].to_numpy(dtype=float),
        )
        null_aucs.append(null_auc)

    if null_aucs:
        null_auc_array = np.asarray(null_aucs, dtype=float)
        percentile = float(100.0 * np.mean(null_auc_array <= auc))
    else:
        null_auc_array = np.asarray([], dtype=float)
        percentile = 100.0

    if percentile < 60.0:
        level = "Low"
    elif percentile < 80.0:
        level = "Moderate"
    elif percentile < 95.0:
        level = "High"
    else:
        level = "Very High"

    return {
        "curve_df": curve_df,
        "auc": auc,
        "auc_normalized": auc_normalized,
        "level": level,
        "metric": metric_resolved,
        "masking_strategy": masking_strategy,
        "n_masked_zones": max_k,
        "ranking_df": zone_ranking_df.iloc[:max_k].copy(),
        "null_auc_distribution": null_auc_array,
        "null_auc_mean": float(null_auc_array.mean()) if len(null_auc_array) else None,
        "null_auc_std": float(null_auc_array.std(ddof=0)) if len(null_auc_array) else None,
        "null_percentile": percentile,
    }
