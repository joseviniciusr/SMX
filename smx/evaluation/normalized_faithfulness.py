"""Normalized AOPC (NAOPC) evaluation for SMX explanations.

This module adds a target-class, progressive masking evaluation without
changing the legacy :func:`progressive_masking_faithfulness` protocol.  The
observed AOPC is normalized per sample by the attainable lower and upper
bounds for the same estimator, input, masking strategy, and set of zones.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from smx.evaluation.faithfulness import (
    FaithfulnessMetric,
    MaskingStrategy,
    _compute_reference_fill_values,
    _infer_metric,
    _prepare_zone_ranking,
)
from smx.zones.extraction import extract_spectral_zones


NormalizationMode = str


def _as_numeric_vector(values: Any, *, name: str) -> np.ndarray:
    """Convert estimator output to a one-dimensional numeric vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return array


def _class_indices(classes: Any, targets: np.ndarray) -> np.ndarray:
    """Map target labels to columns in a scikit-learn class-output array."""
    classes_array = np.asarray(classes)
    indices = []
    for target in targets:
        matches = np.flatnonzero(classes_array == target)
        if len(matches) != 1:
            raise ValueError("Could not map a predicted class to estimator.classes_.")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def _predicted_labels(estimator: Any, X: pd.DataFrame, output: np.ndarray) -> np.ndarray:
    """Return predicted labels, using the model output when predict is absent."""
    if hasattr(estimator, "predict"):
        return np.asarray(estimator.predict(X))
    if output.ndim == 2:
        return np.argmax(output, axis=1)
    if output.ndim == 1:
        return (output >= 0).astype(int)
    raise ValueError("Unable to infer the originally predicted class.")


def _target_score_function(
    estimator: Any,
    X_original: pd.DataFrame,
    metric: str,
) -> tuple[np.ndarray, Callable[[pd.DataFrame], np.ndarray], bool]:
    """Create a target-class scorer and return its original scores.

    The boolean in the return value indicates whether the legacy absolute
    prediction shift should be used for ``mean_abs_diff`` fallback scoring.
    """
    if metric == "probability_shift":
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "Normalized faithfulness metric 'probability_shift' requires "
                "an estimator with predict_proba()."
            )
        original_output = np.asarray(estimator.predict_proba(X_original), dtype=float)
        if original_output.ndim != 2:
            raise ValueError("predict_proba() must return a 2D array.")
        if hasattr(estimator, "predict"):
            targets = _predicted_labels(estimator, X_original, original_output)
            if not hasattr(estimator, "classes_"):
                target_indices = np.asarray(targets, dtype=int)
            else:
                target_indices = _class_indices(estimator.classes_, targets)
        else:
            target_indices = np.argmax(original_output, axis=1)

        def score(X: pd.DataFrame) -> np.ndarray:
            output = np.asarray(estimator.predict_proba(X), dtype=float)
            if output.ndim != 2 or output.shape[1] != original_output.shape[1]:
                raise ValueError("predict_proba() returned an inconsistent shape.")
            return output[np.arange(len(output)), target_indices]

        return score(X_original), score, False

    if metric == "decision_function_shift":
        if not hasattr(estimator, "decision_function"):
            raise ValueError(
                "Normalized faithfulness metric 'decision_function_shift' "
                "requires an estimator with decision_function()."
            )
        original_output = np.asarray(estimator.decision_function(X_original), dtype=float)
        if original_output.ndim not in (1, 2):
            raise ValueError("decision_function() must return a 1D or 2D array.")

        if original_output.ndim == 1:
            if hasattr(estimator, "predict") and hasattr(estimator, "classes_"):
                targets = _predicted_labels(estimator, X_original, original_output)
                positive = np.asarray(estimator.classes_)[-1]
                direction = np.where(targets == positive, 1.0, -1.0)
            else:
                direction = np.where(original_output >= 0.0, 1.0, -1.0)

            def score(X: pd.DataFrame) -> np.ndarray:
                output = _as_numeric_vector(
                    estimator.decision_function(X), name="decision_function()"
                )
                return output * direction

            return original_output * direction, score, False

        if hasattr(estimator, "predict"):
            targets = _predicted_labels(estimator, X_original, original_output)
            if hasattr(estimator, "classes_"):
                target_indices = _class_indices(estimator.classes_, targets)
            else:
                target_indices = np.argmax(original_output, axis=1)
        else:
            target_indices = np.argmax(original_output, axis=1)

        def score(X: pd.DataFrame) -> np.ndarray:
            output = np.asarray(estimator.decision_function(X), dtype=float)
            if output.ndim != 2 or output.shape[1] != original_output.shape[1]:
                raise ValueError("decision_function() returned an inconsistent shape.")
            return output[np.arange(len(output)), target_indices]

        return original_output[np.arange(len(original_output)), target_indices], score, False

    if metric == "mean_abs_diff":
        if not hasattr(estimator, "predict"):
            raise ValueError(
                "Normalized faithfulness metric 'mean_abs_diff' requires predict()."
            )
        original_output = _as_numeric_vector(
            estimator.predict(X_original), name="predict()"
        )

        def score(X: pd.DataFrame) -> np.ndarray:
            return _as_numeric_vector(estimator.predict(X), name="predict()")

        # This preserves the absolute-shift semantics of the current SMX
        # fallback while still allowing the generic AOPC implementation below.
        return original_output, score, True

    raise ValueError(f"Unsupported normalized faithfulness metric '{metric}'.")


def _aopc_delta(
    original_scores: np.ndarray,
    masked_scores: np.ndarray,
    *,
    absolute_shift: bool,
) -> np.ndarray:
    """Return per-sample score drops for a masked subset."""
    delta = original_scores - masked_scores
    return np.abs(delta) if absolute_shift else delta


def _compute_aopc(curve_scores: np.ndarray) -> np.ndarray:
    """Compute per-sample AOPC from a ``(n_zones, n_samples)`` curve.

    The baseline ``k=0`` is intentionally excluded: this follows the AOPC
    convention used by the reference implementation and the proposal.
    """
    values = np.asarray(curve_scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("curve_scores must be a 2D array.")
    if values.shape[0] == 0:
        raise ValueError("At least one masked zone is required to compute AOPC.")
    return values.mean(axis=0)


def _find_aopc_bounds_exact(
    subset_scores: Mapping[int, np.ndarray],
    original_scores: np.ndarray,
    n_zones: int,
    *,
    absolute_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact per-sample AOPC bounds using subset dynamic programming.

    A masked input depends on the set of masked zones, not the order used to
    reach that set.  Dynamic programming therefore produces the same exact
    bounds as enumerating all ``n_zones!`` rankings while requiring only
    ``2**n_zones`` model evaluations.
    """
    original_scores = np.asarray(original_scores, dtype=float)
    n_samples = len(original_scores)
    n_subsets = 1 << n_zones
    if len(subset_scores) != n_subsets:
        raise ValueError("subset_scores must contain every subset, including mask 0.")

    best = np.full((n_subsets, n_samples), -np.inf, dtype=float)
    worst = np.full((n_subsets, n_samples), np.inf, dtype=float)
    best[0] = 0.0
    worst[0] = 0.0

    for mask in range(1, n_subsets):
        masked_scores = np.asarray(subset_scores[mask], dtype=float)
        delta = _aopc_delta(
            original_scores, masked_scores, absolute_shift=absolute_shift
        )
        predecessors = [mask ^ (1 << bit) for bit in range(n_zones) if mask & (1 << bit)]
        candidates_best = np.vstack([best[previous] for previous in predecessors]) + delta
        candidates_worst = np.vstack([worst[previous] for previous in predecessors]) + delta
        best[mask] = np.max(candidates_best, axis=0)
        worst[mask] = np.min(candidates_worst, axis=0)

    full_mask = n_subsets - 1
    return worst[full_mask] / n_zones, best[full_mask] / n_zones


def _find_aopc_bounds_beam(
    get_subset_scores: Callable[[int], np.ndarray],
    original_scores: np.ndarray,
    n_zones: int,
    *,
    beam_size: int,
    absolute_shift: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate per-sample AOPC bounds with deterministic beam search."""
    if int(beam_size) < 1:
        raise ValueError("beam_size must be a positive integer.")
    beam_size = int(beam_size)
    original_scores = np.asarray(original_scores, dtype=float)
    n_samples = len(original_scores)
    lower = np.empty(n_samples, dtype=float)
    upper = np.empty(n_samples, dtype=float)

    for sample_idx, original in enumerate(original_scores):
        upper_states = [(0.0, 0, ())]
        lower_states = [(0.0, 0, ())]
        for _depth in range(n_zones):
            def expand(states: list[tuple[float, int, tuple[int, ...]]]):
                expanded = []
                for total, mask, order in states:
                    for bit in range(n_zones):
                        if mask & (1 << bit):
                            continue
                        next_mask = mask | (1 << bit)
                        masked = get_subset_scores(next_mask)[sample_idx]
                        delta = float(
                            _aopc_delta(
                                np.asarray([original]),
                                np.asarray([masked]),
                                absolute_shift=absolute_shift,
                            )[0]
                        )
                        expanded.append((total + delta, next_mask, order + (bit,)))
                return expanded

            upper_candidates = expand(upper_states)
            lower_candidates = expand(lower_states)
            upper_states = sorted(
                upper_candidates, key=lambda item: (-item[0], item[2])
            )[:beam_size]
            lower_states = sorted(
                lower_candidates, key=lambda item: (item[0], item[2])
            )[:beam_size]

        upper[sample_idx] = upper_states[0][0] / n_zones
        lower[sample_idx] = lower_states[0][0] / n_zones

    return lower, upper


def normalized_progressive_masking_faithfulness(
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
    normalization: NormalizationMode = "auto",
    beam_size: int = 5,
    exact_max_zones: int = 9,
) -> Dict[str, Any]:
    """Evaluate SMX faithfulness with model/input-specific NAOPC bounds.

    ``normalization='exact'`` evaluates all masked subsets and finds the
    attainable bounds by dynamic programming. ``'beam'`` uses deterministic
    upper/lower beam searches and is intended for larger zone sets.
    ``'auto'`` selects exact normalization through ``exact_max_zones``.
    """
    started = time.perf_counter()
    if not isinstance(X_eval, pd.DataFrame):
        raise TypeError("X_eval must be a pandas DataFrame.")
    if normalization not in {"auto", "exact", "beam"}:
        raise ValueError("normalization must be 'auto', 'exact', or 'beam'.")
    if int(exact_max_zones) < 1:
        raise ValueError("exact_max_zones must be a positive integer.")

    zone_ranking_df = _prepare_zone_ranking(ranking_df)
    zone_dict = extract_spectral_zones(X_eval, list(spectral_cuts))
    available_zones = [z for z in zone_ranking_df["Zone"].tolist() if z in zone_dict]
    if not available_zones:
        raise ValueError("No ranked zones overlap the provided evaluation dataset.")
    if max_k is None:
        max_k = len(available_zones)
    max_k = max(1, min(int(max_k), len(available_zones)))
    available_zones = available_zones[:max_k]
    n_zones = len(available_zones)

    metric_resolved = _infer_metric(metric, estimator)
    X_reference = X_eval if X_reference is None else X_reference
    fill_values = _compute_reference_fill_values(
        X_reference, masking_strategy, constant_value
    )
    original_scores, score_function, absolute_shift = _target_score_function(
        estimator, X_eval, metric_resolved
    )
    if len(original_scores) != len(X_eval):
        raise ValueError("Estimator output length does not match X_eval.")

    zone_columns = []
    for zone_name in available_zones:
        columns = [c for c in zone_dict[zone_name].columns if c in X_eval.columns]
        zone_columns.append(list(dict.fromkeys(columns)))

    score_cache: Dict[int, np.ndarray] = {0: np.asarray(original_scores, dtype=float)}
    n_model_queries = 1

    def get_subset_scores(mask: int) -> np.ndarray:
        nonlocal n_model_queries
        if mask in score_cache:
            return score_cache[mask]
        X_masked = X_eval.copy()
        columns = [
            column
            for bit, bit_columns in enumerate(zone_columns)
            if mask & (1 << bit)
            for column in bit_columns
        ]
        # Dataset zone definitions may touch or overlap at boundaries. A
        # column must be masked once, even when it belongs to more than one
        # selected zone; this mirrors the legacy masking protocol.
        columns = list(dict.fromkeys(columns))
        if columns:
            X_masked.loc[:, columns] = fill_values.loc[columns].to_numpy()
        score_cache[mask] = np.asarray(score_function(X_masked), dtype=float)
        n_model_queries += 1
        return score_cache[mask]

    observed_curve_rows = []
    observed_deltas = []
    observed_mask = 0
    for k, zone_name in enumerate(available_zones, start=1):
        observed_mask |= 1 << (k - 1)
        delta = _aopc_delta(
            original_scores,
            get_subset_scores(observed_mask),
            absolute_shift=absolute_shift,
        )
        observed_deltas.append(delta)
        observed_curve_rows.append(
            {
                "k": k,
                "masked_zone": zone_name,
                "masked_zones": tuple(available_zones[:k]),
                "score": float(np.mean(delta)),
                "score_std": float(np.std(delta, ddof=0)),
            }
        )
    observed_aopc = _compute_aopc(np.vstack(observed_deltas))

    if normalization == "auto":
        method = "exact" if n_zones <= int(exact_max_zones) else "beam"
    else:
        method = normalization

    if method == "exact":
        for mask in range(1, 1 << n_zones):
            get_subset_scores(mask)
        lower, upper = _find_aopc_bounds_exact(
            score_cache,
            original_scores,
            n_zones,
            absolute_shift=absolute_shift,
        )
        used_beam_size = None
    else:
        lower, upper = _find_aopc_bounds_beam(
            get_subset_scores,
            original_scores,
            n_zones,
            beam_size=beam_size,
            absolute_shift=absolute_shift,
        )
        used_beam_size = int(beam_size)

    bound_width = upper - lower
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper))) * 32
    valid = bound_width > tolerance
    naopc = np.full(len(observed_aopc), np.nan, dtype=float)
    naopc[valid] = (observed_aopc[valid] - lower[valid]) / bound_width[valid]
    degenerate = ~valid
    if np.any(degenerate):
        warnings.warn(
            "NAOPC bounds are degenerate for one or more samples; their NAOPC "
            "values are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    per_sample = pd.DataFrame(
        {
            "sample_index": X_eval.index.to_numpy(),
            "aopc": observed_aopc,
            "lower_bound": lower,
            "upper_bound": upper,
            "bound_width": bound_width,
            "naopc": naopc,
            "status": np.where(degenerate, "degenerate_bounds", "ok"),
        },
        index=X_eval.index,
    )
    naopc_mean = float(np.nanmean(naopc)) if np.any(np.isfinite(naopc)) else float("nan")
    aopc_mean = float(np.mean(observed_aopc))
    result = {
        "aopc": aopc_mean,
        "naopc": naopc_mean,
        "lower_bound": float(np.mean(lower)),
        "upper_bound": float(np.mean(upper)),
        "normalization": method,
        "beam_size": used_beam_size,
        "n_zones": n_zones,
        "per_sample": per_sample,
        "curve_df": pd.DataFrame(observed_curve_rows),
        "naopc_mean": naopc_mean,
        "naopc_median": float(np.nanmedian(naopc)) if np.any(np.isfinite(naopc)) else float("nan"),
        "naopc_std": float(np.nanstd(naopc, ddof=0)) if np.any(np.isfinite(naopc)) else float("nan"),
        "aopc_mean": aopc_mean,
        "bound_width_mean": float(np.mean(bound_width)),
        "n_model_queries": n_model_queries,
        "runtime_seconds": float(time.perf_counter() - started),
        "metric": metric_resolved,
        "masking_strategy": masking_strategy,
        "ranking_df": zone_ranking_df.iloc[:n_zones].copy(),
        "status": "degenerate_bounds" if np.any(degenerate) else "ok",
    }
    return result


__all__ = [
    "normalized_progressive_masking_faithfulness",
    "_compute_aopc",
    "_find_aopc_bounds_exact",
    "_find_aopc_bounds_beam",
]
