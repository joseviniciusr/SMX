"""
Threshold mapping and predicate interpretation utilities.

Functions in this module translate LRC results from the preprocessed (score)
space back to the natural (unpreprocessed) spectral space, and reconstruct
multivariate threshold spectra for visualisation.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers (not part of public API)
# ---------------------------------------------------------------------------


def _extract_zone_from_predicate(predicate_rule: str) -> str:
    """Return the spectral zone name embedded in a predicate rule string."""
    if "<=" in predicate_rule:
        return predicate_rule.split("<=")[0].strip()
    if ">" in predicate_rule:
        return predicate_rule.split(">")[0].strip()
    raise ValueError(f"Unrecognised operator in predicate: '{predicate_rule}'")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def extract_predicate_info(predicate_rule: str) -> dict:
    """Extract components from a predicate rule string.

    Parameters
    ----------
    predicate_rule : str
        Rule in the format ``"zone_name <= threshold"`` or
        ``"zone_name > threshold"``.

    Returns
    -------
    dict
        ``{'zone': str, 'operator': str, 'threshold': float}``

    Examples
    --------
    >>> extract_predicate_info("Ca ka <= 25.50")
    {'zone': 'Ca ka', 'operator': '<=', 'threshold': 25.5}
    """
    if "<=" in predicate_rule:
        parts = predicate_rule.split("<=")
        operator = "<="
    elif ">" in predicate_rule:
        parts = predicate_rule.split(">")
        operator = ">"
    else:
        raise ValueError(f"Unrecognised operator in: '{predicate_rule}'")

    return {
        "zone": parts[0].strip(),
        "operator": operator,
        "threshold": float(parts[1].strip()),
    }


def map_thresholds_to_natural(
    lrc_df: pd.DataFrame,
    zone_sums_preprocessed: pd.DataFrame,
    zone_sums_natural: pd.DataFrame,
) -> pd.DataFrame:
    """Map predicate thresholds from the preprocessed space to natural space.

    For each predicate in *lrc_df*, this finds the calibration sample whose
    zone score in the *preprocessed* space is closest to the predicate's
    threshold, and retrieves that sample's value in the *natural*
    (unpreprocessed) space as the best approximation.

    Parameters
    ----------
    lrc_df : pd.DataFrame
        LRC results.  Must contain columns ``'Zone'``, ``'Threshold'``,
        ``'Operator'``, and ``'Node'``.
    zone_sums_preprocessed : pd.DataFrame
        Zone aggregation scores computed on *preprocessed* calibration data
        (same zones as *lrc_df*).
    zone_sums_natural : pd.DataFrame
        Zone aggregation scores computed on *original* (unprocessed) data.

    Returns
    -------
    pd.DataFrame
        Copy of *lrc_df* with additional columns:

        * ``'Threshold_Natural'`` — threshold value in the natural space
        * ``'Reference_Sample_Index'`` — index of the nearest calibration sample
        * ``'Approximation_Error'`` — distance (preprocessed space) to the
          nearest sample
        * ``'Node_Natural'`` — predicate rule string using the natural threshold
    """
    result_df = lrc_df.copy()

    natural_thresholds = []
    sample_indices = []
    approximation_errors = []
    node_natural_list = []

    for _, row in result_df.iterrows():
        zone_name = row["Zone"]
        threshold_val = row["Threshold"]
        operator = row["Operator"]

        if (
            zone_name is None
            or threshold_val is None
            or zone_name not in zone_sums_preprocessed.columns
        ):
            natural_thresholds.append(None)
            sample_indices.append(None)
            approximation_errors.append(None)
            node_natural_list.append(None)
            continue

        threshold = float(threshold_val)
        zone_vals = zone_sums_preprocessed[zone_name]
        distances = (zone_vals - threshold).abs()
        closest_idx = distances.idxmin()

        natural_value = zone_sums_natural.loc[closest_idx, zone_name]
        error = distances.loc[closest_idx]

        node_natural = (
            f"{zone_name} {operator} {natural_value:.6f}"
            if operator is not None and natural_value is not None
            else None
        )

        natural_thresholds.append(natural_value)
        sample_indices.append(closest_idx)
        approximation_errors.append(error)
        node_natural_list.append(node_natural)

    result_df["Threshold_Natural"] = natural_thresholds
    result_df["Reference_Sample_Index"] = sample_indices
    result_df["Approximation_Error"] = approximation_errors
    result_df["Node_Natural"] = node_natural_list
    return result_df


def reconstruct_threshold_to_spectrum(
    threshold_value: float,
    zone_name: str,
    pca_info_dict: Dict,
) -> pd.Series:
    """Reconstruct a scalar threshold to a multivariate threshold spectrum.

    Uses the PCA model fitted during zone aggregation to reconstruct a
    threshold value *in score space* back into the original spectral variable
    space:

    .. math::

        \\tau = \\bar{x} + q \\cdot \\mathbf{w}

    where :math:`\\bar{x}` is the zone mean, :math:`\\mathbf{w}` the PC1
    loadings vector, and :math:`q` the threshold score value.

    Parameters
    ----------
    threshold_value : float
        Threshold in PC1 score space.
    zone_name : str
        Name of the spectral zone.
    pca_info_dict : dict
        PCA info dictionary as stored in
        :attr:`smx.zones.aggregation.ZoneAggregator.pca_info_`.

    Returns
    -------
    pd.Series
        Threshold spectrum indexed by original column names.
    """
    info = pca_info_dict[zone_name]
    spectrum = info["mean"] + threshold_value * info["loadings"]
    return pd.Series(
        spectrum,
        index=info["columns"],
        name=f"threshold_{threshold_value:.4f}",
    )
