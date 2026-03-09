"""
plot_threshold_spectrum: visualise a multivariate threshold overlaid on
the original spectral zone, coloured by class.

Requires ``plotly``.  The dependency is optional — import errors produce a
clear, actionable message rather than a hard package-load failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from smx.graph.interpretation import reconstruct_threshold_to_spectrum


def plot_threshold_spectrum(
    lrc_natural_df: pd.DataFrame,
    row_index: int,
    spectral_zones_original: Dict[str, pd.DataFrame],
    pca_info_dict_original: Dict,
    y_labels: pd.Series,
    output_path: Union[str, Path],
    class_colors: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """Reconstruct a threshold to spectrum space and save an HTML plot.

    The plot overlays the reconstructed multivariate threshold (in red) on
    top of the individual sample spectra for the chosen spectral zone,
    coloured by class label.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        LRC DataFrame with natural-scale thresholds.  Must contain columns
        ``'Zone'``, ``'Threshold_Natural'``, and ``'Node_Natural'``.
    row_index : int
        Row of *lrc_natural_df* to visualise.
    spectral_zones_original : dict[str, pd.DataFrame]
        Spectral zones extracted from the *unpreprocessed* calibration data.
    pca_info_dict_original : dict
        PCA info from :class:`smx.zones.aggregation.ZoneAggregator` fitted on
        the natural (unpreprocessed) data.
    y_labels : pd.Series
        Class labels aligned with the calibration data rows.
    output_path : str or Path
        Destination path for the output ``.html`` file.
    class_colors : dict, optional
        Mapping of class label → colour string.
        Defaults to ``{'A': 'gold', 'B': 'blue'}``.

    Returns
    -------
    pd.Series
        Reconstructed threshold spectrum.

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_threshold_spectrum. "
            "Install it with: pip install plotly"
        ) from exc

    if class_colors is None:
        class_colors = {"A": "gold", "B": "blue"}

    zone_name = lrc_natural_df.iloc[row_index]["Zone"]
    threshold_score = float(lrc_natural_df.iloc[row_index]["Threshold_Natural"])

    threshold_spectrum = reconstruct_threshold_to_spectrum(
        threshold_value=threshold_score,
        zone_name=zone_name,
        pca_info_dict=pca_info_dict_original,
    )

    zone_df = spectral_zones_original[zone_name]
    x_values = pd.to_numeric(zone_df.columns, errors="coerce")

    fig = go.Figure()

    seen_classes: set = set()
    for idx, row in zone_df.iterrows():
        class_label = y_labels.iloc[idx] if idx < len(y_labels) else "Unknown"
        show_legend = class_label not in seen_classes
        seen_classes.add(class_label)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=row.values,
                mode="lines",
                line=dict(
                    color=class_colors.get(class_label, "rgba(128,128,128,0.3)"),
                    width=0.5,
                ),
                name=f"Class {class_label}",
                legendgroup=class_label,
                showlegend=show_legend,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=threshold_spectrum.values,
            mode="lines",
            line=dict(color="red", width=4, dash="dash"),
            name=f"Threshold Spectrum ({threshold_spectrum.name})",
        )
    )

    node_natural = lrc_natural_df.iloc[row_index].get("Node_Natural", "")
    fig.update_layout(
        title=f"Zone '{zone_name}' — Multivariate Threshold (Predicate: {node_natural})",
        xaxis_title="Energy / Wavelength",
        yaxis_title="Intensity",
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.write_html(str(output_path))

    return threshold_spectrum
