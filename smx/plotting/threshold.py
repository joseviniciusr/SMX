"""
plot_threshold_spectrum: visualise a multivariate threshold overlaid on
the original spectral zone, coloured by class.

Requires ``plotly``.  The dependency is optional — import errors produce a
clear, actionable message rather than a hard package-load failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from smx.graph.interpretation import reconstruct_threshold_to_spectrum
from smx.plotting._common import (
    PathLike,
    class_labels,
    finalize_figure,
    legend_below,
    pop_deprecated_kwargs,
    require_plotly,
    resolve_theme,
    unique_classes,
)
from smx.plotting.theme import SMXTheme

if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_threshold_spectrum(
    lrc_natural_df: pd.DataFrame,
    row_index: int,
    zones_natural: Optional[Dict[str, pd.DataFrame]] = None,
    pca_info_natural: Optional[Dict] = None,
    y_labels: Optional[Union[pd.Series, Sequence]] = None,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 900,
    height: Optional[int] = 450,
    return_df: bool = False,
    **deprecated: Any,
) -> Union["go.Figure", Tuple["go.Figure", pd.Series]]:
    """Plot a predicate threshold reconstructed in the original spectral space.

    The reconstructed multivariate threshold is drawn on top of the individual
    sample spectra of the predicate's zone, coloured by class label.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        LRC table with natural-scale thresholds (``SMX.lrc_natural_``).  Must
        contain ``Zone`` and ``Threshold_Natural``; ``Node_Natural`` is used
        in the default title when present.
    row_index : int
        *Positional* row of *lrc_natural_df* to visualise (``iloc``).
    zones_natural : dict[str, pd.DataFrame]
        Spectral zones of the *unpreprocessed* calibration data
        (``SMX.zones_natural_``).
    pca_info_natural : dict
        PCA info fitted on the natural zones (``SMX.pca_info_natural_``).
    y_labels : pd.Series or array-like, optional
        Class labels aligned row by row with the zone DataFrames.  When
        omitted, all spectra are drawn in a single colour.
    output_path : str or Path, optional
        Export destination; ``.html`` or a static image suffix (``.png``,
        ``.svg``, ``.pdf``, …, requires ``kaleido``).  Nothing is written when
        ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    class_colors : dict[str, str], optional
        Per-class colours.  Overrides ``theme.class_colors``.
    theme : SMXTheme, optional
        Visual theme.  Defaults to :data:`smx.plotting.theme.DEFAULT_THEME`.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, threshold_spectrum)``.

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.Series)
        The figure, plus the reconstructed threshold spectrum when
        *return_df* is ``True``.

    Notes
    -----
    ``spectral_zones_original`` and ``pca_info_dict_original`` are accepted as
    deprecated aliases of *zones_natural* and *pca_info_natural*.
    """
    resolved = pop_deprecated_kwargs(
        "plot_threshold_spectrum",
        deprecated,
        {
            "spectral_zones_original": "zones_natural",
            "pca_info_dict_original": "pca_info_natural",
        },
        {"zones_natural": zones_natural, "pca_info_natural": pca_info_natural},
    )
    zones_natural = resolved["zones_natural"]
    pca_info_natural = resolved["pca_info_natural"]
    if zones_natural is None or pca_info_natural is None:
        raise TypeError("plot_threshold_spectrum() requires zones_natural and pca_info_natural.")

    go = require_plotly()
    theme = resolve_theme(theme)

    if not -len(lrc_natural_df) <= row_index < len(lrc_natural_df):
        raise IndexError(
            f"row_index {row_index} is out of range for lrc_natural_df with {len(lrc_natural_df)} rows."
        )
    row = lrc_natural_df.iloc[row_index]
    zone_name = row["Zone"]
    if pd.isna(zone_name):
        raise ValueError(f"Row {row_index} of lrc_natural_df has no zone.")
    if zone_name not in zones_natural:
        raise KeyError(f"Zone '{zone_name}' is not present in zones_natural.")
    threshold_score = float(row["Threshold_Natural"])

    threshold_spectrum = reconstruct_threshold_to_spectrum(
        threshold_value=threshold_score,
        zone_name=zone_name,
        pca_info_dict=pca_info_natural,
    )

    zone_df = zones_natural[zone_name]
    x_values = pd.to_numeric(zone_df.columns.astype(str), errors="coerce").to_numpy(dtype=float)
    values = zone_df.to_numpy(dtype=float)

    if y_labels is None:
        labels = pd.Series(["All samples"] * len(zone_df))
        colors = {"All samples": theme.reference_line_color}
        names = {"All samples": "Samples"}
    else:
        labels = class_labels(y_labels, n_rows=len(zone_df))
        colors = theme.class_color_map(unique_classes(labels), class_colors)
        names = {cls: f"Class {cls}" for cls in colors}

    fig = go.Figure()
    # One trace per class: individual spectra are separated by NaN gaps, which
    # keeps the figure light even for thousands of samples.
    for cls, color in colors.items():
        rows = values[(labels == cls).to_numpy()]
        if rows.size == 0:
            continue
        xs = np.tile(np.append(x_values, np.nan), len(rows))
        ys = np.hstack([rows, np.full((len(rows), 1), np.nan)]).ravel()
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=0.6),
            opacity=0.6,
            name=names[cls],
            legendgroup=str(cls),
            hoverinfo="skip",
            connectgaps=False,
        ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=threshold_spectrum.to_numpy(dtype=float),
        mode="lines",
        line=dict(
            color=theme.threshold_color,
            width=theme.threshold_line_width,
            dash=theme.threshold_line_dash,
        ),
        name="Threshold spectrum",
        hovertemplate=f"Threshold score: {threshold_score:.4g}<br>x: %{{x}}<br>y: %{{y:.4g}}<extra></extra>",
    ))

    node_natural = row.get("Node_Natural")
    if isinstance(node_natural, str) and node_natural:
        default_title = f"Multivariate threshold — {node_natural}"
    else:
        default_title = f"Multivariate threshold — zone '{zone_name}'"

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title=default_title,
        width=width,
        height=height,
        output_path=output_path,
        data=threshold_spectrum,
        return_df=return_df,
        xaxis_title="Spectral variables",
        yaxis_title="Intensity",
        legend=legend_below(),
    )
