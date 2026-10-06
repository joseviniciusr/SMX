"""
Plot spectral zones and zone-level rankings on top of a spectrum.

* :func:`plot_spectrum_with_zones` shades spectral zones / backgrounds behind
  a spectrum (used by :func:`smx.zones.build.building_spectral_zones`).
* :func:`plot_zone_ranking_over_spectrum` colours each zone by its ranking
  score, accepting either a ``zone`` / ``score`` / ``rank`` table or an SMX
  LRC table with ``Zone`` / ``Local_Reaching_Centrality`` columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from smx.plotting._common import (
    PathLike,
    boundary_line,
    finalize_figure,
    legend_below,
    normalize_cuts,
    numeric_spectrum,
    prepare_zone_ranking_df,
    require_plotly,
    resolve_theme,
    score_colorbar,
    score_colors,
    stitch_zones,
)
from smx.plotting.theme import SMXTheme, build_blended_colorscale

if TYPE_CHECKING:
    import plotly.graph_objects as go

SpectrumLike = Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]

# Kept for backward compatibility with code importing the private helper.
_prepare_zone_ranking_df = prepare_zone_ranking_df


def _aggregate(spectrum_df: pd.DataFrame, aggregation: str) -> pd.Series:
    if aggregation == "mean":
        return spectrum_df.mean(axis=0)
    if aggregation == "median":
        return spectrum_df.median(axis=0)
    raise ValueError("aggregation must be 'mean' or 'median'.")


def _build_reference_spectrum(
    reference_spectrum: SpectrumLike,
    cuts_df: pd.DataFrame,
    aggregation: str,
) -> pd.Series:
    """Collapse a Series / DataFrame / zone dict into one numeric-indexed spectrum."""
    if isinstance(reference_spectrum, pd.Series):
        spectrum = numeric_spectrum(reference_spectrum)
    elif isinstance(reference_spectrum, pd.DataFrame):
        if reference_spectrum.empty:
            raise ValueError("Reference spectrum DataFrame is empty.")
        spectrum = numeric_spectrum(_aggregate(reference_spectrum, aggregation))
    elif isinstance(reference_spectrum, dict):
        spectrum = stitch_zones(
            reference_spectrum,
            cuts_df["zone"].tolist(),
            lambda df: _aggregate(df, aggregation),
        )
        if spectrum.empty:
            raise ValueError(
                "Could not build a reference spectrum from the zone dictionary: none of "
                "its keys match the zones in spectral_cuts."
            )
    else:
        raise TypeError(
            "reference_spectrum must be a pandas Series, pandas DataFrame, "
            "or dict[str, pandas.DataFrame]."
        )
    spectrum = spectrum.dropna()
    if spectrum.empty:
        raise ValueError("Reference spectrum is empty after preprocessing.")
    return spectrum


def plot_spectrum_with_zones(
    spectrum: Union[pd.Series, pd.DataFrame, np.ndarray],
    spectral_cuts: Iterable,
    identified_peaks: Optional[Iterable[int]] = None,
    identified_minima: Optional[Iterable[int]] = None,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    zone_color: str = "rgb(173, 216, 230)",
    background_color: str = "rgb(0, 34, 75)",
    zone_opacity: Optional[float] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 500,
    return_df: bool = False,
) -> Union["go.Figure", Tuple["go.Figure", pd.DataFrame]]:
    """Plot a spectrum with spectral zones highlighted in the background.

    Cuts whose name (or group) contains ``"background"`` are shaded with
    *background_color*; all other cuts with *zone_color*.

    Parameters
    ----------
    spectrum : pandas.Series, pandas.DataFrame, or numpy.ndarray
        Spectrum values.  A DataFrame or 2-D array is averaged over rows.
        Numeric Series/column labels are used as x positions; otherwise the
        positional index is used.
    spectral_cuts : iterable
        Zone definitions in any format accepted by
        :func:`smx.extract_spectral_zones`.
    identified_peaks : iterable of int, optional
        Positional indices of local maxima to mark on the plot.
    identified_minima : iterable of int, optional
        Positional indices of local minima to mark on the plot.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    zone_color : str, default "rgb(173, 216, 230)"
        Fill colour (any CSS colour) for spectral zones.
    background_color : str, default "rgb(0, 34, 75)"
        Fill colour (any CSS colour) for background zones.
    zone_opacity : float, optional
        Opacity of the shaded regions.  Defaults to ``theme.zone_opacity``.
    theme : SMXTheme, optional
        Visual theme controlling fonts and line styles.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, cuts_df)`` with the parsed cuts.

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    go = require_plotly()
    theme = resolve_theme(theme)
    opacity = theme.zone_opacity if zone_opacity is None else zone_opacity

    if isinstance(spectrum, pd.DataFrame):
        if spectrum.empty:
            raise ValueError("spectrum DataFrame is empty.")
        spectrum_series = spectrum.mean(axis=0)
    elif isinstance(spectrum, pd.Series):
        spectrum_series = spectrum
    else:
        array = np.asarray(spectrum, dtype=float)
        spectrum_series = pd.Series(np.nanmean(array, axis=0) if array.ndim > 1 else array)
    if spectrum_series.empty:
        raise ValueError("spectrum is empty.")

    x_numeric = pd.to_numeric(spectrum_series.index.astype(str), errors="coerce")
    if x_numeric.isna().any():
        x_values = np.arange(len(spectrum_series), dtype=float)
    else:
        x_values = x_numeric.to_numpy(dtype=float)
    y_values = spectrum_series.to_numpy(dtype=float)

    cuts_df = normalize_cuts(spectral_cuts)
    cuts_df["is_background"] = [
        "background" in name.lower() or "background" in zone.lower()
        for name, zone in zip(cuts_df["name"], cuts_df["zone"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        line=dict(color=theme.reference_line_color, width=theme.reference_line_width),
        name="Spectrum",
    ))

    for row in cuts_df.itertuples(index=False):
        fig.add_vrect(
            x0=row.start,
            x1=row.end,
            fillcolor=background_color if row.is_background else zone_color,
            opacity=opacity,
            line_width=0,
            layer="below",
        )
    for boundary in sorted(set(cuts_df["start"]) | set(cuts_df["end"])):
        fig.add_vline(x=boundary, line=boundary_line(theme))

    # Legend proxies for the shaded regions (only for the kinds present).
    for is_background, color, name in [
        (False, zone_color, "Spectral zone"),
        (True, background_color, "Background"),
    ]:
        if (cuts_df["is_background"] == is_background).any():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=14, color=color, opacity=opacity, symbol="square"),
                name=name,
            ))

    for indices, color, symbol, name in [
        (identified_peaks, "#e41a1c", "circle", "Identified peaks"),
        (identified_minima, "#4daf4a", "triangle-up", "Identified minima"),
    ]:
        if indices is None:
            continue
        idx = np.asarray(list(indices), dtype=int)
        idx = idx[(idx >= 0) & (idx < len(y_values))]
        if idx.size:
            fig.add_trace(go.Scatter(
                x=x_values[idx],
                y=y_values[idx],
                mode="markers",
                marker=dict(size=8, color=color, symbol=symbol),
                name=name,
            ))

    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    y_span = y_max - y_min if y_max > y_min else 1.0
    fig.update_yaxes(range=[y_min - 0.05 * y_span, y_max + 0.08 * y_span])

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="Spectrum with spectral zones",
        width=width,
        height=height,
        output_path=output_path,
        data=cuts_df.drop(columns="is_background"),
        return_df=return_df,
        xaxis_title="Spectral variables",
        yaxis_title="Intensity",
        legend=legend_below(),
    )


def plot_zone_ranking_over_spectrum(
    zone_ranking_df: pd.DataFrame,
    spectral_cuts: Iterable,
    reference_spectrum: SpectrumLike,
    *,
    output_path: Optional[PathLike] = None,
    aggregation: str = "mean",
    title: Optional[Union[str, bool]] = None,
    spectrum_name: str = "Reference spectrum",
    colorscale: Optional[str] = None,
    annotation_y: float = 1.06,
    class_spectra: Optional[Dict[str, SpectrumLike]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 500,
    return_df: bool = False,
) -> Union["go.Figure", Tuple["go.Figure", pd.DataFrame]]:
    """Plot ranked spectral zones as coloured bands over a spectrum.

    Each zone is shaded according to its ranking score and annotated with its
    rank, name and score.  A colorbar on the right maps colour to score.

    Parameters
    ----------
    zone_ranking_df : pd.DataFrame
        Either a ranking table with ``zone`` / ``score`` (/ ``rank``) columns
        or an SMX LRC table with ``Zone`` / ``Local_Reaching_Centrality``.
        Several rows per zone are collapsed to the strongest score.
    spectral_cuts : iterable
        Zone definitions as accepted by :class:`smx.pipeline.SMX`.  Grouped
        cuts are shaded with the score of their group.
    reference_spectrum : pd.Series, pd.DataFrame, or dict[str, pd.DataFrame]
        Spectrum drawn as the background line.  DataFrames are aggregated with
        *aggregation*; zone dictionaries are aggregated per zone and stitched
        back together following *spectral_cuts*.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    aggregation : {'mean', 'median'}, default 'mean'
        Aggregation used when a spectrum is given as a DataFrame or zone dict.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    spectrum_name : str, default 'Reference spectrum'
        Legend label for the reference spectrum.
    colorscale : str, optional
        Plotly colorscale for the zone bands.  Defaults to ``theme.colorscale``.
    annotation_y : float, default 1.06
        Zone-label annotation y-position in paper coordinates.
    class_spectra : dict[str, Series | DataFrame | dict[str, DataFrame]], optional
        Per-class spectra to overlay, keyed by class label.  Values accept the
        same forms as *reference_spectrum*.
    class_colors : dict[str, str], optional
        Per-class colours.  Overrides ``theme.class_colors``.
    theme : SMXTheme, optional
        Visual theme.  Defaults to :data:`smx.plotting.theme.DEFAULT_THEME`.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, ranking_df)`` with the normalized
        ``zone`` / ``score`` / ``rank`` table.

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)

    Notes
    -----
    The colorbar palette is pre-blended with the white plot background so it
    matches the semi-transparent zone bands exactly.
    """
    go = require_plotly()
    theme = resolve_theme(theme)
    colorscale = colorscale or theme.colorscale
    opacity = theme.zone_opacity

    ranking_df = prepare_zone_ranking_df(zone_ranking_df)
    cuts_df = normalize_cuts(spectral_cuts)
    spectrum = _build_reference_spectrum(reference_spectrum, cuts_df, aggregation)

    class_series: Dict[str, pd.Series] = {}
    for label, source in (class_spectra or {}).items():
        series = _build_reference_spectrum(source, cuts_df, aggregation)
        if not series.empty:
            class_series[label] = series
    colors = theme.class_color_map(class_series, class_colors)

    plot_df = cuts_df.merge(ranking_df, on="zone", how="left")
    score_min = float(ranking_df["score"].min())
    score_max = float(ranking_df["score"].max())

    all_values = np.concatenate([s.to_numpy(dtype=float) for s in [spectrum, *class_series.values()]])
    y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    y_span = y_max - y_min if y_max > y_min else 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spectrum.index.to_numpy(dtype=float),
        y=spectrum.to_numpy(dtype=float),
        mode="lines",
        line=dict(
            color=theme.reference_line_color,
            width=theme.reference_line_width,
            dash=theme.reference_line_dash,
        ),
        name=spectrum_name,
    ))
    for label, series in class_series.items():
        fig.add_trace(go.Scatter(
            x=series.index.to_numpy(dtype=float),
            y=series.to_numpy(dtype=float),
            mode="lines",
            line=dict(color=colors[label], width=theme.class_line_width),
            name=f"Class {label}",
        ))

    band_colors = score_colors(plot_df["score"].fillna(score_min), colorscale, score_min, score_max)
    hover_x, hover_text = [], []
    for row, color in zip(plot_df.itertuples(index=False), band_colors):
        has_score = pd.notna(row.score)
        fig.add_vrect(
            x0=row.start,
            x1=row.end,
            fillcolor=color if has_score else theme.no_data_color,
            opacity=opacity,
            line_width=0,
            layer="below",
        )
        label_parts = [f"#{int(row.rank)}", row.zone, f"{row.score:.3f}"] if has_score else [row.zone]
        fig.add_annotation(
            x=(row.start + row.end) / 2.0,
            y=annotation_y,
            xref="x",
            yref="paper",
            text="<br>".join(label_parts),
            showarrow=False,
            align="center",
            font=dict(size=theme.annotation_font_size, family=theme.font_family),
        )
        hover_x.append((row.start + row.end) / 2.0)
        details = f"Rank: {int(row.rank)}<br>Score: {row.score:.4f}" if has_score else "No ranking value"
        hover_text.append(f"Zone: {row.zone}<br>Range: {row.start:.3f} – {row.end:.3f}<br>{details}")

    for boundary in sorted(set(plot_df["start"]) | set(plot_df["end"])):
        fig.add_vline(x=boundary, line=boundary_line(theme))

    # Invisible markers carrying the per-zone hover text.
    fig.add_trace(go.Scatter(
        x=hover_x,
        y=[y_max + 0.04 * y_span] * len(hover_x),
        mode="markers",
        marker=dict(size=10, opacity=0),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    # Invisible scatter whose sole purpose is to render the score colorbar.
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=build_blended_colorscale(colorscale, opacity),
            cmin=score_min,
            cmax=score_max,
            color=[score_min],
            size=0,
            opacity=0,
            showscale=True,
            colorbar=score_colorbar(theme, score_min, score_max, x=1.02, xanchor="left"),
        ),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_yaxes(range=[y_min - 0.05 * y_span, y_max + 0.12 * y_span])

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="Zone ranking over spectrum",
        width=width,
        height=height,
        output_path=output_path,
        data=ranking_df,
        return_df=return_df,
        xaxis_title="Spectral variables",
        yaxis_title="Intensity",
        margin=dict(t=110, r=100, b=90, l=60),
        legend=legend_below(),
    )
