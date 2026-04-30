"""
Plot zone-level ranking overlays on top of a reference spectrum.

The main entry point, :func:`plot_zone_ranking_over_spectrum`, accepts either:

* a precomputed ranking DataFrame with ``zone`` / ``score`` / ``rank`` columns
* an SMX LRC table with ``Zone`` / ``Local_Reaching_Centrality`` columns

and writes an HTML Plotly figure where each spectral zone is highlighted as a
ranked band over the reference spectrum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from smx.plotting.theme import DEFAULT_THEME, SMXTheme, blend_with_white, build_blended_colorscale


def _prepare_zone_ranking_df(zone_ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported ranking-table shapes into zone / score / rank form."""
    if zone_ranking_df is None or zone_ranking_df.empty:
        raise ValueError("zone_ranking_df must be a non-empty DataFrame.")

    ranking_df = zone_ranking_df.copy()

    if {"zone", "score"}.issubset(ranking_df.columns):
        normalized = ranking_df.rename(columns={"zone": "zone", "score": "score"}).copy()
    elif {"Zone", "Local_Reaching_Centrality"}.issubset(ranking_df.columns):
        # LRC tables may contain multiple predicates per zone. Collapse them to
        # a single score per zone using the strongest centrality observed.
        normalized = (
            ranking_df.groupby("Zone", as_index=False)["Local_Reaching_Centrality"]
            .max()
            .rename(columns={"Zone": "zone", "Local_Reaching_Centrality": "score"})
        )
    else:
        raise ValueError(
            "zone_ranking_df must contain either "
            "('zone', 'score') or ('Zone', 'Local_Reaching_Centrality') columns."
        )

    normalized["zone"] = normalized["zone"].astype(str)
    normalized["score"] = pd.to_numeric(normalized["score"], errors="coerce")
    normalized = normalized.dropna(subset=["score"])
    normalized = normalized.sort_values("score", ascending=False).reset_index(drop=True)

    if "rank" in ranking_df.columns and {"zone", "score"}.issubset(ranking_df.columns):
        normalized["rank"] = pd.to_numeric(ranking_df.loc[normalized.index, "rank"], errors="coerce")
        if normalized["rank"].isna().any():
            normalized["rank"] = np.arange(1, len(normalized) + 1)
    else:
        normalized["rank"] = np.arange(1, len(normalized) + 1)

    return normalized[["zone", "score", "rank"]]


def _aggregate_spectrum_df(
    spectrum_df: pd.DataFrame,
    aggregation: str,
) -> pd.Series:
    if spectrum_df.empty:
        raise ValueError("Reference spectrum DataFrame is empty.")

    if aggregation == "mean":
        spectrum = spectrum_df.mean(axis=0)
    elif aggregation == "median":
        spectrum = spectrum_df.median(axis=0)
    else:
        raise ValueError("aggregation must be 'mean' or 'median'.")

    spectrum.index = pd.to_numeric(spectrum.index.astype(str), errors="coerce")
    spectrum = spectrum[~spectrum.index.isna()]
    return spectrum.sort_index()


def _build_reference_spectrum(
    reference_spectrum: Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]],
    spectral_cuts: Iterable,
    aggregation: str,
) -> pd.Series:
    if isinstance(reference_spectrum, pd.Series):
        spectrum = reference_spectrum.copy()
        spectrum.index = pd.to_numeric(spectrum.index.astype(str), errors="coerce")
        spectrum = spectrum[~spectrum.index.isna()]
        return spectrum.sort_index()

    if isinstance(reference_spectrum, pd.DataFrame):
        return _aggregate_spectrum_df(reference_spectrum, aggregation=aggregation)

    if isinstance(reference_spectrum, dict):
        series_parts = []
        seen_x = set()
        for cut in spectral_cuts:
            if isinstance(cut, dict):
                zone_name = str(cut["name"])
            else:
                zone_name = str(cut[0]) if len(cut) == 3 else f"{cut[0]}-{cut[1]}"

            zone_df = reference_spectrum.get(zone_name)
            if zone_df is None or zone_df.empty:
                continue
            zone_series = _aggregate_spectrum_df(zone_df, aggregation=aggregation)
            zone_series = zone_series[~zone_series.index.isin(seen_x)]
            seen_x.update(zone_series.index.tolist())
            series_parts.append(zone_series)

        if not series_parts:
            raise ValueError("Could not build a reference spectrum from the provided zone dictionary.")

        return pd.concat(series_parts).sort_index()

    raise TypeError(
        "reference_spectrum must be a pandas Series, pandas DataFrame, "
        "or dict[str, pandas.DataFrame]."
    )


_DEFAULT_CLASS_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


def plot_zone_ranking_over_spectrum(
    zone_ranking_df: pd.DataFrame,
    spectral_cuts: Iterable,
    reference_spectrum: Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]],
    output_path: Optional[Union[str, Path]],
    *,
    aggregation: str = "mean",
    title: Optional[str] = None,
    spectrum_name: str = "Reference spectrum",
    colorscale: str = "YlOrRd",
    annotation_y: float = 1.06,
    class_spectra: Optional[Dict[str, Union[pd.Series, pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 500,
    theme: Optional[SMXTheme] = None,
    return_fig: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, "go.Figure"]]:
    """Save a plot showing ranked zones overlaid on a spectrum.

    The output format is inferred from *output_path*:

    * ``.html`` — interactive Plotly figure (default, no extra dependency)
    * ``.png``, ``.svg``, ``.pdf``, ``.jpg`` — static image via ``kaleido``
      (install with ``pip install kaleido``)

    Parameters
    ----------
    zone_ranking_df : pd.DataFrame
        Either a ranking table with ``zone`` / ``score`` / ``rank`` columns
        or an SMX LRC table with ``Zone`` / ``Local_Reaching_Centrality``.
    spectral_cuts : iterable
        Zone definitions as accepted by :class:`smx.pipeline.SMX`.
    reference_spectrum : pd.Series, pd.DataFrame, or dict[str, pd.DataFrame]
        Spectrum used as the background line. If a DataFrame is provided, rows
        are aggregated with ``aggregation``. If a zone dictionary is provided,
        each zone is aggregated and stitched back together following
        ``spectral_cuts`` order.
    output_path : str or Path, optional
        Destination file. If ``None``, no file is written.
    aggregation : {'mean', 'median'}, default 'mean'
        Aggregation used when *reference_spectrum* is a DataFrame or zone dict.
    title : str, optional
        Figure title.
    spectrum_name : str, default 'Reference spectrum'
        Legend label for the background spectrum.
    colorscale : str, default 'YlOrRd'
        Plotly colorscale name used for zone bands.
    annotation_y : float, default 1.06
        Annotation y-position in paper coordinates.
    class_spectra : dict[str, Series | DataFrame | dict[str, DataFrame]], optional
        Per-class spectra to overlay. Keys are class labels; values accept the
        same forms as *reference_spectrum*. Each class is plotted as a separate
        colored line using ``aggregation`` to collapse rows.
    class_colors : dict[str, str], optional
        Hex/CSS color strings keyed by class label. Missing labels fall back to
        a built-in palette.
    width : int, default 1200
        Figure width in pixels. Used only for static image exports.
    height : int, default 500
        Figure height in pixels. Used only for static image exports.
    theme : SMXTheme, optional
        Visual theme controlling colors, fonts, line styles, and the Plotly
        template.  Defaults to :data:`smx.plotting.theme.DEFAULT_THEME`.
        Explicit style parameters (``colorscale``, ``class_colors``) take
        precedence over the theme.
    return_fig : bool, default False
        If ``True``, return ``(ranking_df, figure)`` for inline display.

    Notes
    -----
    A vertical colorbar is rendered on the right side of the figure showing the
    LRC-score-to-color mapping. Its palette is pre-blended with the plot
    background so it matches the zone band colors exactly. Tick marks are placed
    at ``score_min`` and ``score_max`` and labeled accordingly.

    Returns
    -------
    pd.DataFrame
        Normalized ranking DataFrame used in the plot.
    """
    try:
        import plotly.graph_objects as go
        from plotly.colors import sample_colorscale
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_zone_ranking_over_spectrum. "
            "Install it with: pip install plotly"
        ) from exc

    theme = theme or DEFAULT_THEME
    # Explicit params take priority; fall back to theme for unset style values
    _colorscale = colorscale if colorscale != "YlOrRd" else theme.colorscale

    ranking_df = _prepare_zone_ranking_df(zone_ranking_df)
    spectrum = _build_reference_spectrum(reference_spectrum, spectral_cuts, aggregation=aggregation)
    spectrum = spectrum.dropna()
    if spectrum.empty:
        raise ValueError("Reference spectrum is empty after preprocessing.")

    # Build per-class aggregated spectra when provided
    class_series: Dict[str, pd.Series] = {}
    if class_spectra:
        resolved_colors: Dict[str, str] = {}
        _used_palette: list = []
        for label, src in class_spectra.items():
            cs = _build_reference_spectrum(src, spectral_cuts, aggregation=aggregation).dropna()
            if not cs.empty:
                class_series[label] = cs
            resolved_colors[label] = (
                (class_colors or {}).get(label)
                or theme.resolve_class_color(label, _used_palette)
            )

    score_min = float(ranking_df["score"].min())
    score_max = float(ranking_df["score"].max())

    _VRECT_OPACITY = theme.zone_opacity

    def _score_to_color(score: float) -> str:
        if score_max == score_min:
            norm = 1.0
        else:
            norm = (float(score) - score_min) / (score_max - score_min)
        return sample_colorscale(_colorscale, [norm])[0]

    # Colorscale whose colors match the blended zone backgrounds exactly
    _blended_colorscale = build_blended_colorscale(_colorscale, _VRECT_OPACITY)

    cut_rows = []
    for cut in spectral_cuts:
        if isinstance(cut, dict):
            zone_name = str(cut["name"])
            start = float(cut["start"])
            end = float(cut["end"])
        elif len(cut) == 3:
            zone_name, start, end = cut
            zone_name = str(zone_name)
            start = float(start)
            end = float(end)
        elif len(cut) == 2:
            start, end = cut
            zone_name = f"{start}-{end}"
            start = float(start)
            end = float(end)
        else:
            raise ValueError("Each spectral cut must have 2 or 3 elements, or dict form.")
        if start > end:
            start, end = end, start
        cut_rows.append({"zone": zone_name, "start": start, "end": end})

    cut_df = pd.DataFrame(cut_rows)
    plot_df = cut_df.merge(ranking_df, on="zone", how="left").sort_values("start").reset_index(drop=True)

    # Compute y-axis bounds across all spectra that will be drawn
    all_values = [spectrum.to_numpy(dtype=float)]
    for cs in class_series.values():
        all_values.append(cs.to_numpy(dtype=float))
    ymax = float(np.nanmax(np.concatenate(all_values)))
    ymin = float(np.nanmin(np.concatenate(all_values)))
    yspan = ymax - ymin if ymax > ymin else 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spectrum.index.to_numpy(dtype=float),
            y=spectrum.to_numpy(dtype=float),
            mode="lines",
            line=dict(
                color=theme.reference_line_color,
                width=theme.reference_line_width,
                dash=theme.reference_line_dash,
            ),
            name=spectrum_name,
        )
    )

    for label, cs in class_series.items():
        fig.add_trace(
            go.Scatter(
                x=cs.index.to_numpy(dtype=float),
                y=cs.to_numpy(dtype=float),
                mode="lines",
                line=dict(color=resolved_colors[label], width=theme.class_line_width),
                name=f"Class {label}",
            )
        )

    for _, row in plot_df.iterrows():
        start = float(row["start"])
        end = float(row["end"])
        zone_name = row["zone"]
        score = row.get("score")
        rank = row.get("rank")
        color = "rgba(180,180,180,0.15)" if pd.isna(score) else _score_to_color(float(score))

        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=_VRECT_OPACITY,
            line_width=0,
            layer="below",
        )
        fig.add_vline(x=start, line=dict(
            color=theme.zone_boundary_color,
            width=theme.zone_boundary_width,
            dash=theme.zone_boundary_dash,
        ))

        midpoint = (start + end) / 2.0
        rank_line = f"#{int(rank)}" if pd.notna(rank) else ""
        score_line = f"{float(score):.3f}" if pd.notna(score) else ""
        label = "<br>".join(part for part in [rank_line, zone_name, score_line] if part)

        fig.add_annotation(
            x=midpoint,
            y=annotation_y,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            align="center",
            font=dict(size=theme.annotation_font_size, family=theme.font_family),
        )

        fig.add_trace(
            go.Scatter(
                x=[midpoint],
                y=[ymax + 0.04 * yspan],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0),
                name=zone_name,
                showlegend=False,
                hovertemplate=(
                    f"Zone: {zone_name}<br>"
                    f"Range: {start:.3f} - {end:.3f}<br>"
                    f"Rank: {int(rank) if pd.notna(rank) else '-'}<br>"
                    f"Score: {float(score):.3f}" if pd.notna(score) else
                    f"Zone: {zone_name}<br>Range: {start:.3f} - {end:.3f}<br>No ranking value"
                ),
            )
        )

    if not plot_df.empty:
        fig.add_vline(
            x=float(plot_df["end"].iloc[-1]),
            line=dict(
                color=theme.zone_boundary_color,
                width=theme.zone_boundary_width,
                dash=theme.zone_boundary_dash,
            ),
        )

    # Invisible scatter whose sole purpose is to render the score colorbar
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=_blended_colorscale,
                cmin=score_min,
                cmax=score_max,
                color=[score_min],
                size=0,
                opacity=0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="LRC score", side="right"),
                    thickness=theme.colorbar_thickness,
                    len=theme.colorbar_len,
                    x=1.02,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    tickmode="array",
                    tickvals=[score_min, score_max],
                    ticktext=[f"{score_min:.3f}<br>(min)", f"{score_max:.3f}<br>(max)"],
                    tickfont=dict(size=10),
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        **theme.plotly_layout(
            title=title or "Zone ranking over spectrum",
            xaxis_title="Energy / Wavelength",
            yaxis_title="Intensity",
            margin=dict(t=110, r=100, b=90, l=60),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="center",
                x=0.5,
            ),
        )
    )
    fig.update_yaxes(range=[ymin - 0.05 * yspan, ymax + 0.12 * yspan])

    if output_path is not None:
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        if suffix == ".html":
            fig.write_html(str(output_path))
        elif suffix in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp"}:
            try:
                fig.write_image(str(output_path), width=width, height=height)
            except ValueError as exc:
                raise ImportError(
                    "Static image export requires kaleido. "
                    "Install it with: pip install kaleido"
                ) from exc
        else:
            raise ValueError(
                f"Unsupported output format '{suffix}'. "
                "Use '.html' for interactive or '.png'/'.svg'/'.pdf' for static image."
            )

    if return_fig:
        return ranking_df, fig
    return ranking_df
