"""
Summary and diagnostic plots for SMX explanation results.

Functions
---------
plot_lrc_bar
    Horizontal bar chart of LRC scores per zone.
plot_predicate_heatmap
    Zone × predicate heatmap of LRC scores.
plot_zone_scores
    Split-violin of PC1 scores per zone by class.
plot_all_thresholds_overlay
    Full-spectrum overlay of top-predicate threshold per zone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from smx.plotting.theme import DEFAULT_THEME, SMXTheme, blend_with_white, build_blended_colorscale


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _write_figure(fig, output_path: Optional[Union[str, Path]], width: int, height: int) -> None:
    if output_path is None:
        return

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


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for SMX plotting. "
            "Install it with: pip install plotly"
        ) from exc


# ── 1. LRC Bar Chart ───────────────────────────────────────────────────────────

def plot_lrc_bar(
    zone_ranking_df: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: int = 800,
    height: int = 500,
) -> pd.DataFrame:
    """Horizontal bar chart of LRC scores per zone.

    Each bar represents a spectral zone and is colored according to the same
    LRC-score colorscale used in :func:`plot_zone_ranking_over_spectrum`,
    making the two plots directly comparable.

    Parameters
    ----------
    zone_ranking_df : pd.DataFrame
        LRC table (``Zone`` / ``Local_Reaching_Centrality`` columns) or a
        pre-normalized ``zone`` / ``score`` / ``rank`` DataFrame.
    output_path : str or Path
        Destination file.  Extension determines format (``.html`` or image).
    title : str, optional
        Figure title.
    colorscale : str, optional
        Plotly colorscale name.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.  Defaults to :data:`~smx.plotting.theme.DEFAULT_THEME`.
    width : int, default 800
        Figure width in pixels (static export only).
    height : int, default 500
        Figure height in pixels (static export only).

    Returns
    -------
    pd.DataFrame
        Normalized ``zone / score / rank`` DataFrame used in the plot.
    """
    go = _require_plotly()
    from plotly.colors import sample_colorscale
    from smx.plotting.zones import _prepare_zone_ranking_df

    theme = theme or DEFAULT_THEME
    _colorscale = colorscale or theme.colorscale
    _opacity = theme.zone_opacity

    ranking_df = _prepare_zone_ranking_df(zone_ranking_df)
    ranking_df = ranking_df.sort_values("score", ascending=True)

    score_total = float(ranking_df["score"].sum())
    ranking_df["pct"] = ranking_df["score"] / max(score_total, 1e-9) * 100

    score_min = float(ranking_df["score"].min())
    score_max = float(ranking_df["score"].max())

    def _norm(s: float) -> float:
        return (s - score_min) / max(score_max - score_min, 1e-9)

    colors = [
        blend_with_white(sample_colorscale(_colorscale, [_norm(s)])[0], _opacity)
        for s in ranking_df["score"]
    ]

    fig = go.Figure(go.Bar(
        x=ranking_df["pct"],
        y=["#" + str(int(r)) + "  " + z for r, z in zip(ranking_df["rank"], ranking_df["zone"])],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#555555", width=1)),
        text=[f"{p:.1f}%" for p in ranking_df["pct"]],
        textposition="outside",
        hovertemplate="Zone: %{y}<br>Share: %{x:.2f}%<br>LRC: %{customdata:.4f}<extra></extra>",
        customdata=ranking_df["score"].tolist(),
    ))

    x_max = float(ranking_df["pct"].max())
    fig.update_layout(
        **theme.plotly_layout(
            title=title or "LRC Score by Spectral Zone",
            xaxis=dict(title="LRC Score (% of total)", range=[0, x_max * 1.25]),
            yaxis=dict(title="Zone"),
            margin=dict(t=80, r=100, b=60, l=180),
        )
    )

    _write_figure(fig, output_path, width, height)
    return ranking_df


# ── 2. Predicate Heatmap ───────────────────────────────────────────────────────

def plot_predicate_heatmap(
    lrc_natural_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]],
    *,
    title: Optional[str] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: int = 1000,
    height: int = 550,
    return_fig: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, "go.Figure"]]:
    """Heatmap of LRC scores across zones and predicate thresholds.

    Rows are spectral zones (sorted by maximum LRC, highest at top). Columns
    are predicates within each zone, grouped by operator (``≤`` then ``>``)
    and sorted by threshold rank within each group.  Cell color encodes LRC
    score on the same colorscale as the bar chart and zone-ranking plot.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        ``smx.lrc_natural_`` — must contain ``Zone``, ``Operator``,
        ``Threshold_Natural``, and ``Local_Reaching_Centrality`` columns.
    output_path : str or Path, optional
        Destination file. If ``None``, no file is written.
    title : str, optional
        Figure title.
    colorscale : str, optional
        Plotly colorscale name.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.
    width : int, default 1000
        Figure width (static export).
    height : int, default 550
        Figure height (static export).
    return_fig : bool, default False
        If ``True``, return ``(pivot_df, figure)`` for inline display.

    Returns
    -------
    pd.DataFrame
        Pivot DataFrame (zones × predicate labels → LRC score).
    """
    go = _require_plotly()
    theme = theme or DEFAULT_THEME
    _colorscale = colorscale or theme.colorscale
    _blended = build_blended_colorscale(_colorscale, theme.zone_opacity)

    df = lrc_natural_df[lrc_natural_df["Zone"].notna()].copy()
    df = df.sort_values(["Zone", "Operator", "Threshold_Natural"])
    df["thresh_rank"] = df.groupby(["Zone", "Operator"]).cumcount() + 1
    op_symbol = {"<=": "≤", ">": ">"}
    df["predicate_label"] = df["Operator"].map(op_symbol) + " T" + df["thresh_rank"].astype(str)

    pivot = df.pivot_table(
        index="Zone",
        columns="predicate_label",
        values="Local_Reaching_Centrality",
        aggfunc="max",
    )

    zone_order = (
        df.groupby("Zone")["Local_Reaching_Centrality"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    pivot = pivot.reindex(zone_order)

    le_cols = sorted(c for c in pivot.columns if c.startswith("≤"))
    gt_cols = sorted(c for c in pivot.columns if c.startswith(">"))
    pivot = pivot[le_cols + gt_cols]

    text_vals = [
        [f"{v:.3f}" if not np.isnan(v) else "—" for v in row]
        for row in pivot.values
    ]

    score_min = float(df["Local_Reaching_Centrality"].min())
    score_max = float(df["Local_Reaching_Centrality"].max())

    # Sentinel value for NaN cells: one step below score_min so they map to a
    # dedicated "no data" color at the bottom of the extended colorscale.
    _SENTINEL = score_min - (score_max - score_min) * 0.25
    _NO_DATA_COLOR = "rgb(220,220,220)"

    z_filled = np.where(np.isnan(pivot.values), _SENTINEL, pivot.values)

    # Prepend a "no data" segment [0, no_data_frac) → _NO_DATA_COLOR, then the
    # blended data colorscale occupies [no_data_frac, 1].
    _data_range = score_max - _SENTINEL
    _no_data_frac = (score_min - _SENTINEL) / max(_data_range, 1e-9)
    _extended_cs = [[0.0, _NO_DATA_COLOR], [_no_data_frac, _NO_DATA_COLOR]] + [
        [_no_data_frac + (1 - _no_data_frac) * pos, color]
        for pos, color in _blended
    ]

    fig = go.Figure(go.Heatmap(
        z=z_filled.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=_extended_cs,
        zmin=_SENTINEL,
        zmax=score_max,
        colorbar=dict(
            title=dict(text="LRC score", side="right"),
            thickness=theme.colorbar_thickness,
            len=theme.colorbar_len,
            tickmode="array",
            tickvals=[score_min, score_max],
            ticktext=[f"{score_min:.3f} (min)", f"{score_max:.3f} (max)"],
            tickfont=dict(size=10),
        ),
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9, family=theme.font_family),
        hovertemplate="Zone: %{y}<br>Predicate: %{x}<br>LRC: %{text}<extra></extra>",
        hoverongaps=False,
        xgap=2,
        ygap=2,
    ))

    # Vertical separator between ≤ and > columns
    if le_cols and gt_cols:
        fig.add_vline(
            x=len(le_cols) - 0.5,
            line=dict(color="white", width=3),
        )
        fig.add_annotation(
            x=(len(le_cols) - 1) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            text="Operator  ≤",
            showarrow=False,
            font=dict(size=theme.annotation_font_size, family=theme.font_family),
        )
        fig.add_annotation(
            x=len(le_cols) + (len(gt_cols) - 1) / 2,
            y=1.04,
            xref="x",
            yref="paper",
            text="Operator  >",
            showarrow=False,
            font=dict(size=theme.annotation_font_size, family=theme.font_family),
        )

    fig.update_layout(
        **theme.plotly_layout(
            title=title or "Predicate LRC Heatmap",
            xaxis=dict(title="Predicate (operator · threshold rank)", tickangle=-30),
            yaxis=dict(title="Zone"),
            margin=dict(t=100, r=120, b=100, l=160),
            plot_bgcolor="#d8d8d8",
        )
    )

    _write_figure(fig, output_path, width, height)
    if return_fig:
        return pivot, fig
    return pivot


# ── 3. Zone PC1 Score Violin ───────────────────────────────────────────────────

def plot_zone_scores(
    zones: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    y_labels: pd.Series,
    output_path: Optional[Union[str, Path]],
    spectral_cuts: Optional[Iterable] = None,
    *,
    title: Optional[str] = None,
    class_colors: Optional[Dict[str, str]] = None,
    theme: Optional[SMXTheme] = None,
    width: int = 1200,
    height: int = 580,
    return_fig: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, "go.Figure"]]:
    """Split-violin plot of PC1 scores per spectral zone, split by class.

    When exactly two classes are present the violins are mirrored (split).
    For three or more classes, separate overlapping violins are drawn.

    Parameters
    ----------
    zones : pd.DataFrame or dict[str, pd.DataFrame]
        Either the full calibration DataFrame (requires ``spectral_cuts``) or
        a pre-extracted zone dict such as ``smx.zones_natural_``.
    y_labels : pd.Series
        Class labels aligned row-wise with *zones*.
    output_path : str or Path, optional
        Destination file. If ``None``, no file is written.
    spectral_cuts : iterable, optional
        Zone boundary definitions.  Required when *zones* is a DataFrame.
    title : str, optional
        Figure title.
    class_colors : dict[str, str], optional
        Per-class hex/CSS colors.  Defaults to ``theme.class_colors``.
    theme : SMXTheme, optional
        Visual theme.
    width : int, default 1200
        Figure width (static export).
    height : int, default 580
        Figure height (static export).
    return_fig : bool, default False
        If ``True``, return ``(zone_scores_df, figure)`` for inline display.

    Returns
    -------
    pd.DataFrame
        Zone PC1 score DataFrame (samples × zones).
    """
    go = _require_plotly()
    from smx.zones.aggregation import ZoneAggregator

    theme = theme or DEFAULT_THEME
    _used: List[str] = []

    if isinstance(zones, pd.DataFrame):
        if spectral_cuts is None:
            raise ValueError("spectral_cuts is required when zones is a DataFrame.")
        from smx.zones.extraction import extract_spectral_zones
        zone_dict = extract_spectral_zones(zones, spectral_cuts)
    else:
        zone_dict = zones

    agg = ZoneAggregator(method="pca")
    agg.fit(zone_dict)
    zone_scores_df = agg.transform(zone_dict)
    zone_cols = zone_scores_df.columns.tolist()

    classes = list(y_labels.unique())
    split_mode = len(classes) == 2
    sides = {classes[0]: "negative", classes[1]: "positive"} if split_mode else {}

    fig = go.Figure()
    for cls in classes:
        mask = (y_labels == cls).values
        color = (class_colors or {}).get(str(cls)) or theme.resolve_class_color(str(cls), _used)
        for zone in zone_cols:
            fig.add_trace(go.Violin(
                x=[zone] * int(mask.sum()),
                y=zone_scores_df.loc[mask, zone].values,
                name=f"Class {cls}",
                legendgroup=str(cls),
                showlegend=(zone == zone_cols[0]),
                side=sides.get(cls, "both"),
                line_color=color,
                fillcolor=color,
                opacity=0.85,
                box_visible=False,
                meanline_visible=True,
                points=False,
                width=0.6,
            ))

    fig.update_layout(
        **theme.plotly_layout(
            title=title or "PC1 Scores by Spectral Zone and Class",
            xaxis=dict(title="Spectral Zone", tickangle=-30),
            yaxis=dict(title="PC 1 Score"),
            violingap=0.05,
            violingroupgap=0.0,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
            margin=dict(t=80, r=40, b=140, l=80),
        )
    )

    _write_figure(fig, output_path, width, height)
    if return_fig:
        return zone_scores_df, fig
    return zone_scores_df


# ── 4. All-Zone Threshold Overlay ──────────────────────────────────────────────

def plot_all_thresholds_overlay(
    lrc_natural_df: pd.DataFrame,
    zones_natural: Dict[str, pd.DataFrame],
    pca_info_natural: Dict,
    y_labels: pd.Series,
    spectral_cuts: Iterable,
    output_path: Optional[Union[str, Path]],
    *,
    title: Optional[str] = None,
    class_colors: Optional[Dict[str, str]] = None,
    theme: Optional[SMXTheme] = None,
    width: int = 1200,
    height: int = 500,
    return_fig: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, "go.Figure"]]:
    """Full-spectrum overlay of the top-ranked threshold per zone.

    Mean class spectra are drawn as solid lines across the full spectral axis.
    The top-ranked predicate threshold for each zone is reconstructed from PCA
    space and overlaid as a dashed line within its zone's x-range.  Threshold
    line colors follow the LRC-score colorscale so the most influential zones
    stand out visually.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        ``smx.lrc_natural_``.
    zones_natural : dict[str, pd.DataFrame]
        ``smx.zones_natural_``.
    pca_info_natural : dict
        ``smx.pca_info_natural_``.
    y_labels : pd.Series
        Class labels aligned row-wise with the calibration data.
    spectral_cuts : iterable
        Zone boundary definitions.
    output_path : str or Path, optional
        Destination file. If ``None``, no file is written.
    title : str, optional
        Figure title.
    class_colors : dict[str, str], optional
        Per-class hex/CSS colors.
    theme : SMXTheme, optional
        Visual theme.
    width : int, default 1200
        Figure width (static export).
    height : int, default 500
        Figure height (static export).
    return_fig : bool, default False
        If ``True``, return ``(top_thresholds_df, figure)`` for inline display.

    Returns
    -------
    pd.DataFrame
        Top-predicate-per-zone DataFrame used in the plot.
    """
    go = _require_plotly()
    from plotly.colors import sample_colorscale
    from smx.graph.interpretation import reconstruct_threshold_to_spectrum

    theme = theme or DEFAULT_THEME
    _used: List[str] = []

    top_per_zone = (
        lrc_natural_df[lrc_natural_df["Zone"].notna()]
        .sort_values("Local_Reaching_Centrality", ascending=False)
        .drop_duplicates(subset=["Zone"])
        .reset_index(drop=True)
    )

    score_min = float(top_per_zone["Local_Reaching_Centrality"].min())
    score_max = float(top_per_zone["Local_Reaching_Centrality"].max())

    def _lrc_color(score: float) -> str:
        norm = (score - score_min) / max(score_max - score_min, 1e-9)
        return sample_colorscale(theme.colorscale, [norm])[0]

    # Parse spectral cuts to get zone names and boundaries
    cut_rows = []
    for cut in spectral_cuts:
        if isinstance(cut, dict):
            cut_rows.append((str(cut["name"]), float(cut["start"]), float(cut["end"])))
        elif len(cut) == 3:
            cut_rows.append((str(cut[0]), float(cut[1]), float(cut[2])))
        else:
            cut_rows.append((f"{cut[0]}-{cut[1]}", float(cut[0]), float(cut[1])))

    fig = go.Figure()

    # ── Mean class spectra (full spectrum, solid) ──────────────────────────
    for cls in y_labels.unique():
        mask = (y_labels == cls).values
        parts = []
        for zone_name, _, _ in cut_rows:
            zone_df = zones_natural.get(zone_name)
            if zone_df is None or zone_df.empty:
                continue
            zone_mean = zone_df[mask].mean(axis=0)
            zone_mean.index = pd.to_numeric(zone_mean.index.astype(str), errors="coerce")
            parts.append(zone_mean)
        if not parts:
            continue
        full_mean = pd.concat(parts).sort_index().dropna()
        color = (class_colors or {}).get(str(cls)) or theme.resolve_class_color(str(cls), _used)
        fig.add_trace(go.Scatter(
            x=full_mean.index.to_numpy(dtype=float),
            y=full_mean.to_numpy(dtype=float),
            mode="lines",
            line=dict(color=color, width=theme.class_line_width),
            name=f"Class {cls}",
        ))

    # ── Per-zone threshold spectra (dashed, LRC-colored) ──────────────────
    for _, row in top_per_zone.iterrows():
        zone_name = str(row["Zone"])
        threshold_score = float(row["Threshold_Natural"])
        lrc_score = float(row["Local_Reaching_Centrality"])

        threshold_spectrum = reconstruct_threshold_to_spectrum(
            threshold_value=threshold_score,
            zone_name=zone_name,
            pca_info_dict=pca_info_natural,
        )
        threshold_spectrum.index = pd.to_numeric(
            threshold_spectrum.index.astype(str), errors="coerce"
        )
        threshold_spectrum = threshold_spectrum.dropna().sort_index()

        t_color = _lrc_color(lrc_score)
        fig.add_trace(go.Scatter(
            x=threshold_spectrum.index.to_numpy(dtype=float),
            y=threshold_spectrum.to_numpy(dtype=float),
            mode="lines",
            line=dict(
                color=t_color,
                width=theme.threshold_line_width,
                dash=theme.threshold_line_dash,
            ),
            name=f"Threshold: {zone_name} (LRC {lrc_score:.3f})",
        ))

    # ── Zone boundary vlines ───────────────────────────────────────────────
    boundaries = sorted({start for _, start, _ in cut_rows} | {end for _, _, end in cut_rows})
    for b in boundaries:
        fig.add_vline(x=b, line=dict(
            color=theme.zone_boundary_color,
            width=theme.zone_boundary_width,
            dash=theme.zone_boundary_dash,
        ))

    fig.update_layout(
        **theme.plotly_layout(
            title=title or "All-Zone Threshold Overlay",
            xaxis_title="Energy / Wavelength",
            yaxis_title="Intensity",
            legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center"),
            margin=dict(t=80, r=40, b=150, l=80),
        )
    )

    _write_figure(fig, output_path, width, height)
    if return_fig:
        return top_per_zone, fig
    return top_per_zone
