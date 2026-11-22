"""
Summary and diagnostic plots for SMX explanation results.

Functions
---------
plot_lrc_bar
    Horizontal bar chart of LRC scores per zone.
plot_predicate_heatmap
    Zone × predicate heatmap of LRC scores.
plot_zone_scores
    Split-violin of zone scores (PC1 by default) per zone by class.
plot_all_thresholds_overlay
    Full-spectrum overlay of the top-predicate threshold per zone.
plot_faithfulness_curve
    Progressive masking curve with shaded AUC and summary annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from smx.plotting._common import (
    PathLike,
    boundary_line,
    class_labels,
    finalize_figure,
    legend_below,
    normalize_cuts,
    prepare_zone_ranking_df,
    require_plotly,
    resolve_theme,
    score_colorbar,
    score_colors,
    stitch_zones,
    unique_classes,
)
from smx.plotting.theme import SMXTheme, blend_with_white, build_blended_colorscale

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from smx.zones.aggregation import ZoneAggregator

FigureOrTuple = Union["go.Figure", Tuple["go.Figure", pd.DataFrame]]


# ── 1. LRC Bar Chart ───────────────────────────────────────────────────────────

def plot_lrc_bar(
    zone_ranking_df: pd.DataFrame,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 800,
    height: Optional[int] = 500,
    return_df: bool = False,
) -> FigureOrTuple:
    """Horizontal bar chart of LRC scores per zone.

    Each bar shows the zone's share of the total LRC and is coloured with the
    same colorscale as :func:`plot_zone_ranking_over_spectrum`, making the two
    plots directly comparable.

    Parameters
    ----------
    zone_ranking_df : pd.DataFrame
        LRC table (``Zone`` / ``Local_Reaching_Centrality`` columns) or a
        ``zone`` / ``score`` / ``rank`` DataFrame.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    colorscale : str, optional
        Plotly colorscale name.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.  Defaults to :data:`~smx.plotting.theme.DEFAULT_THEME`.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, ranking_df)`` with the normalized
        ``zone`` / ``score`` / ``rank`` / ``pct`` table (highest score first).

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    go = require_plotly()
    theme = resolve_theme(theme)
    colorscale = colorscale or theme.colorscale

    ranking_df = prepare_zone_ranking_df(zone_ranking_df)
    ranking_df["pct"] = ranking_df["score"] / max(float(ranking_df["score"].sum()), 1e-12) * 100
    score_min = float(ranking_df["score"].min())
    score_max = float(ranking_df["score"].max())

    # Plotly draws the first category at the bottom: reverse so #1 is on top.
    bars = ranking_df.iloc[::-1]
    colors = [
        blend_with_white(c, theme.zone_opacity)
        for c in score_colors(bars["score"], colorscale, score_min, score_max)
    ]

    fig = go.Figure(go.Bar(
        x=bars["pct"],
        y=[f"#{r}  {z}" for r, z in zip(bars["rank"], bars["zone"])],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#555555", width=1)),
        text=[f"{p:.1f}%" for p in bars["pct"]],
        textposition="outside",
        cliponaxis=False,
        customdata=bars["score"].to_numpy(),
        hovertemplate="Zone: %{y}<br>Share: %{x:.2f}%<br>LRC: %{customdata:.4f}<extra></extra>",
    ))

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="LRC score by spectral zone",
        width=width,
        height=height,
        output_path=output_path,
        data=ranking_df,
        return_df=return_df,
        xaxis=dict(title="LRC score (% of total)", range=[0, float(ranking_df["pct"].max()) * 1.2]),
        yaxis=dict(title="Zone"),
        margin=dict(t=80, r=60, b=60, l=160),
    )


# ── 2. Predicate Heatmap ───────────────────────────────────────────────────────

_OPERATOR_SYMBOLS = {"<=": "≤", "<": "<", ">=": "≥", ">": ">"}


def plot_predicate_heatmap(
    lrc_natural_df: pd.DataFrame,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1000,
    height: Optional[int] = 550,
    return_df: bool = False,
) -> FigureOrTuple:
    """Heatmap of LRC scores across zones and predicate thresholds.

    Rows are spectral zones (highest maximum LRC at the top).  Columns are the
    predicates of each zone, grouped by operator (``≤`` then ``>``) and
    numbered by increasing threshold within each group (``T1``, ``T2``, …).
    Cell colour encodes the LRC score on the same colorscale as the bar chart
    and zone-ranking plot; zones without a given predicate are left grey.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        LRC table — must contain ``Zone``, ``Operator``, ``Threshold_Natural``
        and ``Local_Reaching_Centrality`` columns (``SMX.lrc_natural_``).
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    colorscale : str, optional
        Plotly colorscale name.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, pivot)`` with the zones × predicates
        LRC table (rows in display order, top row first).

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    go = require_plotly()
    theme = resolve_theme(theme)
    colorscale = colorscale or theme.colorscale

    required = {"Zone", "Operator", "Threshold_Natural", "Local_Reaching_Centrality"}
    missing = required.difference(lrc_natural_df.columns)
    if missing:
        raise ValueError("lrc_natural_df is missing required columns: " + ", ".join(sorted(missing)))

    df = lrc_natural_df[lrc_natural_df["Zone"].notna()].copy()
    if df.empty:
        raise ValueError("lrc_natural_df has no rows with a zone.")
    df["Zone"] = df["Zone"].astype(str)
    df = df.sort_values(["Zone", "Operator", "Threshold_Natural"], kind="stable")
    df["thresh_rank"] = df.groupby(["Zone", "Operator"]).cumcount() + 1

    operator_order = {op: i for i, op in enumerate(["<=", "<", ">=", ">"])}
    column_keys = sorted(
        {(op, rank) for op, rank in zip(df["Operator"], df["thresh_rank"])},
        key=lambda key: (operator_order.get(key[0], len(operator_order)), str(key[0]), key[1]),
    )
    labels = {key: f"{_OPERATOR_SYMBOLS.get(key[0], key[0])} T{key[1]}" for key in column_keys}
    df["predicate_label"] = [labels[(op, r)] for op, r in zip(df["Operator"], df["thresh_rank"])]

    zone_order = (
        df.groupby("Zone")["Local_Reaching_Centrality"].max().sort_values(ascending=False).index.tolist()
    )
    pivot = df.pivot_table(
        index="Zone",
        columns="predicate_label",
        values="Local_Reaching_Centrality",
        aggfunc="max",
    ).reindex(index=zone_order, columns=[labels[key] for key in column_keys])
    pivot.columns.name = "Predicate"

    score_min = float(df["Local_Reaching_Centrality"].min())
    score_max = float(df["Local_Reaching_Centrality"].max())
    z = pivot.to_numpy(dtype=float)
    text = [[f"{v:.3f}" if np.isfinite(v) else "—" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        # Plotly draws the first row at the bottom: reverse so the top zone is on top.
        z=np.where(np.isfinite(z), z, None)[::-1].tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist()[::-1],
        text=text[::-1],
        texttemplate="%{text}",
        textfont=dict(size=9, family=theme.font_family),
        colorscale=build_blended_colorscale(colorscale, theme.zone_opacity),
        zmin=score_min,
        zmax=score_max,
        colorbar=score_colorbar(theme, score_min, score_max),
        hovertemplate="Zone: %{y}<br>Predicate: %{x}<br>LRC: %{text}<extra></extra>",
        hoverongaps=False,
        xgap=2,
        ygap=2,
    ))

    # Group headers (and a separator) for each operator.
    columns = pivot.columns.tolist()
    groups: Dict[str, list] = {}
    for i, (op, _) in enumerate(column_keys):
        groups.setdefault(op, []).append(i)
    for n, (op, positions) in enumerate(groups.items()):
        fig.add_annotation(
            x=(positions[0] + positions[-1]) / 2,
            y=1.0,
            xref="x",
            yref="paper",
            yanchor="bottom",
            text=f"Operator {_OPERATOR_SYMBOLS.get(op, op)}",
            showarrow=False,
            font=dict(size=theme.annotation_font_size, family=theme.font_family),
        )
        if n:
            fig.add_vline(x=positions[0] - 0.5, line=dict(color="white", width=4))

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="Predicate LRC heatmap",
        width=width,
        height=height,
        output_path=output_path,
        data=pivot,
        return_df=return_df,
        xaxis=dict(
            title="Predicate (operator · threshold rank)",
            categoryorder="array",
            categoryarray=columns,
            showgrid=False,
        ),
        yaxis=dict(title="Zone", showgrid=False),
        margin=dict(t=100, r=120, b=80, l=160),
        plot_bgcolor=theme.no_data_color,
    )


# ── 3. Zone Score Violin ───────────────────────────────────────────────────────

def plot_zone_scores(
    zones: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    y_labels: Union[pd.Series, Sequence],
    spectral_cuts: Optional[Iterable] = None,
    *,
    aggregator: Optional["ZoneAggregator"] = None,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 580,
    return_df: bool = False,
) -> FigureOrTuple:
    """Violin plot of per-zone scores, split by class.

    With exactly two classes the violins are mirrored (split); with three or
    more classes they are drawn side by side.

    Parameters
    ----------
    zones : pd.DataFrame or dict[str, pd.DataFrame]
        Either a full spectra DataFrame (requires *spectral_cuts*) or a zone
        dictionary such as ``SMX.zones_natural_``.
    y_labels : pd.Series or array-like
        Class labels aligned row by row with *zones*.
    spectral_cuts : iterable, optional
        Zone definitions.  Required when *zones* is a DataFrame.
    aggregator : ZoneAggregator, optional
        A *fitted* aggregator used to compute the zone scores, e.g. to reuse
        the PCA fitted on calibration data.  When omitted, a PCA aggregator is
        fitted on *zones*.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    class_colors : dict[str, str], optional
        Per-class colours.  Overrides ``theme.class_colors``.
    theme : SMXTheme, optional
        Visual theme.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, zone_scores_df)`` (samples × zones).

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    if isinstance(spectral_cuts, (str, Path)):
        raise TypeError("output_path is keyword-only: use plot_zone_scores(..., output_path=...).")

    go = require_plotly()
    from smx.zones.aggregation import ZoneAggregator

    theme = resolve_theme(theme)

    if isinstance(zones, pd.DataFrame):
        if spectral_cuts is None:
            raise ValueError("spectral_cuts is required when zones is a DataFrame.")
        from smx.zones.extraction import extract_spectral_zones

        zone_dict = extract_spectral_zones(zones, spectral_cuts)
    else:
        zone_dict = zones

    if aggregator is None:
        aggregator = ZoneAggregator(method="pca").fit(zone_dict)
    elif not aggregator.is_fitted_:
        raise ValueError("aggregator must be fitted before being passed to plot_zone_scores().")
    zone_scores_df = aggregator.transform(zone_dict)
    zone_cols = [str(c) for c in zone_scores_df.columns]

    labels = class_labels(y_labels, n_rows=len(zone_scores_df))
    classes = unique_classes(labels)
    colors = theme.class_color_map(classes, class_colors)
    split = len(classes) == 2
    sides = dict(zip(classes, ["negative", "positive"])) if split else {}

    fig = go.Figure()
    # One trace per class and zone so each violin is scaled on its own
    # (violins sharing a trace share one density scale).
    for cls in classes:
        mask = (labels == cls).to_numpy()
        for n, zone in enumerate(zone_scores_df.columns):
            fig.add_trace(go.Violin(
                x=[zone_cols[n]] * int(mask.sum()),
                y=zone_scores_df[zone].to_numpy(dtype=float)[mask],
                name=f"Class {cls}",
                legendgroup=str(cls),
                showlegend=n == 0,
                offsetgroup=str(cls),
                alignmentgroup="zones",
                side=sides.get(cls, "both"),
                line_color=colors[cls],
                fillcolor=colors[cls],
                opacity=0.85,
                box_visible=False,
                meanline_visible=True,
                points=False,
                width=0.6 if split else None,
            ))

    score_name = "PC1 score" if aggregator.method == "pca" else f"Zone score ({aggregator.method})"
    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title=f"{score_name} by spectral zone and class",
        width=width,
        height=height,
        output_path=output_path,
        data=zone_scores_df,
        return_df=return_df,
        xaxis=dict(title="Spectral zone", tickangle=-30, categoryorder="array", categoryarray=zone_cols),
        yaxis=dict(title=score_name),
        violinmode="overlay" if split else "group",
        violingap=0.05,
        violingroupgap=0.05,
        legend=legend_below(),
        margin=dict(t=80, r=40, b=140, l=80),
    )


# ── 4. All-Zone Threshold Overlay ──────────────────────────────────────────────

def plot_all_thresholds_overlay(
    lrc_natural_df: pd.DataFrame,
    zones_natural: Dict[str, pd.DataFrame],
    pca_info_natural: Dict,
    y_labels: Optional[Union[pd.Series, Sequence]],
    spectral_cuts: Iterable,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    class_colors: Optional[Dict[str, str]] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1200,
    height: Optional[int] = 500,
    return_df: bool = False,
) -> FigureOrTuple:
    """Full-spectrum overlay of the top-ranked threshold per zone.

    Per-class mean spectra (with their min–max envelope shaded) are drawn
    across the full spectral axis.  The top-ranked predicate threshold of each
    zone is reconstructed from PCA space and drawn as a dashed line within the
    zone, coloured by LRC score so the most influential zones stand out.

    Parameters
    ----------
    lrc_natural_df : pd.DataFrame
        ``SMX.lrc_natural_``.
    zones_natural : dict[str, pd.DataFrame]
        ``SMX.zones_natural_``.
    pca_info_natural : dict
        ``SMX.pca_info_natural_``.
    y_labels : pd.Series or array-like, optional
        Class labels aligned row by row with the zone DataFrames.  When
        ``None``, a single mean spectrum over all samples is drawn.
    spectral_cuts : iterable
        Zone boundary definitions.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    class_colors : dict[str, str], optional
        Per-class colours.  Overrides ``theme.class_colors``.
    colorscale : str, optional
        Colorscale for the threshold lines.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.
    width, height : int, optional
        Figure size in pixels (display and export).
    return_df : bool, default False
        If ``True``, return ``(fig, top_per_zone)`` with the top predicate of
        each zone (highest LRC first).

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    go = require_plotly()
    from plotly.colors import sample_colorscale

    from smx.graph.interpretation import reconstruct_threshold_to_spectrum

    theme = resolve_theme(theme)
    colorscale = colorscale or theme.colorscale

    cuts_df = normalize_cuts(spectral_cuts)
    zone_names = [z for z in dict.fromkeys(cuts_df["zone"]) if z in zones_natural]
    if not zone_names:
        raise ValueError("None of the zones in spectral_cuts are present in zones_natural.")
    n_rows = len(zones_natural[zone_names[0]])

    top_per_zone = (
        lrc_natural_df[lrc_natural_df["Zone"].notna()]
        .sort_values("Local_Reaching_Centrality", ascending=False, kind="stable")
        .drop_duplicates(subset=["Zone"])
        .reset_index(drop=True)
    )
    if top_per_zone.empty:
        raise ValueError("lrc_natural_df has no rows with a zone.")

    if y_labels is None:
        labels = pd.Series(["all"] * n_rows)
        colors = {"all": theme.reference_line_color}
        names = {"all": "All samples"}
    else:
        labels = class_labels(y_labels, n_rows=n_rows)
        colors = theme.class_color_map(unique_classes(labels), class_colors)
        names = {cls: f"Class {cls}" for cls in colors}

    fig = go.Figure()
    for cls, color in colors.items():
        mask = (labels == cls).to_numpy()
        subset = {z: zones_natural[z][mask] for z in zone_names}
        upper = stitch_zones(subset, zone_names, lambda df: df.max(axis=0))
        lower = stitch_zones(subset, zone_names, lambda df: df.min(axis=0)).reindex(upper.index)
        mean = stitch_zones(subset, zone_names, lambda df: df.mean(axis=0))
        x = upper.index.to_numpy(dtype=float)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper.to_numpy(dtype=float), lower.to_numpy(dtype=float)[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.18,
            line=dict(width=0),
            name=f"{names[cls]} range",
            legendgroup=str(cls),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=mean.index.to_numpy(dtype=float),
            y=mean.to_numpy(dtype=float),
            mode="lines",
            line=dict(color=color, width=theme.class_line_width),
            name=f"{names[cls]} mean (min–max shaded)",
            legendgroup=str(cls),
        ))

    # Sample the colorscale from 25 % up so low-LRC thresholds stay visible.
    score_min = float(top_per_zone["Local_Reaching_Centrality"].min())
    score_max = float(top_per_zone["Local_Reaching_Centrality"].max())
    stops = np.linspace(0.25, 1.0, 16)
    visible_scale = [[float(i / 15), c] for i, c in enumerate(sample_colorscale(colorscale, list(stops)))]
    line_colors = score_colors(top_per_zone["Local_Reaching_Centrality"], visible_scale, score_min, score_max)

    for n, (row, color) in enumerate(zip(top_per_zone.itertuples(index=False), line_colors)):
        zone_name = str(row.Zone)
        threshold = reconstruct_threshold_to_spectrum(
            threshold_value=float(row.Threshold_Natural),
            zone_name=zone_name,
            pca_info_dict=pca_info_natural,
        )
        threshold.index = pd.to_numeric(threshold.index.astype(str), errors="coerce")
        threshold = threshold[~threshold.index.isna()].sort_index()
        lrc = float(row.Local_Reaching_Centrality)
        fig.add_trace(go.Scatter(
            x=threshold.index.to_numpy(dtype=float),
            y=threshold.to_numpy(dtype=float),
            mode="lines",
            line=dict(color=color, width=theme.threshold_line_width, dash=theme.threshold_line_dash),
            name="Top threshold per zone",
            legendgroup="thresholds",
            showlegend=n == 0,
            hovertemplate=f"Zone: {zone_name}<br>LRC: {lrc:.4f}<br>x: %{{x}}<br>y: %{{y:.4g}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=visible_scale,
            cmin=score_min,
            cmax=score_max,
            color=[score_min],
            size=0,
            opacity=0,
            showscale=True,
            colorbar=score_colorbar(theme, score_min, score_max, title="Threshold LRC"),
        ),
        hoverinfo="skip",
        showlegend=False,
    ))

    for boundary in sorted(set(cuts_df["start"]) | set(cuts_df["end"])):
        fig.add_vline(x=boundary, line=boundary_line(theme))

    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="Top threshold per zone over the class spectra",
        width=width,
        height=height,
        output_path=output_path,
        data=top_per_zone,
        return_df=return_df,
        xaxis_title="Spectral variables",
        yaxis_title="Intensity",
        legend=legend_below(),
        margin=dict(t=80, r=40, b=110, l=80),
    )


# ── 5. Faithfulness Curve ─────────────────────────────────────────────────────

def _level_intervals_text() -> str:
    from smx.evaluation.faithfulness import FAITHFULNESS_LEVEL_BOUNDS

    lines, lower = [], None
    for bound, level in FAITHFULNESS_LEVEL_BOUNDS:
        lines.append(f"{level}: &lt; {bound:g}" if lower is None else f"{level}: {lower:g} – {bound:g}")
        lower = bound
    lines.append(f"Very High: ≥ {lower:g}")
    return "<b>Level by percentile</b><br>" + "<br>".join(lines)


def plot_faithfulness_curve(
    faithfulness_result: Dict,
    *,
    output_path: Optional[PathLike] = None,
    title: Optional[Union[str, bool]] = None,
    colorscale: Optional[str] = None,
    theme: Optional[SMXTheme] = None,
    width: Optional[int] = 1100,
    height: Optional[int] = 560,
    show_percentile: bool = False,
    show_faithfulness_level: bool = True,
    show_summary: bool = True,
    show_level_intervals: bool = True,
    return_df: bool = False,
) -> FigureOrTuple:
    """Plot the progressive masking faithfulness curve and its AUC.

    The summary (faithfulness level, AUC, metric and level intervals) is shown
    in a panel to the right of the plot so it never hides the curve.

    Parameters
    ----------
    faithfulness_result : dict
        Output of :meth:`smx.pipeline.SMX.evaluate_faithfulness` or
        :func:`smx.progressive_masking_faithfulness`.  Must contain
        ``curve_df`` (with ``k`` and ``score`` columns); ``auc``,
        ``auc_normalized``, ``level``, ``null_percentile``, ``metric`` and
        ``n_masked_zones`` are shown when present.
    output_path : str or Path, optional
        Export destination (``.html`` or a static image suffix).  Nothing is
        written when ``None``.
    title : str, optional
        Figure title.  ``None`` uses a default title; ``""`` removes it.
    colorscale : str, optional
        Colorscale for the curve and markers.  Defaults to ``theme.colorscale``.
    theme : SMXTheme, optional
        Visual theme.
    width, height : int, optional
        Figure size in pixels (display and export).
    show_percentile : bool, default False
        Include the random-baseline percentile in the summary.
    show_faithfulness_level : bool, default True
        Show the faithfulness level.
    show_summary : bool, default True
        Show AUC, normalized AUC and metric.
    show_level_intervals : bool, default True
        Show the percentile intervals of each faithfulness level.
    return_df : bool, default False
        If ``True``, return ``(fig, curve_df)``.

    Returns
    -------
    plotly.graph_objects.Figure or (Figure, pd.DataFrame)
    """
    go = require_plotly()
    theme = resolve_theme(theme)
    colorscale = colorscale or theme.colorscale

    if not isinstance(faithfulness_result, dict):
        raise TypeError("faithfulness_result must be a dictionary.")
    if "curve_df" not in faithfulness_result:
        raise ValueError("faithfulness_result must contain 'curve_df'.")
    curve_df = faithfulness_result["curve_df"]
    if not isinstance(curve_df, pd.DataFrame) or curve_df.empty:
        raise ValueError("faithfulness_result['curve_df'] must be a non-empty DataFrame.")
    missing = {"k", "score"}.difference(curve_df.columns)
    if missing:
        raise ValueError(
            "faithfulness_result['curve_df'] is missing required columns: " + ", ".join(sorted(missing))
        )

    curve_df = curve_df.sort_values("k").reset_index(drop=True)
    # The curve starts at (0, 0): nothing masked means no prediction shift.
    k_line = np.concatenate([[0.0], curve_df["k"].to_numpy(dtype=float)])
    score_line = np.concatenate([[0.0], curve_df["score"].to_numpy(dtype=float)])

    score_min = float(curve_df["score"].min())
    score_max = float(curve_df["score"].max())
    blended = build_blended_colorscale(colorscale, theme.zone_opacity)
    (line_color,) = score_colors([score_max], colorscale, score_min, score_max)
    (fill_color,) = score_colors([score_max], blended, score_min, score_max)
    zone_labels = (
        curve_df["masked_zone"].astype(str).to_numpy()
        if "masked_zone" in curve_df.columns
        else np.repeat("", len(curve_df))
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_line,
        y=score_line,
        mode="lines",
        name="Faithfulness curve (AUC shaded)",
        line=dict(color=line_color, width=theme.threshold_line_width + 1),
        fill="tozeroy",
        fillcolor=fill_color,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=curve_df["k"],
        y=curve_df["score"],
        mode="markers+text",
        name="Prediction shift after masking",
        marker=dict(
            size=10,
            color=curve_df["score"],
            colorscale=blended,
            cmin=score_min,
            cmax=score_max,
            line=dict(color=line_color, width=1.5),
        ),
        text=zone_labels,
        textposition="top center",
        textfont=dict(size=theme.annotation_font_size, family=theme.font_family),
        cliponaxis=False,
        customdata=zone_labels,
        hovertemplate=(
            "k masked zones: %{x}<br>"
            "Prediction shift: %{y:.6f}<br>"
            "Latest masked zone: %{customdata}<extra></extra>"
        ),
    ))
    for k in curve_df["k"]:
        fig.add_vline(x=float(k), line=boundary_line(theme), layer="below")

    sections = []
    level = faithfulness_result.get("level")
    if show_faithfulness_level and level is not None:
        sections.append(f"Faithfulness level<br><b>{level}</b>")
    if show_summary:
        lines = []
        for key, label, fmt in [
            ("auc", "AUC", "{:.4f}"),
            ("auc_normalized", "Normalized AUC", "{:.4f}"),
            ("null_percentile", "Percentile", "{:.1f}%"),
            ("metric", "Metric", "{}"),
        ]:
            value = faithfulness_result.get(key)
            if value is None or (key == "null_percentile" and not show_percentile):
                continue
            lines.append(f"{label}: " + fmt.format(value if key == "metric" else float(value)))
        if lines:
            sections.append("<br>".join(lines))
    if show_level_intervals:
        sections.append(_level_intervals_text())
    if sections:
        fig.add_annotation(
            x=1.02,
            y=1.0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bordercolor="rgba(140,140,140,0.35)",
            borderwidth=1,
            borderpad=8,
            bgcolor="rgba(255,255,255,0.88)",
            text="<br><br>".join(sections),
        )

    n_masked = faithfulness_result.get("n_masked_zones") or 0
    return finalize_figure(
        fig,
        theme=theme,
        title=title,
        default_title="Faithfulness via progressive zone masking",
        width=width,
        height=height,
        output_path=output_path,
        data=curve_df,
        return_df=return_df,
        xaxis=dict(
            title="k masked zones (cumulative top-ranked masking)",
            tickmode="linear",
            tick0=0,
            dtick=1,
            range=[-0.2, max(float(n_masked), float(curve_df["k"].max())) + 0.4],
        ),
        yaxis=dict(title="Prediction shift score", rangemode="tozero"),
        legend=legend_below(),
        margin=dict(t=90, r=230 if sections else 40, b=100, l=80),
    )
