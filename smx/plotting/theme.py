"""
SMX visual theme system.

All plotting helpers accept an optional ``theme`` argument of type
:class:`SMXTheme`.  When omitted, :data:`DEFAULT_THEME` is used.

Example — using the default theme::

    from smx.plotting import plot_zone_ranking_over_spectrum
    fig = plot_zone_ranking_over_spectrum(...)

Example — overriding selected fields::

    from smx.plotting.theme import SMXTheme
    my_theme = SMXTheme(font_family="Georgia", colorscale="Blues")
    fig = plot_zone_ranking_over_spectrum(..., theme=my_theme)
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional


@dataclass
class SMXTheme:
    """Visual style configuration shared across all SMX plots.

    Parameters
    ----------
    template : str
        Plotly layout template (e.g. ``'plotly_white'``, ``'simple_white'``).
    font_family : str
        CSS font-family string applied to all text in the figure.
    font_size : int
        Base font size (px) for axis labels, tick labels, and annotations.
    class_colors : dict[str, str]
        Mapping from class label to hex/CSS color string.  Labels not present
        fall back to ``fallback_palette``.
    fallback_palette : list[str]
        Ordered list of colors used for class labels not found in
        ``class_colors``.
    colorscale : str
        Plotly colorscale name used for LRC-score zone bands and the colorbar.
    zone_opacity : float
        Opacity applied to zone background rectangles (vrect).
    no_data_color : str
        Fill color for cells/zones without a score (e.g. missing predicates).
    reference_line_color : str
        Color for the overall reference/mean spectrum line.
    reference_line_width : int
        Stroke width (px) for the reference spectrum line.
    reference_line_dash : str
        Plotly dash style for the reference spectrum (e.g. ``'dash'``).
    class_line_width : int
        Stroke width (px) for per-class mean spectrum lines.
    threshold_color : str
        Color for reconstructed threshold spectrum lines.
    threshold_line_width : int
        Stroke width (px) for threshold spectrum lines.
    threshold_line_dash : str
        Plotly dash style for threshold lines.
    zone_boundary_color : str
        Color for the vertical dotted zone-boundary lines.
    zone_boundary_width : int
        Stroke width (px) for zone boundary lines.
    zone_boundary_dash : str
        Plotly dash style for zone boundary lines.
    colorbar_thickness : int
        Thickness (px) of the LRC-score colorbar.
    colorbar_len : float
        Fractional length of the colorbar relative to the plot height.
    annotation_font_size : int
        Font size (px) used for zone-label annotations above the plot.
    """

    # ── Layout ────────────────────────────────────────────────────────────────
    template: str = "plotly_white"
    font_family: str = "Inter, Helvetica Neue, Arial, sans-serif"
    font_size: int = 13

    # ── Class colours ─────────────────────────────────────────────────────────
    class_colors: Dict[str, str] = field(default_factory=lambda: {
        "A": "#e41a1c",
        "B": "#377eb8",
        "C": "#4daf4a",
        "D": "#984ea3",
        "E": "#ff7f00",
        "F": "#a65628",
        "G": "#f781bf",
        "H": "#999999",
    })
    fallback_palette: List[str] = field(default_factory=lambda: [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ])

    # ── Zone ranking colorscale ────────────────────────────────────────────────
    colorscale: str = "YlOrRd"
    zone_opacity: float = 0.28
    no_data_color: str = "rgb(220,220,220)"

    # ── Reference / mean spectrum ──────────────────────────────────────────────
    reference_line_color: str = "#2b2b2b"
    reference_line_width: int = 2
    reference_line_dash: str = "dash"

    # ── Per-class mean spectrum ────────────────────────────────────────────────
    class_line_width: int = 2

    # ── Threshold spectrum ─────────────────────────────────────────────────────
    threshold_color: str = "#c0392b"
    threshold_line_width: int = 3
    threshold_line_dash: str = "dash"

    # ── Zone boundaries ───────────────────────────────────────────────────────
    zone_boundary_color: str = "rgba(80,80,80,0.25)"
    zone_boundary_width: int = 1
    zone_boundary_dash: str = "dot"

    # ── Colorbar ──────────────────────────────────────────────────────────────
    colorbar_thickness: int = 15
    colorbar_len: float = 0.75

    # ── Annotations ───────────────────────────────────────────────────────────
    annotation_font_size: int = 11

    # ──────────────────────────────────────────────────────────────────────────

    def resolve_class_color(self, label: str, _used: list | None = None) -> str:
        """Return the color for *label*, falling back to the palette if needed.

        Prefer :meth:`class_color_map` when coloring several labels at once:
        it guarantees distinct colors across the labels of a single figure.

        Parameters
        ----------
        label : str
            Class label to resolve.
        _used : list, optional
            Mutable list of already-consumed palette colors, used when
            assigning palette colors sequentially across multiple labels.
        """
        if label in self.class_colors:
            return self.class_colors[label]
        if _used is not None:
            for color in self.fallback_palette:
                if color not in _used:
                    _used.append(color)
                    return color
        # Deterministic across interpreter runs, unlike the built-in hash().
        index = zlib.crc32(str(label).encode("utf-8")) % len(self.fallback_palette)
        return self.fallback_palette[index]

    def class_color_map(
        self,
        labels: Iterable[Hashable],
        overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[Hashable, str]:
        """Return a ``{label: color}`` mapping for every label in *labels*.

        Colors are resolved with the precedence *overrides* →
        ``class_colors`` → ``fallback_palette``.  Labels are matched by their
        string form, so integer class labels work with string-keyed mappings.
        Palette colors already used by another label in the same call are
        skipped, so every label gets a distinct color while the palette lasts.
        """
        explicit = dict(self.class_colors)
        explicit.update({str(k): v for k, v in (overrides or {}).items()})

        labels = list(dict.fromkeys(labels))
        colors: Dict[Hashable, str] = {
            label: explicit[str(label)] for label in labels if str(label) in explicit
        }
        free = [c for c in self.fallback_palette if c not in colors.values()]
        free = free or list(self.fallback_palette)
        n_assigned = 0
        for label in labels:
            if label not in colors:
                colors[label] = free[n_assigned % len(free)]
                n_assigned += 1
        return {label: colors[label] for label in labels}

    def plotly_layout(self, **overrides) -> dict:
        """Return a ``fig.update_layout`` kwargs dict with theme base values.

        Any keyword passed as *overrides* takes precedence.
        """
        base = dict(
            template=self.template,
            font=dict(family=self.font_family, size=self.font_size),
        )
        base.update(overrides)
        return base


#: Default theme instance used by all SMX plotting functions.
DEFAULT_THEME = SMXTheme()


def blend_with_white(rgb_str: str, opacity: float) -> str:
    """Return the rgb string from compositing *rgb_str* over white at *opacity*.

    Used to match colorscale colors to the actual rendered appearance of zone
    background rectangles, which are drawn with fractional opacity over a white
    plot background.  Accepts ``rgb(...)``/``rgba(...)`` strings and hex colors.
    """
    if rgb_str.startswith("#"):
        from plotly.colors import hex_to_rgb

        vals = list(hex_to_rgb(rgb_str))
    else:
        vals = [float(v) for v in re.findall(r"[\d.]+", rgb_str)][:3]
    r, g, b = (int(round(opacity * v + (1 - opacity) * 255)) for v in vals)
    return f"rgb({r},{g},{b})"


def build_blended_colorscale(colorscale: str, opacity: float, n_stops: int = 32) -> list:
    """Build a Plotly colorscale whose colors are pre-blended with white.

    Parameters
    ----------
    colorscale : str
        Plotly colorscale name (e.g. ``'YlOrRd'``).
    opacity : float
        Opacity used when compositing over white.
    n_stops : int
        Number of discrete color stops in the returned colorscale.

    Returns
    -------
    list of [float, str]
        Colorscale in Plotly's ``[[position, color], ...]`` format.
    """
    from plotly.colors import sample_colorscale
    import numpy as np
    stops = np.linspace(0, 1, n_stops)
    return [
        [float(t), blend_with_white(c, opacity)]
        for t, c in zip(stops, sample_colorscale(colorscale, list(stops)))
    ]
