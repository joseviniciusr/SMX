"""
SMX visual theme system.

All plotting helpers accept an optional ``theme`` argument of type
:class:`SMXTheme`.  When omitted, :data:`DEFAULT_THEME` is used.

Example — using the default theme::

    from smx.plotting import plot_zone_ranking_over_spectrum
    plot_zone_ranking_over_spectrum(..., output_path="out.html")

Example — overriding selected fields::

    from smx.plotting.theme import SMXTheme
    my_theme = SMXTheme(font_family="Georgia", colorscale="Blues")
    plot_zone_ranking_over_spectrum(..., output_path="out.html", theme=my_theme)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
        return self.fallback_palette[hash(label) % len(self.fallback_palette)]

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
    plot background.
    """
    import re
    vals = [int(v) for v in re.findall(r"\d+", rgb_str)][:3]
    r = int(opacity * vals[0] + (1 - opacity) * 255)
    g = int(opacity * vals[1] + (1 - opacity) * 255)
    b = int(opacity * vals[2] + (1 - opacity) * 255)
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
