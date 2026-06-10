"""
Multi-class colour palette utilities for SMX plotting.

All plotting functions that display per-class information should import
``get_class_color_map`` from this module to guarantee consistent colour
assignment across figures.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Ordered list of perceptually distinct colours for up to 20 classes.
# Based on the ColorBrewer qualitative palettes (colorbrewer2.org).
_DEFAULT_PALETTE: List[str] = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
    "#66c2a5",  # teal
    "#fc8d62",  # salmon
    "#8da0cb",  # periwinkle
    "#e78ac3",  # rose
    "#a6d854",  # yellow-green
    "#ffd92f",  # yellow
    "#e5c494",  # sand
    "#b3b3b3",  # light grey
    "#1b9e77",  # dark teal
    "#d95f02",  # burnt orange
    "#7570b3",  # slate purple
    "#e7298a",  # magenta
]


def get_class_color_map(
    class_labels: List[str],
    custom_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return a mapping from class label to hex colour string.

    Parameters
    ----------
    class_labels : list of str
        Ordered list of unique class labels. The order determines
        which palette colour each label receives.
    custom_colors : dict[str, str], optional
        Explicit label-to-colour overrides. Labels present here take
        precedence over the auto-assigned palette colour.

    Returns
    -------
    dict[str, str]
        ``{label: '#rrggbb', ...}`` for every label in *class_labels*.
    """
    custom_colors = custom_colors or {}
    palette = _DEFAULT_PALETTE

    # If more classes than palette entries, cycle through the palette.
    color_map: Dict[str, str] = {}
    for i, label in enumerate(class_labels):
        label_str = str(label)
        if label_str in custom_colors:
            color_map[label_str] = custom_colors[label_str]
        else:
            color_map[label_str] = palette[i % len(palette)]

    return color_map


def resolve_color(
    label: str,
    class_labels: List[str],
    custom_colors: Optional[Dict[str, str]] = None,
) -> str:
    """Return the colour for a single class label.

    Parameters
    ----------
    label : str
        The label to resolve.
    class_labels : list of str
        Full ordered list of labels (for consistent index-based assignment).
    custom_colors : dict[str, str], optional
        Explicit overrides.

    Returns
    -------
    str
        Hex colour string, e.g. ``'#e41a1c'``.
    """
    color_map = get_class_color_map(class_labels, custom_colors)
    return color_map.get(str(label), "#808080")  # neutral grey fallback
