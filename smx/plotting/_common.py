"""Private helpers shared by the SMX plotting functions.

Every public plotting function follows the same contract:

* data arguments come first; everything else is keyword-only;
* ``output_path=None`` — optional export, format inferred from the suffix;
* ``title=None`` uses the plot's default title, ``title=""`` (or ``False``)
  removes it;
* ``width`` / ``height`` size the figure itself (display *and* export);
* the :class:`plotly.graph_objects.Figure` is returned and never shown, so it
  can be customised further; ``return_df=True`` returns ``(fig, data)``
  (a :class:`FigureWithData` tuple that renders both in Jupyter).
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from smx.plotting.theme import DEFAULT_THEME, SMXTheme

PathLike = Union[str, Path]

STATIC_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"}


def require_plotly():
    """Import and return :mod:`plotly.graph_objects` with a helpful error."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for SMX plotting. Install it with: "
            "pip install \"spectral-model-explainer[plotting]\" (or pip install plotly)"
        ) from exc
    return go


def resolve_theme(theme: Optional[SMXTheme]) -> SMXTheme:
    return theme if theme is not None else DEFAULT_THEME


def resolve_title(title: Optional[Union[str, bool]], default: str) -> str:
    """``None`` → *default*; ``""``/``False`` → no title; otherwise *title*."""
    if title is None:
        return default
    if title is False:
        return ""
    return str(title)


def pop_deprecated_kwargs(
    func_name: str,
    kwargs: Dict[str, Any],
    renamed: Dict[str, str],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """Map deprecated keyword names in *kwargs* onto their new names.

    Parameters
    ----------
    func_name : str
        Name used in warning / error messages.
    kwargs : dict
        The ``**kwargs`` received by the public function.
    renamed : dict
        ``{old_name: new_name}``.
    current : dict
        ``{new_name: value}`` as received through the new parameter names.

    Returns
    -------
    dict
        *current* updated with values passed through deprecated names.
    """
    resolved = dict(current)
    for old, new in renamed.items():
        if old not in kwargs:
            continue
        value = kwargs.pop(old)
        if resolved.get(new) is not None:
            raise TypeError(f"{func_name}() got values for both '{new}' and its deprecated alias '{old}'.")
        warnings.warn(
            f"{func_name}(): '{old}' is deprecated; use '{new}' instead.",
            FutureWarning,
            stacklevel=3,
        )
        resolved[new] = value
    if kwargs:
        unexpected = ", ".join(repr(k) for k in kwargs)
        raise TypeError(f"{func_name}() got unexpected keyword argument(s): {unexpected}")
    return resolved


def write_figure(
    fig,
    output_path: Optional[PathLike],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Export *fig* to *output_path*; the format is inferred from the suffix.

    ``.html`` writes an interactive figure; ``.png``, ``.jpg``, ``.jpeg``,
    ``.webp``, ``.svg`` and ``.pdf`` write a static image through ``kaleido``.
    Parent directories are created when missing.
    """
    if output_path is None:
        return

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix != ".html" and suffix not in STATIC_SUFFIXES:
        raise ValueError(
            f"Unsupported output format '{suffix or output_path.name}'. Use '.html' for an "
            "interactive figure or one of "
            + ", ".join(f"'{s}'" for s in sorted(STATIC_SUFFIXES))
            + " for a static image."
        )
    if suffix in STATIC_SUFFIXES and importlib.util.find_spec("kaleido") is None:
        raise ImportError(
            "Static image export requires kaleido. Install it with: pip install kaleido"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".html":
        fig.write_html(str(output_path))
    else:
        fig.write_image(str(output_path), width=width, height=height)


class FigureWithData(tuple):
    """``(fig, data)`` pair returned when ``return_df=True``.

    Behaves exactly like a 2-tuple (``fig, df = plot_...(return_df=True)``);
    in Jupyter, a bare call displays the figure followed by the data instead
    of the tuple's text representation.
    """

    def __new__(cls, fig, data):
        return super().__new__(cls, (fig, data))

    def _ipython_display_(self):
        from IPython.display import display

        for item in self:
            display(item)


def finalize_figure(
    fig,
    *,
    theme: SMXTheme,
    title: Optional[Union[str, bool]],
    default_title: str,
    width: Optional[int],
    height: Optional[int],
    output_path: Optional[PathLike],
    data: Any = None,
    return_df: bool = False,
    **layout: Any,
):
    """Apply the theme/title/size, export, and build the return value."""
    title_text = resolve_title(title, default_title)
    fig.update_layout(
        **theme.plotly_layout(
            title=dict(text=title_text),
            width=width,
            height=height,
            **layout,
        )
    )
    write_figure(fig, output_path, width=width, height=height)
    if return_df:
        return FigureWithData(fig, data)
    return fig


def legend_below() -> dict:
    """Horizontal legend centred at the bottom of the figure.

    Anchored to the figure container (not the plot area), so it never
    collides with rotated tick labels or the x-axis title.
    """
    return dict(orientation="h", yref="container", yanchor="bottom", y=0.02, xanchor="center", x=0.5)


def boundary_line(theme: SMXTheme) -> dict:
    return dict(
        color=theme.zone_boundary_color,
        width=theme.zone_boundary_width,
        dash=theme.zone_boundary_dash,
    )


# ── Data helpers ───────────────────────────────────────────────────────────────

def normalize_cuts(spectral_cuts: Iterable) -> pd.DataFrame:
    """Parse spectral cuts into a ``name / zone / start / end`` DataFrame.

    Accepts every format supported by :func:`smx.extract_spectral_zones`:
    ``(start, end)``, ``(name, start, end)``, ``(name, start, end, group)``
    and the equivalent dicts.  ``zone`` is the key used by the zone
    dictionaries and LRC tables — the group name for grouped cuts, otherwise
    the cut name.  Rows are sorted by ``start``.
    """
    rows = []
    for cut in spectral_cuts:
        group = None
        if isinstance(cut, dict):
            start, end = cut.get("start"), cut.get("end")
            name = cut.get("name", f"{start}-{end}")
            group = cut.get("group")
        elif isinstance(cut, (list, tuple)) and len(cut) == 2:
            start, end = cut
            name = f"{start}-{end}"
        elif isinstance(cut, (list, tuple)) and len(cut) == 3:
            name, start, end = cut
        elif isinstance(cut, (list, tuple)) and len(cut) == 4:
            name, start, end, group = cut
        else:
            raise ValueError(
                "Each spectral cut must be (start, end), (name, start, end), "
                "(name, start, end, group) or a dict with 'start' and 'end'."
            )
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Spectral cut {cut!r} has non-numeric boundaries.") from exc
        if start > end:
            start, end = end, start
        rows.append({
            "name": str(name),
            "zone": str(group) if group is not None else str(name),
            "start": start,
            "end": end,
        })
    if not rows:
        raise ValueError("spectral_cuts must contain at least one cut.")
    return pd.DataFrame(rows).sort_values("start", kind="stable").reset_index(drop=True)


def numeric_spectrum(series: pd.Series) -> pd.Series:
    """Return *series* indexed by numeric spectral positions, sorted."""
    series = series.copy()
    series.index = pd.to_numeric(series.index.astype(str), errors="coerce")
    series = series[~series.index.isna()]
    return series.sort_index()


def stitch_zones(
    zone_dict: Dict[str, pd.DataFrame],
    zone_names: Sequence[str],
    reducer,
) -> pd.Series:
    """Reduce each zone DataFrame with *reducer* and stitch them on one axis."""
    parts: List[pd.Series] = []
    for zone in dict.fromkeys(zone_names):
        zone_df = zone_dict.get(zone)
        if zone_df is None or zone_df.empty:
            continue
        parts.append(numeric_spectrum(reducer(zone_df)))
    if not parts:
        return pd.Series(dtype=float)
    stitched = pd.concat(parts)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched.sort_index().dropna()


def class_labels(y_labels: Union[pd.Series, Sequence], n_rows: Optional[int] = None) -> pd.Series:
    """Return *y_labels* as a positional Series, checking its length."""
    labels = pd.Series(np.asarray(y_labels)).reset_index(drop=True)
    if n_rows is not None and len(labels) != n_rows:
        raise ValueError(
            f"y_labels has {len(labels)} entries but the spectra have {n_rows} rows; "
            "they must be aligned row by row."
        )
    return labels


def unique_classes(labels: pd.Series) -> list:
    """Unique class labels, sorted when the labels are sortable."""
    classes = list(pd.unique(labels))
    try:
        return sorted(classes)
    except TypeError:
        return classes


def prepare_zone_ranking_df(zone_ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported ranking-table shapes into ``zone / score / rank``.

    Accepts either a ranking table with ``zone`` / ``score`` (and optionally
    ``rank``) columns, or an SMX LRC table with ``Zone`` /
    ``Local_Reaching_Centrality`` columns.  Several rows per zone are collapsed
    to the strongest score.  The result is sorted by descending score.
    """
    if zone_ranking_df is None or zone_ranking_df.empty:
        raise ValueError("zone_ranking_df must be a non-empty DataFrame.")

    if {"zone", "score"}.issubset(zone_ranking_df.columns):
        columns = ["zone", "score"] + (["rank"] if "rank" in zone_ranking_df.columns else [])
        ranking = zone_ranking_df[columns].copy()
    elif {"Zone", "Local_Reaching_Centrality"}.issubset(zone_ranking_df.columns):
        ranking = zone_ranking_df[["Zone", "Local_Reaching_Centrality"]].rename(
            columns={"Zone": "zone", "Local_Reaching_Centrality": "score"}
        )
    else:
        raise ValueError(
            "zone_ranking_df must contain either "
            "('zone', 'score') or ('Zone', 'Local_Reaching_Centrality') columns."
        )

    ranking = ranking[ranking["zone"].notna()].copy()
    ranking["zone"] = ranking["zone"].astype(str)
    ranking["score"] = pd.to_numeric(ranking["score"], errors="coerce")
    ranking = ranking.dropna(subset=["score"])
    if ranking.empty:
        raise ValueError("zone_ranking_df contains no numeric scores.")

    if "rank" in ranking.columns:
        ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
        ranking = ranking.groupby("zone", as_index=False, sort=False).agg(
            score=("score", "max"), rank=("rank", "min")
        )
    else:
        ranking = ranking.groupby("zone", as_index=False, sort=False)["score"].max()

    ranking = ranking.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    if "rank" not in ranking.columns or ranking["rank"].isna().any():
        ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["rank"] = ranking["rank"].astype(int)
    return ranking[["zone", "score", "rank"]]


def score_colors(
    scores: Iterable[float],
    colorscale,
    vmin: float,
    vmax: float,
) -> List[str]:
    """Sample *colorscale* for each score normalized to ``[vmin, vmax]``."""
    from plotly.colors import sample_colorscale

    span = vmax - vmin
    norms = [1.0 if span <= 0 else min(max((float(s) - vmin) / span, 0.0), 1.0) for s in scores]
    return sample_colorscale(colorscale, norms) if norms else []


def score_colorbar(
    theme: SMXTheme,
    vmin: float,
    vmax: float,
    title: str = "LRC score",
    **overrides: Any,
) -> dict:
    """Colorbar spec with ticks at the min/max score, shared by all plots."""
    tickvals = [vmin, vmax] if vmax > vmin else [vmax]
    ticktext = [f"{vmin:.3f}<br>(min)", f"{vmax:.3f}<br>(max)"] if vmax > vmin else [f"{vmax:.3f}"]
    colorbar = dict(
        title=dict(text=title, side="right"),
        thickness=theme.colorbar_thickness,
        len=theme.colorbar_len,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(size=10),
    )
    colorbar.update(overrides)
    return colorbar
