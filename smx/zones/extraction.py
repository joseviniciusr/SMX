"""Extract spectral zones from a DataFrame based on numeric column boundaries."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def extract_spectral_zones(
    Xcal: pd.DataFrame,
    cuts: List[Union[Tuple, dict]],
) -> Dict[str, pd.DataFrame]:
    """
    Extract spectral zones from a DataFrame based on specified cuts.

    Parameters
    ----------
    Xcal : pd.DataFrame
        DataFrame with spectral data.  Columns must be numeric (or convertible
        to numeric) values representing wavelengths / energies.
    cuts : list of tuples/lists or dicts
        Each item defines a spectral zone to extract.

        * ``(start, end)`` — zone boundaries; name defaults to ``"start-end"``
        * ``(name, start, end)`` — named zone
        * ``(name, start, end, group)`` — named zone assigned to a group
        * ``{'name': str, 'start': float, 'end': float}`` — dict form
        * ``{'name': str, 'start': float, 'end': float, 'group': str}`` — dict
          form with grouping

        When multiple cuts share the same ``group`` value their column subsets
        are concatenated into a single zone keyed by the group name.  Cuts
        *without* a group are extracted individually under their own name, as
        before.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are zone names (or group names) and values are
        DataFrames with the extracted spectral data (same row index as *Xcal*).

    Examples
    --------
    >>> zones = extract_spectral_zones(X, [('Ca ka', 3.6, 3.7), ('Fe ka', 6.3, 6.5)])
    >>> zones['Ca ka'].shape
    (n_samples, n_cols_in_Ca_ka_zone)

    Group background regions into a single zone:

    >>> cuts = [
    ...     ('background 1', 1.0, 101.0, 'background'),
    ...     ('Feature 1',  101.0, 193.3),
    ...     ('background 2', 193.3, 255.4, 'background'),
    ...     ('Feature 2',  255.4, 341.6),
    ... ]
    >>> zones = extract_spectral_zones(X, cuts)
    >>> 'background' in zones   # True — merged from background 1 & 2
    True
    >>> 'background 1' in zones  # False — individual cuts absorbed into group
    False
    """
    col_nums = pd.to_numeric(Xcal.columns.astype(str), errors="coerce")
    zones: Dict[str, pd.DataFrame] = {}
    # Accumulate column subsets for grouped cuts; order of insertion preserved
    grouped: Dict[str, List[pd.DataFrame]] = {}

    for cut in cuts:
        group: Optional[str] = None

        if isinstance(cut, dict):
            name = cut.get("name", f"{cut.get('start')}-{cut.get('end')}")
            start = cut.get("start")
            end = cut.get("end")
            group = cut.get("group", None)
        elif isinstance(cut, (list, tuple)):
            if len(cut) == 2:
                start, end = cut
                name = f"{start}-{end}"
            elif len(cut) == 3:
                name, start, end = cut
            elif len(cut) == 4:
                name, start, end, group = cut
            else:
                raise ValueError("Cuts in tuple/list format must have 2, 3, or 4 elements.")
        else:
            raise ValueError("Each cut must be a dict or a tuple/list.")

        try:
            s = float(start)
            e = float(end)
        except Exception:
            raise ValueError("start and end must be numeric values (int/float or convertible strings).")

        if s > e:
            s, e = e, s

        mask = (~np.isnan(col_nums)) & (col_nums >= s) & (col_nums <= e)
        zone_df = Xcal.loc[:, mask]

        if group is not None:
            grouped.setdefault(group, []).append(zone_df)
        else:
            zones[name] = zone_df

    # Merge grouped zones by concatenating columns (preserving spectral order)
    for group_name, zone_dfs in grouped.items():
        zones[group_name] = pd.concat(zone_dfs, axis=1)

    return zones
