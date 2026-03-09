"""Extract spectral zones from a DataFrame based on numeric column boundaries."""

from typing import Dict, List, Tuple, Union

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
        * ``{'name': str, 'start': float, 'end': float}`` — dict form

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary where keys are zone names and values are DataFrames with the
        extracted spectral data (same row index as *Xcal*).

    Examples
    --------
    >>> zones = extract_spectral_zones(X, [('Ca ka', 3.6, 3.7), ('Fe ka', 6.3, 6.5)])
    >>> zones['Ca ka'].shape
    (n_samples, n_cols_in_Ca_ka_zone)
    """
    col_nums = pd.to_numeric(Xcal.columns.astype(str), errors="coerce")
    zones: Dict[str, pd.DataFrame] = {}

    for cut in cuts:
        if isinstance(cut, dict):
            name = cut.get("name", f"{cut.get('start')}-{cut.get('end')}")
            start = cut.get("start")
            end = cut.get("end")
        elif isinstance(cut, (list, tuple)):
            if len(cut) == 2:
                start, end = cut
                name = f"{start}-{end}"
            elif len(cut) == 3:
                name, start, end = cut
            else:
                raise ValueError("Cuts in tuple/list format must have 2 or 3 elements.")
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
        zones[name] = Xcal.loc[:, mask]

    return zones
