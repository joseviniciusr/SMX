"""Build spectral zones directly from a single spectrum."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from smx.plotting.theme import DEFAULT_THEME, SMXTheme
from smx.plotting.zones import plot_spectrum_with_zones


def building_spectral_zones(
    spectrum: Union[np.ndarray, "pd.Series", "pd.DataFrame"],
    min_window_length: int = 7,
    prominence: float = 0.3,
    svg_smooth: bool = False,
    svg_window_length: int = 7,
    svg_polyorder: int = 3,
    ploting: bool = True,
    plotting: Optional[bool] = None,
    theme: Optional[SMXTheme] = None,
    title: Optional[str] = None,
    output_path: Optional[Union[str, "Path"]] = None,
    _show_minima: bool = False,
) -> List[Tuple[str, Union[int, float, str], Union[int, float, str]]]:
    """
    Detect local minima/maxima and build spectral cuts from a single spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray, pandas.Series, or pandas.DataFrame
        Spectrum values. If a DataFrame or a 2-D numpy array is provided,
        the mean spectrum (averaged over rows) is used before zone detection.
        If a Series or 1-D array is provided, it is used directly.
    min_window_length : int, default 7
        Window length used by argrelmin to detect local minima.
    prominence : float, default 0.3
        Minimum prominence for local maxima detection.
    svg_smooth : bool, default False
        If True, apply Savitzky-Golay smoothing before peak detection.
    svg_window_length : int, default 7
        Savitzky-Golay window length.
    svg_polyorder : int, default 3
        Savitzky-Golay polynomial order.
    ploting : bool, default True
        When True, plot the spectrum with zone backgrounds and identified peaks.
    plotting : bool, optional
        Backward-compatible alias for ``ploting``.
    theme : SMXTheme, optional
        Optional visual theme for plotting.
    title : str, optional
        Optional plot title when ``ploting=True``.
    output_path : str or Path, optional
        Optional output path for saving the plot (HTML or static image).

    Returns
    -------
    list of tuples
        Spectral cuts in the form ``(label, start, end)``.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    # Respect the optional alias to preserve compatibility with prior notebooks.
    if plotting is not None:
        ploting = plotting

    if spectrum is None:
        raise ValueError("spectrum must be a non-empty array/Series/DataFrame.")

    # Normalize spectrum input to a single 1D series and aligned index labels.
    # For DataFrames with multiple rows (multiple spectra), the mean spectrum is used.
    if pd is not None and isinstance(spectrum, pd.DataFrame):
        if spectrum.empty:
            raise ValueError("spectrum DataFrame is empty.")
        # When multiple spectra are present, compute the mean row-wise along columns.
        spectrum_series = spectrum.mean(axis=0)
    elif pd is not None and isinstance(spectrum, pd.Series):
        spectrum_series = spectrum
    else:
        spectrum_series = None

    # Resolve numeric values and index labels for both pandas and numpy inputs.
    if spectrum_series is not None:
        index_values = np.asarray(spectrum_series.index)
        values = spectrum_series.values.astype(float)
    else:
        arr = np.asarray(spectrum, dtype=float)
        if arr.size == 0:
            raise ValueError("spectrum is empty.")
        # If the array has more than one dimension, average over the first axis
        # (assumes rows are individual spectra and columns are spectral variables).
        if arr.ndim > 1:
            values = np.nanmean(arr, axis=0)
        else:
            values = arr
        index_values = np.arange(len(values))

    # Optional smoothing step to reduce noise before peak detection.
    if svg_smooth:
        from scipy.signal import savgol_filter

        try:
            values_s = savgol_filter(
                values,
                window_length=svg_window_length,
                polyorder=svg_polyorder,
                deriv=1,
            )
        except Exception:
            values_s = values.copy()
    else:
        values_s = values.copy()

    import scipy.signal as signal

    # Detect local minima and maxima from the (optionally) smoothed signal.
    index_min = signal.argrelmin(values_s, axis=0, order=min_window_length)[0]
    index_max = signal.find_peaks(values_s, prominence=prominence)[0]

    # Build spectral cuts based on extrema ordering rules.
    # Keep positional indices for the logic, convert to labels at the end.
    spectral_cuts_idx: List[Tuple[str, int, int]] = []
    sorted_mins = np.sort(index_min.astype(int))
    sorted_maxs = np.sort(index_max.astype(int))

    if len(index_values) > 0: # Ensure there are index values to define the spectrum range.
        spectrum_start_idx = 0
        spectrum_end_idx = len(index_values) - 1

        # Force sentinel minima at the spectrum endpoints to close cycles.
        sentinel_mins = np.array([spectrum_start_idx, spectrum_end_idx], dtype=int)
        if len(sorted_mins) > 0:
            sorted_mins = np.unique(np.concatenate([sorted_mins, sentinel_mins]))
        else:
            sorted_mins = np.unique(sentinel_mins)

        # Build ordered extrema sequence (endpoints are always minima).
        extrema = [(int(i), "min") for i in sorted_mins] + [(int(i), "max") for i in sorted_maxs]
        extrema.sort(key=lambda item: (item[0], 0 if item[1] == "min" else 1))

        first_min = int(sorted_mins[0]) # The first minimum is the leftmost minimum (which may be a sentinel at the start).

        # Internal segments: walk extrema sequence from the first minimum.
        start_pos = next(i for i, (idx, kind) in enumerate(extrema) if kind == "min" and idx == first_min)
        current_min = first_min
        j = start_pos + 1
        while j < len(extrema):
            idx, kind = extrema[j]
            if kind == "max":
                # Consume one or more maxima until the next minimum.
                k = j + 1
                while k < len(extrema) and extrema[k][1] == "max":
                    k += 1
                if k >= len(extrema):
                    break
                next_min = int(extrema[k][0])
                spectral_cuts_idx.append(("zone", current_min, next_min))
                current_min = next_min
                j = k + 1
            else:
                # Consecutive minima -> background until the last in the run.
                last_min_in_run = int(idx)
                k = j + 1
                while k < len(extrema) and extrema[k][1] == "min":
                    last_min_in_run = int(extrema[k][0])
                    k += 1
                spectral_cuts_idx.append(("background", current_min, last_min_in_run))
                current_min = last_min_in_run
                j = k

        # Assign sequential labels after building all cuts.
        zone_counter = 1
        background_counter = 1
        labeled_cuts: List[Tuple[str, int, int]] = []
        for label, start_idx, end_idx in spectral_cuts_idx:
            if label == "zone":
                labeled_cuts.append((f"zone{zone_counter}", start_idx, end_idx))
                zone_counter += 1
            else:
                labeled_cuts.append((f"background{background_counter}", start_idx, end_idx))
                background_counter += 1
        spectral_cuts_idx = labeled_cuts

    # Convert positional indices to labels (or numeric positions for numpy input).
    spectral_cuts: List[Tuple[str, Union[int, float, str], Union[int, float, str]]] = []
    for label, start_idx, end_idx in spectral_cuts_idx:
        start_label = float(index_values[int(start_idx)])
        end_label = float(index_values[int(end_idx)])
        spectral_cuts.append((label, start_label, end_label))

    # Optional visualization using the shared SMX plotting theme.
    if ploting:
        # When no explicit theme is provided, use the 'simple_white' Plotly template
        # instead of the SMXTheme default for building_spectral_zones.
        effective_theme = theme
        if effective_theme is None:
            effective_theme = SMXTheme(template="simple_white")
        plot_spectrum_with_zones(
            spectrum=spectrum_series if spectrum_series is not None else values,
            spectral_cuts=spectral_cuts,
            identified_peaks=index_max,
            identified_minima=index_min if _show_minima else None,
            theme=effective_theme,
            title=title,
            output_path=output_path,
        )

    return spectral_cuts