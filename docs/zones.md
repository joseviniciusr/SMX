# Spectral zones

SMX works on user-defined spectral zones. Each zone is a slice of the spectral
axis, defined by a start and end value. Zones can be named, grouped, and
merged into composite regions. Although these zones are expert-defined, SMX provides tools for automatic zone detection based on local minima and maxima in the spectrum. Once defined, zones are extracted from the data and aggregated into single scores for downstream modeling and interpretation.

## Automatic zone detection with `building_spectral_zones`

`building_spectral_zones` automatically partitions the spectral axis by detecting
local minima and maxima directly from a spectrum. It returns a list of
``(name, start, end)`` cuts that can be passed to ``extract_spectral_zones`` or
used downstream in the SMX pipeline.

```python
from smx.zones.build import building_spectral_zones

# Returns list of (name, start, end) cuts
spectral_cuts = building_spectral_zones(
    spectrum,           # 1-D array, Series, or DataFrame (mean spectrum used for multi-row)
    min_window_length=7,   # window for argrelmin (local minima)
    prominence=0.3,        # minimum prominence for find_peaks (local maxima) - controls sensitivity to noise, as it is the minimum height difference between a peak and its surrounding minima
    svg_smooth=False,       # optionally apply Savitzky-Golay smoothing first
    svg_window_length=11,   # window length for Savitzky-Golay filter (must be odd and >= polyorder + 2)
    svg_polyorder=3,        # polynomial order for Savitzky-Golay filter
    ploting=True,           # interactive Plotly visualization with shaded zones
)
```

The function internally uses:

- ``scipy.signal.argrelmin`` — local minima detection
- ``scipy.signal.find_peaks`` — local maxima detection (with prominence threshold)
- ``scipy.signal.savgol_filter`` — optional smoothing (when ``svg_smooth=True``)

These are standard ``scipy`` functions. ``scipy`` is listed as a core dependency
in ``pyproject.toml`` and is installed automatically with SMX.

## Extraction

Use `extract_spectral_zones` to split a DataFrame into a dictionary of zones:

```python
from smx import extract_spectral_zones

cuts = [
    ("F1", 1.0, 100.0),
    ("background1", 100.0, 200.0, "background"),
    ("F2", 200.0, 300.0),
    ("background2", 300.0, 400.0, "background"),
]

zones = extract_spectral_zones(X_cal, cuts)
```

## Supported cut formats

- `(start, end)`
- `(name, start, end)`
- `(name, start, end, group)`
- `{name, start, end}`
- `{name, start, end, group}`

When a `group` is provided, all zones with the same group name are merged into
one composite zone. Zone boundaries are inclusive, and `start > end` is
automatically corrected.

## Aggregation

Zones are converted to a single score per sample via `ZoneAggregator`:

```python
from smx import ZoneAggregator

aggregator = ZoneAggregator(method="pca")
zone_scores = aggregator.fit_transform(zones)
```

Supported methods include `pca`, `mean`, `sum`, `median`, `max`, `min`, `std`,
`var`, and `extreme`.