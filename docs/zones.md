# Spectral zones

SMX works on user-defined spectral zones. Each zone is a slice of the spectral
axis, defined by a start and end value. Zones can be named, grouped, and
merged into composite regions.

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
