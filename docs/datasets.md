# Synthetic datasets

SMX includes utilities for generating synthetic spectral datasets. These are
useful for demos, tests, and quick experimentation.

## Generate synthetic spectra

```python
from smx import generate_synthetic_spectral_data

classes_config = [
    {
        "name": "A",
        "n_samples": 80,
        "peaks": [250, 380, 550, 700, 850],
        "amplitude_mean": 1.0,
        "width_mean": 15.0,
        "noise_std": 0.04,
    },
    {
        "name": "B",
        "n_samples": 80,
        "peaks": [50, 250, 380, 550, 850],
        "amplitude_mean": 1.2,
        "width_mean": 18.0,
        "noise_std": 0.035,
    },
]

df = generate_synthetic_spectral_data(
    classes_config=classes_config,
    n_points=500,
    x_min=1,
    x_max=1000,
    seed=0,
)
```

The resulting DataFrame has a `Class` column followed by spectral variables
named after their x-axis values.

## Peak model

The generator uses a Gaussian peak model internally via `gaussian_peak_model`.
It supports either scalar peak centers or per-peak dictionaries that override
amplitude and width.
