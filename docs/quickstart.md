# Quickstart

This walkthrough mirrors the scripts in `examples/` and focuses on the
high-level `SMX` pipeline.

## Minimal example

```python
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from smx import SMX, generate_synthetic_spectral_data

# 1) Synthetic dataset
classes_config = [
    {"name": "A", "n_samples": 120, "peaks": [250, 380, 550, 700, 850]},
    {"name": "B", "n_samples": 120, "peaks": [50, 250, 380, 550, 850]},
]

df = generate_synthetic_spectral_data(classes_config, n_points=500, x_min=1, x_max=1000, seed=0)
X = df.drop(columns=["Class"])
y = df["Class"]

# 2) Split and preprocess
X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_mean = X_cal.mean()
X_cal_prep = X_cal - X_mean
X_test_prep = X_test - X_mean

# 3) Train a model
model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_cal_prep, y_cal)

# 4) Configure spectral zones
spectral_cuts = [
    ("F1", 1.0, 100.0),
    ("B1", 100.0, 200.0),
    ("F2", 200.0, 300.0),
    ("B2", 300.0, 330.0),
    ("F3", 330.0, 430.0),
    ("B3", 430.0, 500.0),
    ("F4", 500.0, 600.0),
    ("B4", 600.0, 660.0),
    ("F5", 660.0, 750.0),
    ("B5", 750.0, 815.0),
    ("F6", 815.0, 890.0),
    ("B6", 890.0, 1000.0),
]

# 5) Fit SMX
smx = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.5, 0.75],
    n_repetitions=4,
    n_bags=10,
    metric="perturbation",
    estimator=model,
    perturbation_mode="median",
    perturbation_metric="probability_shift",
)

# Probability of class A as continuous output
class_a_idx = list(model.classes_).index("A")
y_pred_cal = pd.Series(model.predict_proba(X_cal_prep)[:, class_a_idx])

smx.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)

# 6) Inspect the ranked zones
print(smx.lrc_summed_unique_.head())
```

## Plot a ranking over the spectrum

```python
from smx import plot_zone_ranking_over_spectrum

plot_zone_ranking_over_spectrum(
    zone_ranking_df=smx.lrc_natural_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=smx.zones_natural_,
    output_path="zone_ranking.html",
)
```

## Full script and notebook

- `examples/quickstart.py` contains the end-to-end workflow.
- `examples/quickstart.ipynb` runs the same steps in a notebook.
