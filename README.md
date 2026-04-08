<p align="center">
  <img src="SMX_logo.png" alt="SMX logo" width="220">
</p>

# SMX
Spectral Model eXplainer: a XAI tool for spectral-based machine learning models

## Toy Example

This example assumes your spectra have already been loaded into memory.

- `X` is a `pandas.DataFrame` with shape `(n_samples, n_wavelengths)`
- columns are the spectral axis values
- `y` contains the class labels

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from smx import SMX

# Example inputs already available in memory:
# X -> spectral matrix (samples x variables)
# y -> class labels such as "A" / "B"

X_cal, X_test, y_cal, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Simple preprocessing: mean-centre using calibration data only
X_mean = X_cal.mean()
X_cal_prep = X_cal - X_mean
X_test_prep = X_test - X_mean

# Train any sklearn-compatible model
model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
model.fit(X_cal_prep, y_cal)

# SMX expects a continuous prediction for the calibration set
class_order = list(model.classes_)
target_idx = class_order.index("A")
y_pred_cal = pd.Series(model.predict_proba(X_cal_prep)[:, target_idx])

# Define spectral zones
spectral_cuts = [
    ("low_band", 400.0, 550.0),
    ("mid_band", 550.0, 700.0),
    ("high_band", 700.0, 900.0),
]

# Fit SMX
explainer = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.50, 0.75],
    seeds=[0, 1, 2, 3],
    n_bags=10,
    metric="perturbation",
    estimator=model,
    perturbation_mode="median",
    perturbation_metric="probability_shift",
)

explainer.fit(
    X_cal_prep=X_cal_prep,
    y_pred_cal=y_pred_cal,
    X_cal_natural=X_cal,
)

# Ranked predicates with thresholds mapped back to the natural scale
print(
    explainer.lrc_natural_[
        ["Node", "Zone", "Operator", "Threshold_Natural", "Local_Reaching_Centrality"]
    ].head()
)
```

After fitting, the most useful outputs are:

- `explainer.lrc_natural_`: ranked predicates with thresholds in the original spectral scale
- `explainer.lrc_summed_unique_`: one top-ranked predicate per zone
- `explainer.zones_natural_`: extracted natural-scale spectral zones
- `explainer.pca_info_natural_`: PCA information used for interpretation / reconstruction

## Plotting helpers

SMX includes Plotly-based visualization helpers for common explanation views.

### Zone ranking over spectrum

After fitting an `SMX` explainer, you can export the ranked spectral zones over
the reference spectrum:

```python
from smx import SMX

explainer = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.2, 0.4, 0.6, 0.8],
    seeds=[0, 1, 2, 3],
    metric="perturbation",
    estimator=model,
)
explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal_raw)

explainer.plot_zone_ranking_over_spectrum(
    "zone_ranking.html",
    ranking="unique",
)
```

You can also call the standalone plotting function:

```python
from smx import plot_zone_ranking_over_spectrum

plot_zone_ranking_over_spectrum(
    zone_ranking_df=explainer.lrc_summed_unique_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=explainer.zones_natural_,
    output_path="zone_ranking.html",
)
```

Install plotting support with:

```bash
pip install -e .[plotting]
```
