"""
SMX Quickstart — binary classification with a synthetic spectral dataset
========================================================================

This script walks through the complete SMX pipeline:

  1. Generate a synthetic two-class spectral dataset (no external files needed)
  2. Split calibration / test sets
  3. Mean-centre the spectra
  4. Train an SVM classifier
  5. Run the SMX explanation pipeline via SMXExplainer (single object, single call)
  6. Print the ranked spectral zones and export HTML plots

Dependencies: numpy, pandas, scikit-learn, smx
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import smx

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

# =============================================================================
# 1. Synthetic spectral dataset (config inlined from synthetic.json)
# =============================================================================
# Two-class XRF-like dataset:
#   Class A — peaks at 250, 380, 550, 700, 850 (channel units)
#   Class B — peaks at  50, 250, 380, 550, 850 (overlaps A except for F1/F5)
# Spectral axis: 500 points from 1 to 1000 (channel numbers)

CLASSES_CONFIG = [
    {
        "name": "A",
        "n_samples": 156,
        "peaks": [250, 380, 550, 700, 850],
        "amplitude_mean": 1.0,
        "amplitude_std": 0.3,
        "width_mean": 15.0,
        "width_std": 2.0,
        "noise_std": 0.04,
    },
    {
        "name": "B",
        "n_samples": 146,
        "peaks": [50, 250, 380, 550, 850],
        "amplitude_mean": 1.4,
        "amplitude_std": 0.5,
        "width_mean": 15.0,
        "width_std": 1.8,
        "noise_std": 0.035,
    },
]

df = smx.generate_synthetic_spectral_data(
    classes_config=CLASSES_CONFIG,
    n_points=500,
    x_min=1,
    x_max=1000,
    seed=0,
)

X = df.drop(columns=["Class"])
y = df["Class"]

# =============================================================================
# 2. Calibration / test split (stratified)
# =============================================================================
X_cal, X_test, y_cal, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_cal = X_cal.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_cal = y_cal.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# =============================================================================
# 3. Mean-centre preprocessing (fit on calibration set only)
# =============================================================================
X_mean = X_cal.mean()
X_cal_prep = X_cal - X_mean
X_test_prep = X_test - X_mean

# =============================================================================
# 4. SVM classifier
# =============================================================================
svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
svm.fit(X_cal_prep, y_cal)

acc = (svm.predict(X_test_prep) == y_test).mean()
print(f"SVM test accuracy: {acc:.2%}")

# Continuous output used by SMX (probability of class A)
class_order = list(svm.classes_)
class_a_idx = class_order.index("A")
y_pred_cal = pd.Series(svm.predict_proba(X_cal_prep)[:, class_a_idx])

# =============================================================================
# 5. SMX pipeline (single object, single fit call)
# =============================================================================
# Spectral cuts mirror the zones defined in synthetic.json:
#   F1 = exclusive Class B peak region   (  1 – 100)
#   F2 = shared peak region              (200 – 300)
#   F3 = shared peak region              (330 – 430)
#   F4 = shared peak region              (500 – 600)
#   F5 = exclusive Class A peak region   (660 – 750)
#   F6 = shared peak region              (815 – 890)
# Background zones are included to cover the full axis.
spectral_cuts = [
    ("F1",          1.0,   100.0),
    ("background1", 100.0, 200.0),
    ("F2",          200.0, 300.0),
    ("background2", 300.0, 330.0),
    ("F3",          330.0, 430.0),
    ("background3", 430.0, 500.0),
    ("F4",          500.0, 600.0),
    ("background4", 600.0, 660.0),
    ("F5",          660.0, 750.0),
    ("background5", 750.0, 815.0),
    ("F6",          815.0, 890.0),
    ("background6", 890.0, 1000.0),
]

explainer = smx.Explainer(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.50, 0.75],
    seeds=[0, 1, 2, 3],
    n_bags=10,
    n_samples_fraction=0.8,
    min_samples_fraction=0.2,
    metric="perturbation",
    estimator=svm,
    perturbation_mode="median",
    perturbation_metric="probability_shift"
)

explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)

# =============================================================================
# 6. Results
# =============================================================================
print("\n" + "=" * 60)
print("Top predicates by Local Reaching Centrality")
print("=" * 60)

top = (
    explainer.lrc_natural_[explainer.lrc_natural_["Zone"].notna()]
    .drop_duplicates(subset="Zone")
    .sort_values("Local_Reaching_Centrality", ascending=False)
    [["Zone", "Operator", "Threshold_Natural", "Local_Reaching_Centrality"]]
    .reset_index(drop=True)
)
print(top.to_string(index=False))

# =============================================================================
# 7. Export HTML threshold-spectrum plots (one per zone)
# =============================================================================
from pathlib import Path
from smx.plotting import plot_threshold_spectrum

output_dir = Path("smx_quickstart_plots")
output_dir.mkdir(exist_ok=True)

# Use the top-ranked predicate for each zone
top_per_zone = (
    explainer.lrc_natural_[explainer.lrc_natural_["Zone"].notna()]
    .sort_values("Local_Reaching_Centrality", ascending=False)
    .drop_duplicates(subset="Zone")
)

print("\nExporting HTML plots…")
for _, row in top_per_zone.iterrows():
    zone_name = row["Zone"]
    row_index = explainer.lrc_natural_.index[
        explainer.lrc_natural_["Node"] == row["Node"]
    ].tolist()[0]
    html_path = output_dir / f"threshold_{zone_name.replace(' ', '_')}.html"
    plot_threshold_spectrum(
        lrc_natural_df=explainer.lrc_natural_,
        row_index=row_index,
        spectral_zones_original=explainer.zones_natural_,
        pca_info_dict_original=explainer.pca_info_natural_,
        y_labels=y_cal,
        output_path=html_path,
    )
    print(f"  Saved: {html_path}")


