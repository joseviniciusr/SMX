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
np.random.seed(SEED)

# =============================================================================
# 1. Synthetic spectral dataset
# =============================================================================
# Wavelength axis from 1.0 to 10.0 keV in 0.1 steps (91 channels)
wavelengths = np.round(np.arange(1.0, 10.1, 0.1), 1)
N_PER_CLASS = 60


def _gaussian(x, center, amplitude, width):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def _make_spectra(n, peak_centers, rng):
    spectra = []
    for _ in range(n):
        spectrum = rng.normal(0, 0.03, len(wavelengths))  # baseline noise
        for center, amp, width in peak_centers:
            amp_jitter = rng.normal(amp, amp * 0.10)
            width_jitter = abs(rng.normal(width, width * 0.10))
            spectrum += _gaussian(wavelengths, center, amp_jitter, width_jitter)
        spectra.append(spectrum)
    return np.array(spectra)


rng = np.random.default_rng(SEED)

# Class A: strong peak at 2.5, moderate peak at 5.5
spectra_A = _make_spectra(N_PER_CLASS, [(2.5, 1.0, 0.3), (5.5, 0.5, 0.4)], rng)

# Class B: strong peak at 3.5, strong peak at 7.5
spectra_B = _make_spectra(N_PER_CLASS, [(3.5, 1.0, 0.3), (7.5, 0.8, 0.4)], rng)

X = pd.DataFrame(
    np.vstack([spectra_A, spectra_B]),
    columns=wavelengths,
)
y = pd.Series(["A"] * N_PER_CLASS + ["B"] * N_PER_CLASS)

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
spectral_cuts = [
    ("Low",  1.0, 4.0),   # covers Class A peak at 2.5 and Class B peak at 3.5
    ("Mid",  4.0, 6.5),   # covers Class A peak at 5.5
    ("High", 6.5, 10.0),  # covers Class B peak at 7.5
]

explainer = smx.SMXExplainer(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.50, 0.75],
    seeds=[0, 1, 2, 3],
    n_bags=10,
    n_samples_fraction=0.8,
    min_samples_fraction=0.2,
    metric="perturbation",
    estimator=svm,
    perturbation_mode="median",
    perturbation_metric="probability_shift",
    normalize_by_zone_size=True,
    zone_size_exponent=1.0,
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

