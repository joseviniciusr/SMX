"""
SMX Quickstart — binary classification with a synthetic spectral dataset
========================================================================

This script walks through the complete SMX pipeline:

  1. Generate a synthetic two-class spectral dataset (no external files needed)
  2. Split calibration / test sets
  3. Mean-centre the spectra
  4. Train an SVM classifier
  5. Run the SMX explanation pipeline via SMX (single object, single call)
  6. Print the ranked spectral zones and export HTML plots

Dependencies: numpy, pandas, scikit-learn, smx
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from smx import SMX, generate_synthetic_spectral_data, plot_zone_ranking_over_spectrum
from smx.graph.interpretation import reconstruct_threshold_to_spectrum

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "The quickstart plotting examples require plotly. "
        "Install it with: pip install -e .[plotting] "
        "or: pip install plotly"
    ) from exc

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

df = generate_synthetic_spectral_data(
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

smx = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.50, 0.75],
    n_repetitions=4,
    n_bags=10,
    metric="perturbation",
    estimator=svm,
    perturbation_mode="median",
    perturbation_metric="probability_shift"
)

smx.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)

# =============================================================================
# 6. Results
# =============================================================================
print("\n" + "=" * 60)
print("Top predicates by Local Reaching Centrality")
print("=" * 60)

top = (
    smx.lrc_natural_[smx.lrc_natural_["Zone"].notna()]
    .drop_duplicates(subset="Zone")
    .sort_values("Local_Reaching_Centrality", ascending=False)
    [["Zone", "Operator", "Threshold_Natural", "Local_Reaching_Centrality"]]
    .reset_index(drop=True)
)
print(top.to_string(index=False))

# =============================================================================
# 7. Export HTML zone-ranking and threshold-spectrum plots
# =============================================================================
from pathlib import Path
from smx.plotting import plot_threshold_spectrum

output_dir = Path("smx_quickstart_plots")
output_dir.mkdir(exist_ok=True)

zone_ranking_path = output_dir / "zone_ranking_over_spectrum.html"
plot_zone_ranking_over_spectrum(
    zone_ranking_df=smx.lrc_natural_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=smx.zones_natural_,
    output_path=zone_ranking_path,
    title="SMX zone ranking over spectrum",
    spectrum_name="Mean calibration spectrum",
)
print(f"\nSaved zone-ranking plot: {zone_ranking_path}")

# Use the top-ranked predicate for each zone
top_per_zone = (
    smx.lrc_natural_[smx.lrc_natural_["Zone"].notna()]
    .sort_values("Local_Reaching_Centrality", ascending=False)
    .drop_duplicates(subset="Zone")
)

print("\nExporting threshold-spectrum HTML plots…")
for _, row in top_per_zone.iterrows():
    zone_name = row["Zone"]
    row_index = smx.lrc_natural_.index[
        smx.lrc_natural_["Node"] == row["Node"]
    ].tolist()[0]
    html_path = output_dir / f"threshold_{zone_name.replace(' ', '_')}.html"
    plot_threshold_spectrum(
        lrc_natural_df=smx.lrc_natural_,
        row_index=row_index,
        spectral_zones_original=smx.zones_natural_,
        pca_info_dict_original=smx.pca_info_natural_,
        y_labels=y_cal,
        output_path=html_path,
    )
    print(f"  Saved: {html_path}")

# Also export interactive Plotly threshold-spectrum overlays (matching notebook)
def plot_zone_and_save(lrc_natural_df, row_index, spectral_zones_original,
                       pca_info_dict_original, y_labels, output_path):
    zone_name = lrc_natural_df.iloc[row_index]["Zone"]
    threshold_score = float(lrc_natural_df.iloc[row_index]["Threshold_Natural"])

    threshold_spectrum = reconstruct_threshold_to_spectrum(
        threshold_value=threshold_score,
        zone_name=zone_name,
        pca_info_dict=pca_info_dict_original,
    )

    zone_df = spectral_zones_original[zone_name]
    x_values = pd.to_numeric(zone_df.columns, errors="coerce")

    fig = go.Figure()
    CLASS_COLORS = {"A": "gold", "B": "blue"}
    seen_classes = set()
    for idx, r in zone_df.iterrows():
        class_label = y_labels.iloc[idx] if idx < len(y_labels) else "Unknown"
        show_legend = class_label not in seen_classes
        seen_classes.add(class_label)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=r.values,
                mode="lines",
                line=dict(color=CLASS_COLORS.get(class_label, "rgba(128,128,128,0.3)"), width=0.5),
                name=f"Class {class_label}",
                legendgroup=class_label,
                showlegend=show_legend,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=threshold_spectrum.values,
            mode="lines",
            line=dict(color="red", width=4, dash="dash"),
            name=f"Threshold Spectrum ({threshold_spectrum.name})",
        )
    )

    fig.update_layout(
        title=f"Zone '{zone_name}' — Multivariate Threshold (Predicate: {lrc_natural_df.iloc[row_index].get('Node_Natural', '')})",
        xaxis_title="Variables (Artificial Units)",
        yaxis_title="Intensity (Arbitrary Units)",
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(str(output_path))

# Save Plotly versions for the top-ranked predicate per zone as well
plotly_out = Path("smx_quickstart_plots_plotly")
plotly_out.mkdir(exist_ok=True)
for _, row in top_per_zone.iterrows():
    zone_name = row["Zone"]
    row_index = smx.lrc_natural_.index[
        smx.lrc_natural_["Node"] == row["Node"]
    ].tolist()[0]
    html_path = plotly_out / f"threshold_plotly_{zone_name.replace(' ', '_')}.html"
    plot_zone_and_save(
        lrc_natural_df=smx.lrc_natural_,
        row_index=row_index,
        spectral_zones_original=smx.zones_natural_,
        pca_info_dict_original=smx.pca_info_natural_,
        y_labels=y_cal,
        output_path=html_path,
    )
    print(f"  Saved Plotly: {html_path}")
