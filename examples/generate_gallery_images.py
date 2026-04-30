"""Generate static PNG images used in smx/plotting/gallery.md."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from smx import (
    SMX,
    generate_synthetic_spectral_data,
    plot_all_thresholds_overlay,
    plot_lrc_bar,
    plot_predicate_heatmap,
    plot_threshold_spectrum,
    plot_zone_ranking_over_spectrum,
    plot_zone_scores,
)

SEED = 42
ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASSES_CONFIG = [
    {
        "name": "A",
        "n_samples": 116,
        "peaks": [
            {"center": 150, "amplitude_mean": 2.5, "amplitude_std": 0.5, "width_mean": 15.0, "width_std": 2.0},
            {"center": 300, "amplitude_mean": 1.8, "amplitude_std": 0.3, "width_mean": 15.0, "width_std": 2.0},
            {"center": 500, "amplitude_mean": 0.5, "amplitude_std": 0.3, "width_mean": 15.0, "width_std": 2.0},
        ],
        "noise_std": 0.08,
    },
    {
        "name": "B",
        "n_samples": 126,
        "peaks": [
            {"center": 150, "amplitude_mean": 3.3, "amplitude_std": 0.3, "width_mean": 17.0, "width_std": 2.0},
            {"center": 300, "amplitude_mean": 0.8, "amplitude_std": 0.3, "width_mean": 14.0, "width_std": 1.5},
            {"center": 500, "amplitude_mean": 0.45, "amplitude_std": 0.3, "width_mean": 15.0, "width_std": 2.0},
        ],
        "noise_std": 0.1,
    },
]

spectral_cuts = [
    ("background 1", 1.0, 101.0),
    ("Feature 1", 101.0, 193.3),
    ("background 2", 193.3, 255.42),
    ("Feature 2", 255.42, 341.57),
    ("background 3", 341.57, 460.00),
    ("Feature 3", 460.756, 539.90),
    ("background 4", 539.90, 600.0),
]

df = generate_synthetic_spectral_data(
    classes_config=CLASSES_CONFIG, n_points=300, x_min=1, x_max=600, seed=SEED
)
X = df.drop(columns=["Class"])
y = df["Class"]

X_cal, X_test, y_cal, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_cal = X_cal.reset_index(drop=True)
y_cal = y_cal.reset_index(drop=True)

X_mean = X_cal.mean()
X_cal_prep = X_cal - X_mean

svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
svm.fit(X_cal_prep, y_cal)
class_a_idx = list(svm.classes_).index("A")
y_pred_cal = pd.Series(svm.predict_proba(X_cal_prep)[:, class_a_idx])

explainer = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.2, 0.4, 0.6, 0.8],
    n_repetitions=4,
    n_bags=10,
    n_samples_fraction=0.8,
    metric="perturbation",
    estimator=svm,
    perturbation_metric="probability_shift",
)
explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)

CLASS_COLORS = {"A": "#e41a1c", "B": "#377eb8"}

W, H = 1200, 480  # standard gallery dimensions (2.5 : 1)

# ── 1. Zone ranking over spectrum ──────────────────────────────────────────────
print("Generating zone_ranking_over_spectrum.png …")
plot_zone_ranking_over_spectrum(
    zone_ranking_df=explainer.lrc_natural_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=explainer.zones_natural_,
    output_path=ASSETS / "zone_ranking_over_spectrum.png",
    title="SMX zone ranking over spectrum",
    spectrum_name="Mean calibration spectrum",
    class_spectra={"A": X_cal[y_cal == "A"], "B": X_cal[y_cal == "B"]},
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'zone_ranking_over_spectrum.png'}")

# ── 2. Threshold spectrum (top-ranked zone) ────────────────────────────────────
top_per_zone = (
    explainer.lrc_natural_[explainer.lrc_natural_["Zone"].notna()]
    .sort_values("Local_Reaching_Centrality", ascending=False)
    .drop_duplicates(subset=["Zone"])
)

top_row = top_per_zone.iloc[0]
row_index = explainer.lrc_natural_.index[
    explainer.lrc_natural_["Node"] == top_row["Node"]
].tolist()[0]

print("Generating threshold_spectrum.png …")
plot_threshold_spectrum(
    lrc_natural_df=explainer.lrc_natural_,
    row_index=row_index,
    spectral_zones_original=explainer.zones_natural_,
    pca_info_dict_original=explainer.pca_info_natural_,
    y_labels=y_cal,
    output_path=ASSETS / "threshold_spectrum.png",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'threshold_spectrum.png'}")

# ── 3. LRC Bar Chart ───────────────────────────────────────────────────────────
print("Generating lrc_bar.png …")
plot_lrc_bar(
    zone_ranking_df=explainer.lrc_natural_,
    output_path=ASSETS / "lrc_bar.png",
    title="LRC Score by Spectral Zone",
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'lrc_bar.png'}")

# ── 4. Predicate Heatmap ───────────────────────────────────────────────────────
print("Generating predicate_heatmap.png …")
plot_predicate_heatmap(
    lrc_natural_df=explainer.lrc_natural_,
    output_path=ASSETS / "predicate_heatmap.png",
    title="Predicate LRC Heatmap",
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'predicate_heatmap.png'}")

# ── 5. Zone PC1 Score Violin ───────────────────────────────────────────────────
print("Generating zone_scores.png …")
plot_zone_scores(
    zones=explainer.zones_natural_,
    y_labels=y_cal,
    output_path=ASSETS / "zone_scores.png",
    title="PC1 Scores by Spectral Zone and Class",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'zone_scores.png'}")

# ── 6. All-Zone Threshold Overlay ──────────────────────────────────────────────
print("Generating all_thresholds_overlay.png …")
plot_all_thresholds_overlay(
    lrc_natural_df=explainer.lrc_natural_,
    zones_natural=explainer.zones_natural_,
    pca_info_natural=explainer.pca_info_natural_,
    y_labels=y_cal,
    spectral_cuts=spectral_cuts,
    output_path=ASSETS / "all_thresholds_overlay.png",
    title="All-Zone Threshold Overlay",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)
print(f"  Saved: {ASSETS / 'all_thresholds_overlay.png'}")
print("Done.")
