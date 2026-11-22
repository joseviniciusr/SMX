"""Generate the static PNG images used in smx/plotting/gallery.md and docs/.

Usage::

    python examples/generate_gallery_images.py [OUTPUT_DIR ...]

Images are written to ``assets/`` and ``docs/_static/`` by default.
Static export requires ``kaleido``.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from smx import (
    SMX,
    building_spectral_zones,
    generate_synthetic_spectral_data,
    plot_all_thresholds_overlay,
    plot_lrc_bar,
    plot_predicate_heatmap,
    plot_spectrum_with_zones,
    plot_threshold_spectrum,
    plot_zone_ranking_over_spectrum,
    plot_zone_scores,
)

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRS = [Path(p) for p in sys.argv[1:]] or [ROOT / "assets", ROOT / "docs" / "_static"]
ASSETS = OUTPUT_DIRS[0]
ASSETS.mkdir(parents=True, exist_ok=True)

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
    explainer.lrc_natural_,
    spectral_cuts,
    explainer.zones_natural_,
    output_path=ASSETS / "zone_ranking_over_spectrum.png",
    spectrum_name="Mean calibration spectrum",
    class_spectra={"A": X_cal[y_cal == "A"], "B": X_cal[y_cal == "B"]},
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)

# ── 2. Threshold spectrum (top-ranked predicate) ───────────────────────────────
print("Generating threshold_spectrum.png …")
top_position = int(explainer.lrc_natural_["Local_Reaching_Centrality"].to_numpy().argmax())
plot_threshold_spectrum(
    explainer.lrc_natural_,
    top_position,
    explainer.zones_natural_,
    explainer.pca_info_natural_,
    y_cal,
    output_path=ASSETS / "threshold_spectrum.png",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)

# ── 3. LRC Bar Chart ───────────────────────────────────────────────────────────
print("Generating lrc_bar.png …")
plot_lrc_bar(explainer.lrc_natural_, output_path=ASSETS / "lrc_bar.png", width=W, height=H)

# ── 4. Predicate Heatmap ───────────────────────────────────────────────────────
print("Generating predicate_heatmap.png …")
plot_predicate_heatmap(explainer.lrc_natural_, output_path=ASSETS / "predicate_heatmap.png", width=W, height=H)

# ── 5. Zone PC1 Score Violin ───────────────────────────────────────────────────
print("Generating zone_scores.png …")
plot_zone_scores(
    explainer.zones_natural_,
    y_cal,
    output_path=ASSETS / "zone_scores.png",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)

# ── 6. All-Zone Threshold Overlay ──────────────────────────────────────────────
print("Generating all_thresholds_overlay.png …")
plot_all_thresholds_overlay(
    explainer.lrc_natural_,
    explainer.zones_natural_,
    explainer.pca_info_natural_,
    y_cal,
    spectral_cuts,
    output_path=ASSETS / "all_thresholds_overlay.png",
    class_colors=CLASS_COLORS,
    width=W,
    height=H,
)

# ── 7. Faithfulness curve ──────────────────────────────────────────────────────
print("Generating faithfulness_curve.png …")
X_test_prep = X_test.reset_index(drop=True) - X_mean
explainer.evaluate_faithfulness(X_test_prep, ranking="unique", masking_strategy="zero")
explainer.plot_faithfulness(ASSETS / "faithfulness_curve.png", width=W, height=H)

# ── 8. Automatically detected zones ────────────────────────────────────────────
print("Generating detected_zones.png …")
detected_cuts = building_spectral_zones(X_cal, prominence=0.3)
plot_spectrum_with_zones(
    X_cal,
    detected_cuts,
    output_path=ASSETS / "detected_zones.png",
    title="Mean spectrum with detected zones and backgrounds",
    width=W,
    height=H,
)

for extra in OUTPUT_DIRS[1:]:
    extra.mkdir(parents=True, exist_ok=True)
    for image in ASSETS.glob("*.png"):
        if image.name in {"method_overview.png", "SMX_logo.png", "SMX_final_logo.png"}:
            continue
        shutil.copy2(image, extra / image.name)
print("Done:", ", ".join(str(d) for d in OUTPUT_DIRS))
