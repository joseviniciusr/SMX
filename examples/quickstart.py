"""
SMX Quickstart — binary classification with a synthetic spectral dataset
========================================================================

This script walks through the complete SMX pipeline:

  1. Generate a synthetic two-class spectral dataset (no external files needed)
  2. Split calibration / test sets
  3. Mean-centre the spectra
  4. Train an SVM classifier
  5. Run the SMX explanation pipeline (zone extraction → predicate generation
     → bagging → covariance metric → graph → LRC)
  6. Map thresholds back to the natural (unpreprocessed) scale
  7. Print the ranked spectral zones

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
# 5. SMX pipeline
# =============================================================================

# ------------------------------------------------------------------
# 5a. Define spectral zones and extract them
# ------------------------------------------------------------------
spectral_cuts = [
    ("Low",  1.0, 4.0),   # covers Class A peak at 2.5 and Class B peak at 3.5
    ("Mid",  4.0, 6.5),   # covers Class A peak at 5.5
    ("High", 6.5, 10.0),  # covers Class B peak at 7.5
]

zones_prep = smx.extract_spectral_zones(X_cal_prep, spectral_cuts)

# ------------------------------------------------------------------
# 5b. Aggregate each zone to a single score (PCA, PC1)
# ------------------------------------------------------------------
aggregator = smx.ZoneAggregator(method="pca")
zone_scores = aggregator.fit_transform(zones_prep)   # DataFrame: samples × zones
pca_info = aggregator.pca_info_

# ------------------------------------------------------------------
# 5c. Generate binary predicates from quantile thresholds
# ------------------------------------------------------------------
gen = smx.PredicateGenerator(quantiles=[0.25, 0.50, 0.75])
gen.fit(zone_scores)
predicates_df = gen.predicates_df_

print(f"\nGenerated {len(predicates_df)} predicates across {len(spectral_cuts)} zones.")

# ------------------------------------------------------------------
# 5d. Run bagging + covariance metric + graph + LRC across seeds
# ------------------------------------------------------------------
n_cal = len(zone_scores)
SEEDS = [0, 1, 2, 3]

lrc_by_seed = {}

for seed in SEEDS:
    print(f"\n── Seed {seed} ──────────────────────────────────────────────")

    # Bagging
    bagger = smx.PredicateBagger(
        n_bags=10,
        n_samples_per_bag=int(n_cal * 0.8),
        min_samples_per_predicate=int(n_cal * 0.2),
        replace=False,
        sample_bagging=True,
        predicate_bagging=False,
        random_seed=seed,
    )
    bags = bagger.run(zone_scores, y_pred_cal, predicates_df)

    # Tag each sample with a discrete class prediction
    for bag_name, pred_dict in bags.items():
        for rule, df_info in pred_dict.items():
            df_info["Class_Predicted"] = np.where(
                df_info["Predicted_Y"] >= 0.5, "A", "B"
            )

    # Covariance metric
    metric = smx.CovarianceMetric(metric="covariance", threshold=0.01, n_neighbors=5)
    rankings = metric.compute(bags)

    # Build predicate graph (weight edges by PC1 explained variance)
    builder = smx.PredicateGraphBuilder(
        random_state=seed,
        show_details=False,
        var_exp=True,
        pca_info_dict=pca_info,
    )
    graph = builder.build(bags, rankings, metric_column="Covariance")

    # Local Reaching Centrality
    predicate_nodes = [
        n for n, attr in graph.nodes(data=True) if attr.get("node_type") == "predicate"
    ]
    if not predicate_nodes:
        print(f"  Seed {seed} produced an empty graph — skipping.")
        continue

    lrc_df = smx.compute_lrc(graph, predicates_df)
    lrc_df["Seed"] = seed
    lrc_by_seed[seed] = lrc_df

# ------------------------------------------------------------------
# 5e. Aggregate LRC across seeds
# ------------------------------------------------------------------
valid_seeds = list(lrc_by_seed.keys())
lrc_summed, lrc_unique = smx.aggregate_lrc_across_seeds(lrc_by_seed, valid_seeds)

# ------------------------------------------------------------------
# 5f. Map thresholds to natural (unpreprocessed) scale
# ------------------------------------------------------------------
zones_natural = smx.extract_spectral_zones(X_cal, spectral_cuts)
aggregator_natural = smx.ZoneAggregator(method="pca")
zone_scores_natural = aggregator_natural.fit_transform(zones_natural)

lrc_natural = smx.map_thresholds_to_natural(
    lrc_df=lrc_summed,
    zone_sums_preprocessed=zone_scores,
    zone_sums_natural=zone_scores_natural,
)

# =============================================================================
# 6. Results
# =============================================================================
print("\n" + "=" * 60)
print("Top predicates by Local Reaching Centrality")
print("=" * 60)

top = (
    lrc_natural[lrc_natural["Zone"].notna()]
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
    lrc_natural[lrc_natural["Zone"].notna()]
    .sort_values("Local_Reaching_Centrality", ascending=False)
    .drop_duplicates(subset="Zone")
)

pca_info_natural = aggregator_natural.pca_info_

print("\nExporting HTML plots…")
for _, row in top_per_zone.iterrows():
    zone_name = row["Zone"]
    row_index = lrc_natural.index[lrc_natural["Node"] == row["Node"]].tolist()[0]
    html_path = output_dir / f"threshold_{zone_name.replace(' ', '_')}.html"
    plot_threshold_spectrum(
        lrc_natural_df=lrc_natural,
        row_index=row_index,
        spectral_zones_original=zones_natural,
        pca_info_dict_original=pca_info_natural,
        y_labels=y_cal,
        output_path=html_path,
    )
    print(f"  Saved: {html_path}")
