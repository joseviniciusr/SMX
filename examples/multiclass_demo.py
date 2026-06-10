"""
End-to-end demonstration of SMX with a 3-class spectral problem.

Run with: python examples/multiclass_demo.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from smx import SMX, generate_synthetic_spectral_data

# ── Generate data ────────────────────────────────────────────────────────────
classes_config = [
    {"name": "Class_0", "n_samples": 80, "peaks": [200, 500],
     "amplitude_mean": 1.0, "width_mean": 15.0},
    {"name": "Class_1", "n_samples": 80, "peaks": [350, 650],
     "amplitude_mean": 1.2, "width_mean": 12.0},
    {"name": "Class_2", "n_samples": 80, "peaks": [150, 450, 750],
     "amplitude_mean": 0.9, "width_mean": 18.0},
]

df = generate_synthetic_spectral_data(
    classes_config=classes_config,
    n_points=300,
    x_min=0,
    x_max=1000,
    seed=42,
)

y = df["Class"]
X = df.drop(columns=["Class"])

# ── Train/calibration split ───────────────────────────────────────────────────
X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_prep = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_cal_prep = pd.DataFrame(
    scaler.transform(X_cal), columns=X_cal.columns, index=X_cal.index
)

# ── Train model ───────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_prep, y_train)

# ── Obtain class labels for calibration set ───────────────────────────────────
y_class_labels = pd.Series(model.predict(X_cal_prep), name="class")

# ── Define spectral zones ─────────────────────────────────────────────────────
spectral_cuts = [
    ("Zone_A", 0.0, 200.0),
    ("Zone_B", 200.0, 400.0),
    ("Zone_C", 400.0, 600.0),
    ("Zone_D", 600.0, 800.0),
    ("Zone_E", 800.0, 1000.0),
]

# ── Fit SMX ───────────────────────────────────────────────────────────────────
smx = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.5, 0.75],
    n_repetitions=4,
    n_bags=10,
    estimator=model,
    perturbation_metric="probability_shift",
    var_exp=True,
)

smx.fit(X_cal_prep, y_class_labels, X_cal_natural=X_cal)

# ── Inspect results ────────────────────────────────────────────────────────────
print("\n=== Zone ranking (multi-class, 3 classes) ===")
print(smx.lrc_unique_)
print(f"\nValid seeds: {smx.valid_seeds_}")
