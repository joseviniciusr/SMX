# SMX Experiment Refactoring Plan

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Summary of Findings](#2-summary-of-findings)
3. [Decisions Log](#3-decisions-log)
4. [Phase 1: Extract Dataset Metadata into JSON Configs](#phase-1-extract-dataset-metadata-into-json-configs)
5. [Phase 2: Update smx Modules](#phase-2-update-smx-modules)
6. [Phase 3: Build the Generic `run_experiment.py` Script](#phase-3-build-the-generic-run_experimentpy-script)
7. [Phase 4: Create the Interactive Notebook](#phase-4-create-the-interactive-notebook)
8. [Phase 5: Migration & Validation](#phase-5-migration--validation)
9. [File Structure After Refactoring](#file-structure-after-refactoring)
10. [Risk Log & Edge Cases](#risk-log--edge-cases)
11. [Future Improvements](#future-improvements)
12. [Implementation Order Summary](#implementation-order-summary)

---

## 1. Current State Analysis

### 1.1 Directory Layout

```
experiments/
  PLS/
    bank_notes/ forage/ milk/ soil/ soil_types/ synthetic/ tomato/
    sweet_pepper/ soil_types_unir/ soil_vnir/          ← PLS-only datasets
  MLP/
    bank_notes/ forage/ milk/ soil/ soil_types/ synthetic/ tomato/
  SVM/
    bank_notes/ forage/ milk/ soil/ soil_types/ synthetic/ tomato/
```

Each dataset directory contains:
- `pca_aggregator_{dataset}.ipynb` — the main experiment notebook
- `shap_{dataset}.csv` — pre-computed SHAP values (loaded, not computed inline)
- `shap_{dataset}.py` — standalone SHAP computation script (only exists for tomato in MLP/SVM/PLS, and sweet_pepper in PLS)
- `feature_importance.csv`, `rbo_rank.csv`, `lrc_cov_natural.csv`, `lrc_pert_natural.csv` — output artifacts

### 1.2 smx Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `explaining.py` | 2466 | 20 public functions: spectral zone extraction, PCA aggregation, predicates, bagging, covariance/perturbation metrics, graph building, LRC, permutation importance, threshold mapping, plotting |
| `modeling.py` | 753 | `pls_optimized()`, `svm_optimized()`, `mlp_optimized()`, `vip_scores()`, `explained_variance_from_scores()` |
| `preprocessings.py` | ~150 | `poisson()`, `mc()`, `auto_scaling()`, `pareto()`, `msc()`, `modified_poisson()` |
| `debugging.py` | ~90 | `vip_scores_per_zone()`, `regression_coefficients_per_zone()`, `shap_per_zone()`, `rbo_rank_comparison()` |
| `synthetic.py` | ~160 | `generate_synthetic_spectral_data()` |

### 1.3 Config State

- Only `real_datasets/xrf/soil.json` exists — the `.py` script already reads from it.
- All other datasets have their metadata **hardcoded inside notebooks**.

---

## 2. Summary of Findings

### 2.1 What Is Identical Across ALL Notebooks

| Aspect | Value |
|--------|-------|
| Class labels | Binary `'A'` / `'B'` |
| Classification threshold | `>= 0.5` → `'A'`, else `'B'` |
| Random seeds | `[0, 1, 2, 3]` |
| Train/test split | Kennard-Stone, `test_size=0.30` |
| Aggregation method | PCA (`aggregate_spectral_zones_pca`) |
| Quantile predicates | `[0.2, 0.4, 0.6, 0.8]` |
| Bagging parameters | `n_bags=10`, `n_samples_per_bag=training*0.8`, `min_samples_per_predicate=training*0.2`, `replace=False`, `sample_bagging=True`, `predicate_bagging=False` |
| Covariance metric params | `metric='covariance'`, `threshold=0.01`, `n_neighbors=5` |
| Permutation params | `n_repeats=10`, `random_state=42` |

### 2.2 What Varies by Dataset

| Parameter | Example Values |
|-----------|---------------|
| CSV file path | `soil.csv`, `milk.csv`, etc. (all under `real_datasets/xrf/`) |
| Spectral column range | `'1.32':'13.1'` (soil), `'2.66':'22.62'` (milk), etc. |
| Spectral cuts | Completely different per dataset (10-22 zones each) |
| LVmax (PLS only) | 1 (synthetic), 2 (tomato, soil_types), 3 (forage), 4 (soil, milk, bank_notes) |
| Preprocessing | `poisson+mc` (all real datasets) vs `mc` only (synthetic) |
| Threshold plot `row_index` | Varies (0, 62, etc.) — hardcoded per experiment, notebook-only feature |

### 2.3 What Varies by Model

Each model type has fundamentally different prediction semantics that drive different parameter choices throughout the pipeline.

**Why PLS uses `aim='regression'` for perturbation**: PLS-DA is technically a PLS *regression* model (PLSRegression from sklearn) trained with numeric class labels (0/1). Its `predict()` output is continuous (e.g., 0.37, 0.82), then thresholded at 0.5 for classification. Since PLS lacks `predict_proba()`, perturbation must use regression-mode metrics (`mean_abs_diff`) that compare the continuous `predict()` outputs directly.

**Why MLP/SVM use `aim='classification'` for perturbation**: Both MLP (MLPClassifier) and SVM (SVC with `probability=True`) are native classifiers with `predict_proba()`. Perturbation uses `probability_shift` which measures the change in class probabilities after perturbing a zone — a more meaningful classification-specific metric.

**Similarly for permutation importance**: PLS notebooks use `estimator.predict()` (continuous output), while MLP/SVM notebooks use `estimator.predict_proba()[:, 1]` (class probability). These are intentionally different approaches matching each model's prediction semantics, not inconsistencies.

| Aspect | PLS | MLP | SVM |
|--------|-----|-----|-----|
| **Training function** | `pls_optimized()` | `mlp_optimized()` | `svm_optimized()` |
| **Model hyperparams** | `LVmax` (dataset-specific) | `hidden_layer_sizes`, `activation`, `learning_rate`, `max_iter`, `random_state` (stored in JSON) | `kernel` (stored in JSON) |
| **Return tuple indices** | `[3]=model`, `[4]=vip`, `[5]=predres_numeric` | `[3]=model`, `[4]=calres_proba` | `[3]=model`, `[4]=calres_proba`, `[6]=calres_decision` |
| **y_pred extraction** | `plsda_results[5].iloc[:, -1]` (continuous 0-1) | `mlp_model[4]['MLP'].values` (probabilities) | `svm_model[4]['SVC'].values` (probabilities) |
| **Perturbation aim** | `'regression'` — uses `predict()` | `'classification'` — uses `predict_proba()` | `'classification'` — uses `predict_proba()` |
| **Perturbation metric** | `'mean_abs_diff'` | `'probability_shift'` | `'probability_shift'` |
| **Permutation prediction fn** | `estimator.predict()` | `estimator.predict_proba()[:, 1]` | `estimator.predict_proba()[:, 1]` |
| **Model-specific importance** | VIP scores + Regression Coefficients | *(none)* | P-vector (support vectors × dual coefficients) |
| **Feature importance cols** | `VIP_Score, Reg_Coefficient, Shap, Permutation, LRC_pert, LRC_cov` | `Shap, Permutation, LRC_pert, LRC_cov` | `SVM_pvector, Shap, Permutation, LRC_pert, LRC_cov` |
| **SHAP approach** | `KernelExplainer(model.predict, ...)` | `KernelExplainer(model.predict_proba(x)[:, 1], ...)` | `KernelExplainer(model.predict_proba(x)[:, 1], ...)` |
| **Debugging section** | VIP zone, Reg Coef zone, SHAP zone | SHAP zone only | P-vector zone, SHAP zone |

### 2.4 Dual `aim` Semantics

The word "aim" is used in two distinct contexts:

1. **Training aim** (`aim` passed to `*_optimized()`) — always `'classification'` for all current experiments. Controls whether the model is built as a classifier or regressor.

2. **Perturbation aim** (`aim` passed to `calculate_predicate_perturbation()`) — controls which prediction function and metric the perturbation analysis uses. This is model-dependent:
   - PLS: `'regression'` (because PLS only has `predict()`, no `predict_proba()`)
   - MLP/SVM: `'classification'` (because they have `predict_proba()`)

The current `soil.json` has `"aim": "regression"` which refers to the perturbation aim. The JSON config should store both:
- `"training_aim": "classification"` — how the model is trained
- `"perturbation_aim"` — stored **per model** in the config, not globally (see Phase 1)

### 2.5 Other Notes

1. **SHAP is never computed inline**: It's always pre-computed via standalone `shap_*.py` scripts and loaded from CSV. These scripts exist for tomato (all models) and sweet_pepper (PLS). Other datasets have `shap_*.csv` artifacts but the scripts that generated them are not in the repo.
2. **Copy-paste artifacts**: Some MLP notebooks have comments saying "predictions from SVM" — cosmetic issue.
3. **Permutation importance not using module function**: All three model notebooks implement permutation importance inline rather than calling `permutation_importance_per_zone()` from `explaining.py`. The `.py` script (PLS/soil) does use the module function. The module function needs to be extended with a `scoring_fn` parameter to support all models (see Phase 2).

---

## 3. Decisions Log

Decisions made based on clarifications:

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| D1 | **Permutation importance** | Each model uses its native prediction function: PLS → `predict()`, MLP/SVM → `predict_proba()[:, 1]`. The module function `permutation_importance_per_zone()` must be extended with a `scoring_fn` parameter. | Different models have different prediction semantics; this is intentional, not an inconsistency. |
| D2 | **Dual `aim` in JSON** | Store both `"training_aim"` and model-specific perturbation config. Training aim is dataset-level; perturbation aim/metric is model-level (hardcoded per model type in the dispatch table, not per dataset). | PLS always uses `aim='regression'`+`metric='mean_abs_diff'` for perturbation; MLP/SVM always use `aim='classification'`+`metric='probability_shift'`. This is model-intrinsic, not dataset-dependent. |
| D3 | **SHAP computation** | Always load from pre-computed CSV. If CSV is missing, skip SHAP gracefully (warning, not error). SHAP remains a separate offline step. | SHAP via KernelExplainer is too slow to include in the experiment pipeline. |
| D4 | **Dataset-model compatibility** | Each dataset JSON includes a `"compatible_models"` list (e.g., `["pls", "mlp", "svm"]`). The `--model all` flag skips incompatible combinations silently. | Some datasets (sweet_pepper, soil_types_unir, soil_vnir) are PLS-only. |
| D5 | **Threshold plot `row_index`** | Notebook-only feature. The CLI script does not generate threshold plots. The notebook lets the user pick a row interactively. | This is an exploratory visualization, not a standard artifact. |
| D6 | **MLP/SVM hyperparameters** | Stored in each dataset's JSON config under `"model_params"` per model. No CLI flags. | Allows per-dataset tuning while keeping the CLI simple. |
| D7 | **Model return refactoring** | Deferred — not needed for this phase. The generic script will use model-specific extractors (index-based) in the dispatch table. | Normalize return types is a future improvement; current approach works via the dispatch table. |
| D8 | **Output directory** | Preserve current structure: `experiments/{MODEL}/{dataset}/`. | Maintains backward compatibility with existing artifacts. |

| D9 | **soil_types_unir / soil_vnir** | Both get JSON configs and are included in the generic framework. They are PLS-only (`compatible_models: ["pls"]`). | User confirmed. These use external CSVs (VNIR_databases/) and Savitzky-Golay preprocessing — needs special handling in `load_data()` and `preprocess()`. |
| D10 | **Debugging flag scope** | `--debugging` controls **all items not strictly needed for the SMX technique**: model-specific importance (VIP, RegCoef, P-vector), SHAP per zone, permutation importance per zone, RBO rank comparison, and the combined feature importance table. The core SMX pipeline (train → PCA zones → predicates → covariance/perturbation → LRC → natural mapping) runs without `--debugging`. | These extras are for maintainers comparing SMX against other techniques, not for external users. |

---

## Phase 1: Extract Dataset Metadata into JSON Configs

### 1.1 Create JSON config for every dataset

Create one JSON file per dataset in `real_datasets/xrf/`:

```
real_datasets/xrf/
  bank_notes.json
  forage.json
  milk.json
  soil.json            ← already exists, needs restructuring
  soil_types.json
  soil_types_unir.json ← NEW (PLS-only, external CSV, Savitzky-Golay)
  soil_vnir.json       ← NEW (PLS-only, external CSV, Savitzky-Golay)
  sweet_pepper.json
  synthetic.json
  tomato.json
```

### 1.2 JSON Schema

```jsonc
{
  // ─── Dataset Identity ───
  "name": "soil",
  "csv_file": "soil.csv",              // relative to real_datasets/xrf/; null for synthetic
  "separator": ";",
  "class_column": "Class",
  "spectral_range": ["1.32", "13.1"],  // [start_col, end_col] for .loc slicing

  // ─── Preprocessing ───
  "preprocessing": "poisson",           // "poisson" | "mc" | "auto_scaling" | "pareto" | "savgol"
  "preprocessing_mc": true,             // whether to mean-center after scaling
  // For Savitzky-Golay (soil_types_unir, soil_vnir):
  // "preprocessing": "savgol",
  // "savgol_params": {"window_length": 11, "polyorder": 3, "deriv": 0},

  // ─── Training ───
  "training_aim": "classification",     // "classification" | "regression"

  // ─── Experiment Parameters ───
  "random_seeds": [0, 1, 2, 3],
  "test_size": 0.30,

  // ─── Spectral Cuts ───
  "spectral_cuts": [
    ["Al", 1.33, 1.63],
    ["Si", 1.63, 1.86]
    // ...
  ],

  // ─── Model Compatibility & Parameters ───
  "compatible_models": ["pls", "mlp", "svm"],
  "model_params": {
    "pls": {
      "LVmax": 4,
      "cv": 10
    },
    "mlp": {
      "hidden_layer_sizes": [64, 32],
      "activation": "tanh",
      "learning_rate": "adaptive",
      "max_iter": 10,
      "random_state": 1
    },
    "svm": {
      "kernel": "rbf"
    }
  }
}
```

For synthetic, add extra fields:

```jsonc
{
  "name": "synthetic",
  "csv_file": null,
  "is_synthetic": true,
  "synthetic_config": {
    "classes": [
      {"nome": "A", "n_amostras": 156, "picos": [250, 380, 550, 700, 850],
       "amp_media": 1.0, "amp_std": 0.3, "larg_media": 15.0, "larg_std": 2.0, "ruido_std": 0.04},
      {"nome": "B", "n_amostras": 146, "picos": [50, 250, 380, 550, 850],
       "amp_media": 1.4, "amp_std": 0.5, "larg_media": 15.0, "larg_std": 1.8, "ruido_std": 0.035}
    ],
    "n_pontos": 500,
    "x_min": 1,
    "x_max": 1000,
    "seed": 0
  },
  "preprocessing": "mc",
  "preprocessing_mc": false,
  // ...
}
```

### 1.3 Tasks

| # | Task | Details |
|---|------|---------|
| 1.1 | Create `bank_notes.json` | Extract from PLS/bank_notes notebook: spectral_cuts (15 zones), `spectral_range=["2.74","22.71"]`, model_params with LVmax=4 |
| 1.2 | Create `forage.json` | 22 zones, `spectral_range=["1.4","20.81"]`, LVmax=3 |
| 1.3 | Create `milk.json` | 10 zones, `spectral_range=["2.66","22.62"]`, LVmax=4 |
| 1.4 | Restructure `soil.json` | Current JSON only has `aim`, `LVmax`, `random_seeds`, `spectral_cuts`. Needs full schema: add `csv_file`, `spectral_range`, `preprocessing`, `training_aim`, `model_params`, `compatible_models` |
| 1.5 | Create `soil_types.json` | 21 zones, `spectral_range=["1.32","13.1"]`, LVmax=2 |
| 1.6 | Create `sweet_pepper.json` | 21 zones, `spectral_range=["2.12","23.08"]`, LVmax=1, `compatible_models=["pls"]` (PLS-only) |
| 1.7 | Create `synthetic.json` | Special: includes `synthetic_config`, `preprocessing="mc"`, LVmax=1 |
| 1.8 | Create `tomato.json` | 17 zones, `spectral_range=["2.12","23.08"]`, LVmax=2 |
| 1.9 | Create `soil_types_unir.json` | 50-unit float zones via `np.arange(1351.54, 2150.47, 50)`, `spectral_range=["1351.542978","2150.472084"]`, LVmax=2, `preprocessing="savgol"`, `compatible_models=["pls"]`. CSV at external path `VNIR_databases/soil_types/uNIR.csv` |
| 1.10 | Create `soil_vnir.json` | 100-unit integer zones via `range(400,2500,100)`, `spectral_range=["400","2498"]`, LVmax=3, `preprocessing="savgol"`, `compatible_models=["pls"]`. CSV at external path `VNIR_databases/soil/plsda/soil_vnir.csv` |

### 1.4 Config Loader Utility

Create `smx/config.py`:

```python
import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'real_datasets' / 'xrf'

def load_dataset_config(dataset_name: str) -> dict:
    """Load and validate a dataset JSON config by name."""
    config_path = CONFIGS_DIR / f'{dataset_name}.json'
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for dataset '{dataset_name}' at {config_path}")
    with open(config_path) as f:
        return json.load(f)

def list_available_datasets() -> list[str]:
    """List all datasets that have JSON configs."""
    return sorted(p.stem for p in CONFIGS_DIR.glob('*.json'))

def get_compatible_datasets(model_name: str) -> list[str]:
    """List datasets compatible with a given model."""
    result = []
    for ds in list_available_datasets():
        config = load_dataset_config(ds)
        if model_name in config.get('compatible_models', []):
            result.append(ds)
    return result
```

---

## Phase 2: Update smx Modules

### 2.1 Permutation Importance — Add `scoring_fn`

**Context**: `permutation_importance_per_zone()` currently uses `estimator.predict()` for all models. This is correct for PLS (which outputs continuous regression values). MLP/SVM notebooks manually override this with `predict_proba()[:, 1]` to measure probability shift. This is intentional — each model type has a different prediction semantic.

**Change**: Add an optional `scoring_fn` parameter so the generic script can pass the correct function per model:

```python
def permutation_importance_per_zone(
    estimator, X, spectral_cuts,
    n_repeats=10, random_state=42,
    scoring_fn=None  # if None, uses estimator.predict()
):
```

The dispatch table will supply:
- PLS: `scoring_fn=None` → uses `estimator.predict()` (continuous 0–1 output)
- MLP: `scoring_fn=lambda X: model.predict_proba(X)[:, 1]`
- SVM: `scoring_fn=lambda X: model.predict_proba(X)[:, 1]`

This is backward-compatible — existing code passes nothing and gets the same behavior.

### 2.2 SVM P-vector — Add to `debugging.py`

**Context**: SVM notebooks compute P-vector importance inline. There is no module-level function for it, unlike VIP (PLS) which already has `vip_scores_per_zone()`.

**Change**: Add `svm_pvector_per_zone(svm_model, X_columns, spectral_cuts)` to `debugging.py`. Extracted from SVM notebooks.

### 2.3 Tasks

| # | Task | Details |
|---|------|---------|
| 2.1 | Add `scoring_fn` param to `permutation_importance_per_zone()` in `explaining.py` | Backward-compatible default (`None` = `predict`) |
| 2.2 | Add `svm_pvector_per_zone()` to `debugging.py` | New function, extracted from SVM notebooks |
| 2.3 | Verify `shap_per_zone()` works with CSV path for all models | Already takes CSV path — should be fine |

---

## Phase 3: Build the Generic `run_experiment.py` Script

### 3.1 CLI Interface

Location: `experiments/run_experiment.py`

```bash
python experiments/run_experiment.py \
  --dataset soil \
  --model pls \
  --method covariance \
  --debugging
```

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--dataset` | `{name}` or `all` | *(required)* | Dataset name (must have JSON config) |
| `--model` | `pls`, `mlp`, `svm`, `all` | *(required)* | Model type |
| `--method` | `covariance`, `perturbation`, `all` | `all` | Which LRC method(s) to run |
| `--debugging` | flag (store_true) | `False` | Enable non-core extras: permutation importance, model-specific importance (VIP/RegCoef/P-vector), SHAP zone, RBO, feature importance table |

When `--model all` is used with a dataset, only the models listed in the dataset's `compatible_models` will run. Incompatible model requests are skipped with a warning.

### 3.2 Script Architecture

```python
# experiments/run_experiment.py

def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load (or generate) dataset and split via Kennard-Stone.
    Handles: CSV in real_datasets/xrf/, external paths (VNIR), synthetic generation."""

def preprocess(config: dict, Xcal, Xpred) -> tuple:
    """Apply preprocessing based on config (poisson/mc/savgol/etc)."""

def train_model(model_name: str, config: dict, Xcal_prep, ycal, Xpred_prep, ypred) -> tuple:
    """Dispatch to pls_optimized / mlp_optimized / svm_optimized.
    Returns the raw tuple from the underlying function."""

def extract_y_continuous(model_name: str, result: tuple) -> pd.Series:
    """Extract continuous predictions from model result tuple.
    PLS: result[6].iloc[:, -1]  (predres_numeric)
    MLP: result[4]['MLP']       (calres_proba)
    SVM: result[4]['SVC']       (calres_proba)
    """

def run_covariance_pipeline(config, zone_scores_df, y_pred, predicates_quantiles,
                            pca_info_dict, output_dir):
    """Covariance metric → graphs → LRC → aggregation → natural mapping."""

def run_perturbation_pipeline(config, model_name, model, zone_scores_df, y_pred,
                               predicates_quantiles, pca_info_dict, Xcalclass_prep, output_dir):
    """Perturbation metric → graphs → LRC → aggregation → natural mapping.
    Uses model-specific aim/metric from dispatch table."""

def run_debugging(model_name, result, config, Xcalclass_prep, spectral_cuts, output_dir):
    """Model-specific importance (VIP, RegCoef, P-vector) + SHAP zone (loaded from CSV)."""

def build_feature_importance_table(model_name, debugging_results, permutation_df,
                                    lrc_cov_df, lrc_pert_df) -> pd.DataFrame:
    """Assemble feature_importance DataFrame with model-appropriate columns."""

def run_single_experiment(dataset: str, model: str, method: str, debugging: bool):
    """Orchestrate a single (dataset, model) experiment."""

def main():
    """Parse CLI args, expand 'all', run experiments."""
```

### 3.3 Detailed Pipeline Flow

```
 1. Load config from JSON
 2. Validate model ∈ config["compatible_models"]
 3. Load / generate data
 4. Kennard-Stone split (per class)
 5. Concatenate classes
 6. Preprocess (poisson/mc)
 7. Train model → raw tuple
 8. Extract y_pred_continuous (model-specific index)
 9. Extract spectral zones (preprocessed)
10. PCA aggregate zones
11. Generate predicates by quantiles
12. Create predicate info dict
13. FOR EACH method (covariance / perturbation):
    a. FOR EACH seed:
       i.   Bagging predicates
       ii.  Class assignment (>= 0.5 → A/B)
       iii. Calculate metric (covariance or perturbation)
    b. Build graphs per seed
    c. Calculate LRC per seed
    d. Aggregate LRC across seeds
    e. Extract spectral zones (original scale)
    f. PCA aggregate zones (original scale)
    g. Map thresholds to natural scale
14. IF debugging:
    a. Permutation importance per zone (with model-specific scoring_fn)
    b. Model-specific importance (VIP/RegCoef for PLS, P-vector for SVM)
    c. SHAP zone (loaded from pre-computed CSV, graceful skip if missing)
    d. Build feature importance table (combines LRC + permutation + model-specific + SHAP)
    e. RBO rank comparison
15. Save all artifacts to output directory
```

> **Note**: `plot_threshold_spectrum()` is **not** called by the CLI script. Threshold plots are notebook-only (see Phase 4).

### 3.4 Model-Specific Dispatch Table

```python
MODEL_CONFIG = {
    'pls': {
        'train_fn': pls_optimized,
        'train_kwargs': lambda cfg: {
            'LVmax': cfg['model_params']['pls']['LVmax'],
            'aim': 'classification',
            'cv': cfg['model_params']['pls'].get('cv', 10),
        },
        'y_pred_extractor': lambda result: result[6].iloc[:, -1],  # predres_numeric
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'regression',
        'perturbation_metric': 'mean_abs_diff',
        'permutation_scoring_fn': None,  # uses predict()
        'importance_methods': ['vip', 'reg_coef'],
        'importance_columns': ['VIP_Score', 'Reg_Coefficient'],
    },
    'mlp': {
        'train_fn': mlp_optimized,
        'train_kwargs': lambda cfg: {
            'aim': 'classification',
            **cfg['model_params']['mlp'],
        },
        'y_pred_extractor': lambda result: result[4]['MLP'],  # calres_proba
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'classification',
        'perturbation_metric': 'probability_shift',
        'permutation_scoring_fn': lambda model: lambda X: model.predict_proba(X)[:, 1],
        'importance_methods': [],
        'importance_columns': [],
    },
    'svm': {
        'train_fn': svm_optimized,
        'train_kwargs': lambda cfg: {
            'aim': 'classification',
            **cfg['model_params']['svm'],
        },
        'y_pred_extractor': lambda result: result[4]['SVC'],  # calres_proba
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'classification',
        'perturbation_metric': 'probability_shift',
        'permutation_scoring_fn': lambda model: lambda X: model.predict_proba(X)[:, 1],
        'importance_methods': ['pvector'],
        'importance_columns': ['SVM_pvector'],
    },
}
```

### 3.5 Output Directory

Artifacts saved to `experiments/{MODEL}/{dataset}/` (preserving current structure):

```
experiments/PLS/soil/
  feature_importance.csv
  rbo_rank.csv
  lrc_cov_natural.csv
  lrc_pert_natural.csv
  shap_soil.csv      ← pre-computed, loaded by debugging
```

### 3.6 Tasks

| # | Task | Details |
|---|------|---------|
| 3.1 | Create `experiments/run_experiment.py` skeleton | argparse + main structure |
| 3.2 | Implement `load_data()` | CSV loading + synthetic generation dispatch |
| 3.3 | Implement `preprocess()` | Dispatch to poisson/mc/etc based on config |
| 3.4 | Implement `train_model()` | Dispatch to PLS/MLP/SVM with config-based params |
| 3.5 | Implement `extract_y_continuous()` | Model-specific tuple index extraction |
| 3.6 | Implement `run_covariance_pipeline()` | Port from soil.py, parameterize |
| 3.7 | Implement `run_perturbation_pipeline()` | Port from soil.py, parameterize with model-specific aim/metric |
| 3.8 | Implement `run_debugging()` | Permutation importance + VIP (PLS) + RegCoef (PLS) + P-vector (SVM) + SHAP zone from CSV + RBO + feature importance table |
| 3.9 | Implement `build_feature_importance_table()` | Model-specific column assembly (called within run_debugging) |
| 3.10 | Implement `run_single_experiment()` | Orchestrator |
| 3.11 | Implement `main()` + argparse | CLI handling, `all` expansion, compatibility check |

---

## Phase 4: Create the Interactive Notebook

### 4.1 Location

`experiments/run_experiment.ipynb`

### 4.2 Cell Layout

| Cell # | Type | Content |
|--------|------|---------|
| 1 | Markdown | Title + instructions |
| 2 | Code | Imports + path setup |
| 3 | Code | **Parameter selectors** — dropdown widgets for dataset, model, method, debugging |
| 4 | Code | Load config + display dataset metadata summary |
| 5 | Code | Load data + Kennard-Stone split + display shapes |
| 6 | Code | Preprocess + display preprocessed data head |
| 7 | Code | Train model + display results table |
| 8 | Code | PCA aggregation + predicates + display zone info |
| 9 | Code | Run covariance pipeline (if selected) + display LRC table |
| 10 | Code | Run perturbation pipeline (if selected) + display LRC table |
| 11 | Code | Permutation importance + display |
| 12 | Code | Model-specific importance (if debugging) + display |
| 13 | Code | Feature importance summary table |
| 14 | Code | RBO rank comparison (if debugging) |
| 15 | Code | **Threshold spectrum plots** (inline, interactive Plotly) |
| 16 | Code | Save all artifacts |

### 4.3 Widget Implementation

```python
import ipywidgets as widgets
from IPython.display import display
from smx.config import list_available_datasets, get_compatible_datasets

dataset_selector = widgets.Dropdown(
    options=list_available_datasets(),
    description='Dataset:',
)

# Model options update dynamically based on selected dataset's compatible_models
model_selector = widgets.Dropdown(
    options=['pls', 'mlp', 'svm'],
    description='Model:',
)

method_selector = widgets.Dropdown(
    options=['all', 'covariance', 'perturbation'],
    description='Method:',
)

debugging_toggle = widgets.Checkbox(
    value=False,
    description='Debugging',
)

def update_models(change):
    """Filter model options based on dataset's compatible_models."""
    config = load_dataset_config(change['new'])
    model_selector.options = config.get('compatible_models', ['pls', 'mlp', 'svm'])

dataset_selector.observe(update_models, names='value')
display(dataset_selector, model_selector, method_selector, debugging_toggle)
```

### 4.4 Key Differences from CLI Script

| Feature | CLI Script | Notebook |
|---------|-----------|----------|
| Threshold spectrum plots | NOT included | Inline Plotly via `plot_threshold_spectrum()` |
| Intermediate DataFrames | Saved only | Displayed inline via `display()` |
| Parameters | `argparse` | Dropdown widgets |
| Output | File artifacts only | Visual + file artifacts |

### 4.5 Shared Code

The notebook imports core functions from `run_experiment.py`:

```python
from run_experiment import (
    load_data, preprocess, train_model, extract_y_continuous,
    run_covariance_pipeline, run_perturbation_pipeline,
    run_debugging, build_feature_importance_table
)
```

This avoids code duplication. The notebook adds visualization on top.

### 4.6 Tasks

| # | Task | Details |
|---|------|---------|
| 4.1 | Create `experiments/run_experiment.ipynb` | Notebook file with widget cells |
| 4.2 | Implement dynamic widget parameter selection | Dataset → model filtering |
| 4.3 | Wire widget values to core functions | Import from `run_experiment.py` |
| 4.4 | Add inline Plotly threshold spectrum plots | `plot_threshold_spectrum(...).show()` |
| 4.5 | Add inline display of intermediate DataFrames | `display(df)` at key steps |
| 4.6 | Test with multiple dataset/model combinations | Verify all combos work |

---

## Phase 5: Migration & Validation

### 5.1 Validation Strategy

For each existing (model, dataset) combination that has output artifacts:

1. Run `python experiments/run_experiment.py --dataset {ds} --model {model} --method all --debugging`
2. Compare generated `feature_importance.csv` against existing one
3. Compare `lrc_cov_natural.csv` and `lrc_pert_natural.csv` against existing
4. Compare `rbo_rank.csv` against existing

If outputs match (within floating-point tolerance via `np.allclose`), the migration is correct.

### 5.2 Datasets × Models Matrix

Use each dataset's `compatible_models` field to determine valid combinations:

| Dataset | PLS | MLP | SVM | Notes |
|---------|-----|-----|-----|-------|
| bank_notes | ✅ | ✅ | ✅ | |
| forage | ✅ | ✅ | ✅ | |
| milk | ✅ | ✅ | ✅ | |
| soil | ✅ | ✅ | ✅ | Reference dataset (has .py script) |
| soil_types | ✅ | ✅ | ✅ | |
| synthetic | ✅ | ✅ | ✅ | mc preprocessing, no CSV |
| tomato | ✅ | ✅ | ✅ | Has shap_*.py scripts |
| sweet_pepper | ✅ | ❌ | ❌ | `compatible_models: ["pls"]` |
| soil_types_unir | ✅ | ❌ | ❌ | PLS-only, external CSV, Savitzky-Golay preprocessing |
| soil_vnir | ✅ | ❌ | ❌ | PLS-only, external CSV, Savitzky-Golay preprocessing |

### 5.3 Tasks

| # | Task | Details |
|---|------|---------|
| 5.1 | Validate PLS/soil (baseline) | Compare against existing script output |
| 5.2 | Validate PLS across 3+ datasets | bank_notes, milk, tomato |
| 5.3 | Validate MLP/soil | Compare against existing notebook output |
| 5.4 | Validate SVM/soil | Compare against existing notebook output |
| 5.5 | Cross-validate 2-3 MLP and SVM datasets | Edge cases |
| 5.6 | Archive old notebooks | Move to `experiments/_archive/` or create git tag |

---

## File Structure After Refactoring

```
SMX/
├── requirements.txt
├── plan.md
├── smx/
│   ├── __init__.py
│   ├── config.py              ← NEW: config loader + dataset registry
│   ├── debugging.py           ← MODIFIED: add svm_pvector_per_zone()
│   ├── explaining.py          ← MODIFIED: scoring_fn in permutation_importance_per_zone()
│   ├── modeling.py            ← UNCHANGED (no ModelResult refactoring)
│   ├── preprocessings.py      ← UNCHANGED
│   └── synthetic.py           ← UNCHANGED
├── real_datasets/
│   └── xrf/
│       ├── bank_notes.csv
│       ├── bank_notes.json    ← NEW
│       ├── forage.csv
│       ├── forage.json        ← NEW
│       ├── milk.csv
│       ├── milk.json          ← NEW
│       ├── soil.csv
│       ├── soil.json          ← RESTRUCTURED (add full schema fields)
│       ├── soil_types.csv
│       ├── soil_types.json    ← NEW
│       ├── sweet_pepper.csv
│       ├── soil_types_unir.json ← NEW (PLS-only, external CSV)
│       ├── soil_vnir.json     ← NEW (PLS-only, external CSV)
│       ├── sweet_pepper.csv
│       ├── sweet_pepper.json  ← NEW
│       ├── synthetic.json     ← NEW (no CSV)
│       ├── tomato.csv
│       └── tomato.json        ← NEW
├── experiments/
│   ├── run_experiment.py      ← NEW: generic CLI script
│   ├── run_experiment.ipynb   ← NEW: interactive notebook
│   ├── PLS/                   ← KEPT: existing artifacts preserved
│   ├── MLP/
│   └── SVM/
```

---

## Risk Log & Edge Cases

| Risk | Impact | Mitigation |
|------|--------|------------|
| SHAP computation is extremely slow | Can't include in standard run | SHAP loaded from pre-computed CSV; graceful skip if missing |
| Floating-point differences across runs | Validation diffs | Use `np.allclose` for comparison |
| MLP `max_iter=10` may not converge | Bad results | Existing behavior; not our change to make |
| Pre-computed SHAP CSV missing for a model/dataset | Debugging incomplete | Log warning, skip SHAP in importance table |
| Model not in `compatible_models` | Wasted run | Validate upfront, skip with warning |

---

## Future Improvements

These items were considered but deferred. They can be tackled independently after the core refactoring is complete:

1. **ModelResult dataclass**: Normalize the return interfaces of `pls_optimized()` (7-tuple), `mlp_optimized()` (6-tuple), and `svm_optimized()` (8-tuple) into a common `ModelResult` dataclass/dict with named fields. This would eliminate fragile index-based extraction but requires updating all existing notebooks and scripts.

2. **SHAP computation integration**: Instead of loading SHAP from pre-computed CSVs, add a `--shap` flag to the CLI that computes SHAP values on-the-fly. This is slow (~hours for some datasets) so it should remain opt-in.

3. **JSON Schema validation**: Create a `real_datasets/xrf/schema.json` for formal validation of dataset config files.

---

## Implementation Order Summary

```
Phase 1 (JSON configs)           → Independent, no code changes needed
Phase 2 (Update smx modules)     → Independent of Phase 1, small scope
Phase 3 (run_experiment.py)      → Depends on Phases 1-2
Phase 4 (Notebook)               → Depends on Phase 3
Phase 5 (Validation)             → Depends on Phases 3-4
```

Phases 1 and 2 can be done in parallel. Phase 4 depends on Phase 3 being mostly complete.
