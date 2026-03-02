"""
Generic experiment runner for SMX pipeline.

Usage:
    python experiments/run_experiment.py --dataset soil --model pls --method all --debugging
    python experiments/run_experiment.py --dataset all --model all --method covariance
"""

import argparse
import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import kennard_stone as ks

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
SMX_DIR = WORKSPACE_ROOT / 'smx'
if str(SMX_DIR) not in sys.path:
    sys.path.insert(0, str(SMX_DIR))

import preprocessings as prepr
from modeling import pls_optimized, svm_optimized, mlp_optimized
import explaining as exp
import debugging as dbg
from config import load_dataset_config, list_available_datasets
from synthetic import generate_synthetic_spectral_data

# ── Model dispatch table ─────────────────────────────────────────────────────
MODEL_CONFIG = {
    'pls': {
        'train_fn': pls_optimized,
        'train_kwargs': lambda cfg: {
            'LVmax': cfg['model_params']['pls']['LVmax'],
            'aim': 'classification',
            'cv': cfg['model_params']['pls'].get('cv', 10),
        },
        'y_pred_extractor': lambda result: result[5].iloc[:, -1],
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'regression',
        'perturbation_metric': 'mean_abs_diff',
        'permutation_scoring_fn': None,
        'importance_methods': ['vip', 'reg_coef'],
        'importance_columns': ['VIP_Score', 'Reg_Coefficient'],
    },
    'mlp': {
        'train_fn': mlp_optimized,
        'train_kwargs': lambda cfg: {
            'aim': 'classification',
            **cfg['model_params']['mlp'],
        },
        'y_pred_extractor': lambda result: result[4]['MLP'],
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
        'y_pred_extractor': lambda result: result[4]['SVC'],
        'model_extractor': lambda result: result[3],
        'perturbation_aim': 'classification',
        'perturbation_metric': 'probability_shift',
        'permutation_scoring_fn': lambda model: lambda X: model.predict_proba(X)[:, 1],
        'importance_methods': ['pvector'],
        'importance_columns': ['SVM_pvector'],
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(config):
    """Load (or generate) dataset and split via Kennard-Stone.

    Returns: Xcalclass, Xpredclass, ycalclass, ypredclass, data_complete
    """
    spectral_range = config['spectral_range']
    class_col = config.get('class_column', 'Class')
    test_size = config.get('test_size', 0.30)

    if config.get('is_synthetic'):
        syn_cfg = config['synthetic_config']
        data_complete = generate_synthetic_spectral_data(
            configuracao_classes=syn_cfg['classes'],
            n_pontos=syn_cfg['n_pontos'],
            x_min=syn_cfg['x_min'],
            x_max=syn_cfg['x_max'],
            seed=syn_cfg['seed'],
        )
        # For synthetic, spectral data is all columns except 'Class'
        spectral_cols = [c for c in data_complete.columns if c != class_col]
        data = data_complete[spectral_cols]
    else:
        csv_file = config['csv_file']
        sep = config.get('separator', ';')
        if config.get('csv_file_is_external'):
            csv_path = WORKSPACE_ROOT / csv_file
        else:
            csv_path = WORKSPACE_ROOT / 'real_datasets' / 'xrf' / csv_file
        data_complete = pd.read_csv(str(csv_path), sep=sep)
        start_col, end_col = spectral_range
        data = data_complete.loc[:, start_col:end_col]

    data_A = data_complete[data_complete[class_col] == 'A'].reset_index(drop=True)
    data_B = data_complete[data_complete[class_col] == 'B'].reset_index(drop=True)

    if config.get('is_synthetic'):
        spectral_cols_list = [c for c in data_complete.columns if c != class_col]
        XA_all = data_A[spectral_cols_list]
        XB_all = data_B[spectral_cols_list]
    else:
        start_col, end_col = spectral_range
        XA_all = data_A.loc[:, start_col:end_col]
        XB_all = data_B.loc[:, start_col:end_col]

    XA_cal, XA_pred = ks.train_test_split(XA_all, test_size=test_size)
    XA_cal = XA_cal.reset_index(drop=True)
    XA_pred = XA_pred.reset_index(drop=True)

    XB_cal, XB_pred = ks.train_test_split(XB_all, test_size=test_size)
    XB_cal = XB_cal.reset_index(drop=True)
    XB_pred = XB_pred.reset_index(drop=True)

    Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)
    Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
    ycalclass = pd.Series(['A'] * XA_cal.shape[0] + ['B'] * XB_cal.shape[0])
    ypredclass = pd.Series(['A'] * XA_pred.shape[0] + ['B'] * XB_pred.shape[0])

    return Xcalclass, Xpredclass, ycalclass, ypredclass


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(config, Xcal, Xpred):
    """Apply preprocessing based on config. Returns (Xcal_prep, Xpred_prep, preprocess_info)."""
    method = config.get('preprocessing', 'poisson')
    mc = config.get('preprocessing_mc', True)

    if method == 'poisson':
        Xcal_prep, mean_original, mean_poisson = prepr.poisson(Xcal, mc=mc)
        Xpred_prep = (Xpred / np.sqrt(mean_original)) - mean_poisson
        return Xcal_prep, Xpred_prep, {'mean_original': mean_original, 'mean_poisson': mean_poisson}

    elif method == 'mc':
        Xcal_prep, mean_original = prepr.mc(Xcal)
        Xpred_prep = pd.DataFrame(Xpred.values - mean_original.values, columns=Xpred.columns)
        return Xcal_prep, Xpred_prep, {'mean_original': mean_original}

    elif method == 'savgol':
        from scipy.signal import savgol_filter
        params = config.get('savgol_params', {})
        wl = params.get('window_length', 11)
        po = params.get('polyorder', 3)
        deriv = params.get('deriv', 0)

        Xcal_sg = pd.DataFrame(
            savgol_filter(Xcal.values, window_length=wl, polyorder=po, deriv=deriv, axis=1),
            columns=Xcal.columns
        )
        Xpred_sg = pd.DataFrame(
            savgol_filter(Xpred.values, window_length=wl, polyorder=po, deriv=deriv, axis=1),
            columns=Xpred.columns
        )
        if mc:
            Xcal_prep, mean_sg = prepr.mc(Xcal_sg)
            Xpred_prep = pd.DataFrame(Xpred_sg.values - mean_sg.values, columns=Xpred.columns)
            return Xcal_prep, Xpred_prep, {'mean_sg': mean_sg}
        return Xcal_sg, Xpred_sg, {}

    elif method == 'auto_scaling':
        Xcal_prep, sd_original, mean_original = prepr.auto_scaling(Xcal)
        Xpred_prep = pd.DataFrame(
            (Xpred.values / sd_original.values) - mean_original.values,
            columns=Xpred.columns
        )
        return Xcal_prep, Xpred_prep, {'sd_original': sd_original, 'mean_original': mean_original}

    elif method == 'pareto':
        Xcal_prep, mean_pareto = prepr.pareto(Xcal, mc=mc)
        sd = np.std(Xcal, axis=0)
        escala_pareto = 1 / np.sqrt(sd)
        if mc:
            Xpred_prep = pd.DataFrame(
                (Xpred.values * escala_pareto.values) - mean_pareto.values,
                columns=Xpred.columns
            )
        else:
            Xpred_prep = pd.DataFrame(Xpred.values * escala_pareto.values, columns=Xpred.columns)
        return Xcal_prep, Xpred_prep, {'mean_pareto': mean_pareto}

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model_name, config, Xcal_prep, ycal, Xpred_prep, ypred):
    """Dispatch to pls_optimized / mlp_optimized / svm_optimized."""
    mc = MODEL_CONFIG[model_name]
    train_fn = mc['train_fn']
    kwargs = mc['train_kwargs'](config)
    result = train_fn(Xcal_prep, ycal, Xpred=Xpred_prep, ypred=ypred, **kwargs)
    return result


def extract_y_continuous(model_name, result):
    """Extract continuous predictions from model result tuple."""
    return pd.Series(MODEL_CONFIG[model_name]['y_pred_extractor'](result).values)


# ── LRC Pipelines ────────────────────────────────────────────────────────────

def _run_lrc_pipeline(config, metric_type, zone_scores_df, y_pred, predicates_quantiles,
                      pca_info_dict, Xcalclass, Xcalclass_prep, spectral_cuts,
                      model=None, model_name=None):
    """Run covariance or perturbation LRC pipeline across seeds.

    Returns: (lrc_summed_df, lrc_summed_unique_df, lrc_natural_df)
    """
    random_seeds = config.get('random_seeds', [0, 1, 2, 3])
    training_samples = len(zone_scores_df)
    y_pred_series = pd.Series(y_pred.values) if hasattr(y_pred, 'values') else pd.Series(y_pred)

    # Phase 1: compute metrics per seed
    all_results = {}
    for seed in random_seeds:
        print(f"\n{'='*70}")
        print(f"Processing seed ({metric_type}): {seed}")
        print(f"{'='*70}\n")

        bags_result_seed = exp.bagging_predicates(
            zone_sums_df=zone_scores_df,
            y_predicted_numeric=y_pred_series,
            predicates_df=predicates_quantiles[0],
            n_bags=10,
            n_samples_per_bag=int(training_samples * 0.8),
            min_samples_per_predicate=int(training_samples * 0.2),
            replace=False,
            sample_bagging=True,
            predicate_bagging=False,
            random_seed=seed
        )

        for bag_name, pred_dict in bags_result_seed.items():
            for pred_rule, df_info in pred_dict.items():
                df_info['Class_Predicted'] = np.where(
                    df_info['Predicted_Y'] >= 0.5, 'A', 'B'
                )

        if metric_type == 'covariance':
            metric_results = exp.calculate_predicate_metrics(
                bags_result=bags_result_seed,
                metric='covariance',
                threshold=0.01,
                n_neighbors=5
            )
        else:  # perturbation
            mc = MODEL_CONFIG[model_name]
            metric_results = exp.calculate_predicate_perturbation(
                estimator=model,
                Xcalclass_prep=Xcalclass_prep,
                folds_struct=bags_result_seed,
                predicates_df=predicates_quantiles[0],
                spectral_cuts=spectral_cuts,
                perturbation_mode='median',
                stats_source='full',
                aim=mc['perturbation_aim'],
                metric=mc['perturbation_metric'],
                verbose=True
            )

        all_results[seed] = {
            'bags_result': bags_result_seed,
            'metric_results': metric_results
        }

    # Phase 2: build graphs per seed
    metric_column = 'Covariance' if metric_type == 'covariance' else 'Perturbation'
    graphs_by_seed = {}
    for seed in random_seeds:
        print(f"\n{'='*70}")
        print(f"Processing graph ({metric_type}) - Seed: {seed}")
        print(f"{'='*70}\n")
        DG = exp.build_predicate_graph(
            bags_result=all_results[seed]['bags_result'],
            predicate_ranking_dict=all_results[seed]['metric_results'],
            metric_column=metric_column,
            random_state=seed,
            show_details=True,
            var_exp=True,
            pca_info_dict=pca_info_dict
        )
        graphs_by_seed[seed] = DG

    # Phase 3: LRC per seed
    lrc_by_seed = {}
    for seed in random_seeds:
        DG = graphs_by_seed[seed]
        predicate_nodes = [n for n, attr in DG.nodes(data=True)
                           if attr.get('node_type') == 'predicate']
        if len(predicate_nodes) == 0:
            print(f"  WARNING: seed {seed} produced an empty graph ({metric_type}), skipping.")
            continue
        lrc_df_seed = exp.calculate_lrc_single_graph(DG, predicates_quantiles[0])
        lrc_df_seed['Seed'] = seed
        lrc_by_seed[seed] = lrc_df_seed

    if not lrc_by_seed:
        raise RuntimeError(
            f"All seeds produced empty graphs for {metric_type}. "
            "The model may be degenerate (e.g., all predictions on one side)."
        )

    # Phase 4: aggregate across seeds
    valid_seeds = list(lrc_by_seed.keys())
    lrc_summed_df, lrc_summed_unique_df = exp.aggregate_lrc_across_seeds(
        lrc_by_seed, valid_seeds
    )
    print(lrc_summed_unique_df)

    # Phase 5: map to natural scale
    spectral_zones_original = exp.extract_spectral_zones(Xcalclass, spectral_cuts)
    zones_original = exp.aggregate_spectral_zones_pca(spectral_zones_original)

    lrc_natural_df = exp.map_thresholds_to_natural(
        lrc_df=lrc_summed_df,
        zone_sums_preprocessed=zone_scores_df,
        zone_sums_natural=zones_original[0]
    )
    print(lrc_natural_df)

    return lrc_summed_df, lrc_summed_unique_df, lrc_natural_df


# ── Debugging helpers ────────────────────────────────────────────────────────

def run_debugging(model_name, result, config, Xcalclass_prep, spectral_cuts, output_dir):
    """Run model-specific importance methods + SHAP zone + permutation importance.

    Returns dict with zone-deduplicated DataFrames for each method.
    """
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)
    results = {}

    # Model-specific importance
    if 'vip' in mc['importance_methods']:
        vip_scores_mat = result[4]  # PLS vip
        results['vip'] = dbg.vip_scores_per_zone(vip_scores_mat, spectral_cuts)
        print("VIP scores per zone:")
        print(results['vip'])

    if 'reg_coef' in mc['importance_methods']:
        results['reg_coef'] = dbg.regression_coefficients_per_zone(model, spectral_cuts)
        print("Regression coefficients per zone:")
        print(results['reg_coef'])

    if 'pvector' in mc['importance_methods']:
        results['pvector'] = dbg.svm_pvector_per_zone(
            model, Xcalclass_prep.columns, spectral_cuts
        )
        print("SVM P-vector per zone:")
        print(results['pvector'])

    # SHAP from pre-computed CSV
    dataset_name = config['name']
    shap_csv = output_dir / f'shap_{dataset_name}.csv'
    if shap_csv.exists():
        results['shap'] = dbg.shap_per_zone(str(shap_csv), spectral_cuts)
    else:
        warnings.warn(f"SHAP CSV not found at {shap_csv}, skipping SHAP.")

    # Permutation importance
    scoring_fn = mc['permutation_scoring_fn']
    if scoring_fn is not None:
        scoring_fn = scoring_fn(model)

    perm_unique, perm_full = exp.permutation_importance_per_zone(
        estimator=model,
        X=Xcalclass_prep,
        spectral_cuts=spectral_cuts,
        n_repeats=10,
        random_state=42,
        scoring_fn=scoring_fn,
    )
    results['permutation'] = perm_unique
    print("Permutation importance per zone:")
    print(perm_unique)

    return results


def build_feature_importance_table(model_name, debugging_results,
                                   lrc_cov_unique=None, lrc_pert_unique=None):
    """Assemble feature_importance DataFrame with model-appropriate columns."""
    mc = MODEL_CONFIG[model_name]

    # Collect all zone lists
    zone_lists = {}

    if lrc_cov_unique is not None:
        zone_lists['LRC_covariance'] = list(lrc_cov_unique['Zone'])
    if lrc_pert_unique is not None:
        zone_lists['LRC_perturbation'] = list(lrc_pert_unique['Zone'])

    if 'permutation' in debugging_results:
        zone_lists['Permutation'] = list(debugging_results['permutation']['Zone'])

    if 'vip' in debugging_results:
        zone_lists['VIP_Score'] = list(debugging_results['vip']['Zone'])
    if 'reg_coef' in debugging_results:
        zone_lists['Reg_Coefficient'] = list(debugging_results['reg_coef']['Zone'])
    if 'pvector' in debugging_results:
        zone_lists['SVM_pvector'] = list(debugging_results['pvector']['Zone'])
    if 'shap' in debugging_results:
        zone_lists['Shap'] = list(debugging_results['shap']['Zone'])

    if not zone_lists:
        return pd.DataFrame()

    max_len = max(len(v) for v in zone_lists.values())

    def pad(lst):
        return lst + [None] * (max_len - len(lst))

    # Canonical column order per model type (model-specific first, then common)
    COLUMN_ORDER = {
        'pls': ['VIP_Score', 'Reg_Coefficient', 'Shap', 'Permutation',
                 'LRC_perturbation', 'LRC_covariance'],
        'svm': ['SVM_pvector', 'Shap', 'Permutation',
                 'LRC_perturbation', 'LRC_covariance'],
        'mlp': ['Shap', 'Permutation', 'LRC_perturbation', 'LRC_covariance'],
    }
    ordered_cols = [c for c in COLUMN_ORDER.get(model_name, [])
                    if c in zone_lists]

    fi_data = {k: pad(v) for k, v in zone_lists.items()}
    df = pd.DataFrame(fi_data)
    return df[ordered_cols] if ordered_cols else df


# ── Main experiment orchestrator ─────────────────────────────────────────────

def run_single_experiment(dataset, model_name, method, debugging):
    """Orchestrate a single (dataset, model) experiment."""
    print(f"\n{'#'*70}")
    print(f"# Experiment: dataset={dataset}, model={model_name}, method={method}")
    print(f"{'#'*70}\n")

    # 1. Load config
    config = load_dataset_config(dataset)
    spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]

    # 2. Validate model compatibility
    if model_name not in config.get('compatible_models', []):
        print(f"WARNING: model '{model_name}' not compatible with dataset '{dataset}'. Skipping.")
        return

    # 3. Load data
    print("Loading data...")
    Xcalclass, Xpredclass, ycalclass, ypredclass = load_data(config)
    print(f"  Xcal: {Xcalclass.shape}, Xpred: {Xpredclass.shape}")

    # 4. Preprocess
    print("Preprocessing...")
    Xcalclass_prep, Xpredclass_prep, prep_info = preprocess(config, Xcalclass, Xpredclass)

    # 5. Train model
    print(f"Training {model_name}...")
    result = train_model(model_name, config, Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass)

    # 6. Extract predictions
    mc = MODEL_CONFIG[model_name]
    model = mc['model_extractor'](result)
    y_pred_cont = extract_y_continuous(model_name, result)

    # 7. PCA aggregation
    print("PCA aggregation...")
    spectral_zones_class = exp.extract_spectral_zones(Xcalclass_prep, spectral_cuts)
    zone_scores_df, pca_info_dict = exp.aggregate_spectral_zones_pca(spectral_zones_class)
    print(f"  Zone scores shape: {zone_scores_df.shape}")

    # 8. Predicates
    predicates_quantiles = exp.predicates_by_quantiles(zone_scores_df, [0.2, 0.4, 0.6, 0.8])

    # 9. Output directory
    output_dir = SCRIPT_DIR / model_name.upper() / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # 10. Run LRC pipelines
    lrc_cov_unique = None
    lrc_pert_unique = None
    lrc_cov_natural = None
    lrc_pert_natural = None

    if method in ('covariance', 'all'):
        print("\n--- Covariance Pipeline ---")
        _, lrc_cov_unique, lrc_cov_natural = _run_lrc_pipeline(
            config, 'covariance', zone_scores_df, y_pred_cont,
            predicates_quantiles, pca_info_dict,
            Xcalclass, Xcalclass_prep, spectral_cuts,
            model=model, model_name=model_name
        )

    if method in ('perturbation', 'all'):
        print("\n--- Perturbation Pipeline ---")
        _, lrc_pert_unique, lrc_pert_natural = _run_lrc_pipeline(
            config, 'perturbation', zone_scores_df, y_pred_cont,
            predicates_quantiles, pca_info_dict,
            Xcalclass, Xcalclass_prep, spectral_cuts,
            model=model, model_name=model_name
        )

    # 11. Debugging extras
    debugging_results = {}
    if debugging:
        print("\n--- Debugging ---")
        debugging_results = run_debugging(
            model_name, result, config, Xcalclass_prep, spectral_cuts, output_dir
        )

    # 12. Feature importance table
    if debugging:
        features_importance = build_feature_importance_table(
            model_name, debugging_results,
            lrc_cov_unique=lrc_cov_unique,
            lrc_pert_unique=lrc_pert_unique
        )
        if not features_importance.empty:
            features_importance.to_csv(
                output_dir / 'feature_importance.csv', index=False, sep=';'
            )
            print("\nFeature importance table:")
            print(features_importance)

            # RBO
            dbg.rbo_rank_comparison(features_importance, output_dir / 'rbo_rank.csv')

    # 13. Save LRC artifacts
    if lrc_cov_natural is not None:
        lrc_cov_natural.to_csv(output_dir / 'lrc_cov_natural.csv', index=False, sep=';')
    if lrc_pert_natural is not None:
        lrc_pert_natural.to_csv(output_dir / 'lrc_pert_natural.csv', index=False, sep=';')

    print(f"\nDone – artifacts saved to {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SMX Experiment Runner')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (from JSON config) or "all"')
    parser.add_argument('--model', required=True,
                        choices=['pls', 'mlp', 'svm', 'all'],
                        help='Model type or "all"')
    parser.add_argument('--method', default='all',
                        choices=['covariance', 'perturbation', 'all'],
                        help='LRC method(s) to run')
    parser.add_argument('--debugging', action='store_true',
                        help='Enable non-core extras (permutation, model importance, SHAP, RBO)')
    args = parser.parse_args()

    # Expand 'all'
    datasets = list_available_datasets() if args.dataset == 'all' else [args.dataset]
    models = ['pls', 'mlp', 'svm'] if args.model == 'all' else [args.model]

    for ds in datasets:
        for mdl in models:
            try:
                run_single_experiment(ds, mdl, args.method, args.debugging)
            except Exception as e:
                print(f"\nERROR running {ds}/{mdl}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
