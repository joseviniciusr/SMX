"""
PCA Aggregator – Soil (PLS-DA)
Equivalent script of pca_aggregator_soil.ipynb.
Generates: feature_importance.csv, rbo_rank.csv, lrc_cov_natural.csv, lrc_pert_natural.csv
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import kennard_stone as ks
import plotly.graph_objects as go
import rbo
import json
import sys
from pathlib import Path

pd.options.plotting.backend = 'plotly'

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
workspace_root = SCRIPT_DIR.parent.parent.parent
smx_dir = workspace_root / 'smx'
if str(smx_dir) not in sys.path:
    sys.path.insert(0, str(smx_dir))

import preprocessings as prepr
from modeling import pls_optimized
import explaining as exp
import debugging as dbg

IS_DEBUGGING = True

# ── Load config ──────────────────────────────────────────────────────────────
with open(workspace_root / 'real_datasets/xrf/soil.json') as f:
    config = json.load(f)

LVmax = config['LVmax']
aim = config['aim']
random_seeds = config['random_seeds']
spectral_cuts = [tuple(sc) for sc in config['spectral_cuts']]

# ── Load data ────────────────────────────────────────────────────────────────
data_complete = pd.read_csv(
    f'{workspace_root}/real_datasets/xrf/soil.csv', sep=';'
)
data = data_complete.loc[:, '1.32':'13.1']

data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# Kennard-Stone split
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '1.32':'13.1'], test_size=0.30)
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '1.32':'13.1'], test_size=0.30)
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A'] * XA_cal.shape[0] + ['B'] * XB_cal.shape[0])
ypredclass = pd.Series(['A'] * XA_pred.shape[0] + ['B'] * XB_pred.shape[0])

# Preprocessing (Poisson scaling)
Xcalclass_prep, mean_calclass, mean_calclass_poisson = prepr.poisson(Xcalclass, mc=True)
Xpredclass_prep = (Xpredclass / np.sqrt(mean_calclass)) - mean_calclass_poisson

# ── PLS-DA ───────────────────────────────────────────────────────────────────
plsda_results = pls_optimized(
    Xcalclass_prep,
    ycalclass,
    LVmax=LVmax,
    Xpred=Xpredclass_prep,
    ypred=ypredclass,
    aim='classification',
    cv=10
)

pls_model = plsda_results[3]
vip_scores_mat = plsda_results[4]
y_pred_cont = plsda_results[5].iloc[:, -1]

# Quick VIP plot (saved to HTML)
vip_scores_mat.T.plot().write_html(SCRIPT_DIR / 'vip_plot.html')

# spectral_cuts, LVmax, and aim loaded from soil.json

# ── VIP / Regression / SHAP per zone (debugging) ────────────────────────────
if IS_DEBUGGING:
    vip_scores_unique_df = dbg.vip_scores_per_zone(vip_scores_mat, spectral_cuts)
    reg_vet_unique_df = dbg.regression_coefficients_per_zone(pls_model, spectral_cuts)
    shap_unique_df = dbg.shap_per_zone(SCRIPT_DIR / 'shap_soil.csv', spectral_cuts)

# ── PCA aggregation ──────────────────────────────────────────────────────────
spectral_zones_class = exp.extract_spectral_zones(Xcalclass_prep, spectral_cuts)
zone_scores_df, pca_info_dict = exp.aggregate_spectral_zones_pca(spectral_zones_class)
print(f"\nScores DataFrame shape: {zone_scores_df.shape}")

# ── Predicates by quantiles ──────────────────────────────────────────────────
predicates_quantiles = exp.predicates_by_quantiles(zone_scores_df, [0.2, 0.4, 0.6, 0.8])
predicate_info_dict = exp.create_predicate_info_dict(
    predicates_df=predicates_quantiles[0],
    predicate_indicator_df=predicates_quantiles[1],
    zone_aggregated_df=zone_scores_df,
    y_predicted_numeric=y_pred_cont
)

# ── Covariance-based LRC (multiple seeds) ────────────────────────────────────
all_results_cov = {}
training_samples = len(Xcalclass)
y_pred_conticted_numeric = pd.Series(y_pred_cont)

for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando semente (cov): {seed}")
    print(f"{'='*70}\n")

    bags_result_seed = exp.bagging_predicates(
        zone_sums_df=zone_scores_df,
        y_predicted_numeric=y_pred_conticted_numeric,
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

    cov_results_dict_seed = exp.calculate_predicate_metrics(
        bags_result=bags_result_seed,
        metric='covariance',
        threshold=0.01,
        n_neighbors=5
    )

    all_results_cov[seed] = {
        'bags_result': bags_result_seed,
        'cov_results_dict': cov_results_dict_seed
    }

# Build graphs (covariance)
graphs_by_seed = {}
for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando Grafo (cov) - Semente: {seed}")
    print(f"{'='*70}\n")
    DG = exp.build_predicate_graph(
        bags_result=all_results_cov[seed]['bags_result'],
        predicate_ranking_dict=all_results_cov[seed]['cov_results_dict'],
        metric_column='Covariance',
        random_state=seed,
        show_details=True,
        var_exp=True,
        pca_info_dict=pca_info_dict
    )
    graphs_by_seed[seed] = DG

# LRC (covariance)
lrc_cov_by_seed = {}
for seed in random_seeds:
    DG = graphs_by_seed[seed]
    lrc_cov_df_seed = exp.calculate_lrc_single_graph(DG, predicates_quantiles[0])
    lrc_cov_df_seed['Seed'] = seed
    lrc_cov_by_seed[seed] = lrc_cov_df_seed

lrc_cov_all_seeds_df = pd.DataFrame()
for seed in random_seeds:
    lrc_cov_df_seed = lrc_cov_by_seed[seed].rename(
        columns={'Node': f'Predicate_Cov_Seed_{seed}'}
    )
    lrc_cov_all_seeds_df = pd.concat(
        [lrc_cov_all_seeds_df, lrc_cov_df_seed[[f'Predicate_Cov_Seed_{seed}']]],
        axis=1
    )

lrc_cov_unique_by_seed = {}
for seed, lrc_df in lrc_cov_by_seed.items():
    lrc_cov_unique_df = lrc_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
    lrc_cov_unique_df = lrc_cov_unique_df.sort_values(
        by='Local_Reaching_Centrality', ascending=False
    ).reset_index(drop=True)
    lrc_cov_unique_by_seed[seed] = lrc_cov_unique_df

print(lrc_cov_all_seeds_df)

# ── Aggregate covariance LRC across seeds ────────────────────────────────────
lrc_combined_list = []
for seed in random_seeds:
    lrc_combined_list.append(lrc_cov_by_seed[seed].copy())

lrc_all_seeds = pd.concat(lrc_combined_list, ignore_index=True)

lrc_summed_df_cov = lrc_all_seeds.groupby('Node').agg({
    'Local_Reaching_Centrality': 'mean',
    'Zone': 'first',
    'Threshold': 'first',
    'Operator': 'first'
}).reset_index()

lrc_summed_df_cov = lrc_summed_df_cov.sort_values(
    by='Local_Reaching_Centrality', ascending=False
).reset_index(drop=True)

lrc_summed_unique_df_cov = lrc_summed_df_cov.drop_duplicates(
    subset=['Zone'], keep='first'
).reset_index(drop=True)
print(lrc_summed_unique_df_cov)

# ── Map covariance thresholds to natural scale ───────────────────────────────
spectral_zones_original = exp.extract_spectral_zones(Xcalclass, spectral_cuts)
zones_original = exp.aggregate_spectral_zones_pca(spectral_zones_original)

lrc_summed_df_cov_natural = exp.map_thresholds_to_natural(
    lrc_df=lrc_summed_df_cov,
    zone_sums_preprocessed=zone_scores_df,
    zone_sums_natural=zones_original[0]
)
print(lrc_summed_df_cov_natural)

# ── Covariance threshold reconstruction plot ─────────────────────────────────
n = 62
zone_name = lrc_summed_df_cov_natural.iloc[n]['Zone']
threshold_score = float(lrc_summed_df_cov_natural.iloc[n]['Threshold_Natural'])
pca_info_dict_original = zones_original[1]

threshold_spectrum = exp.reconstruct_threshold_to_spectrum(
    threshold_value=threshold_score,
    zone_name=zone_name,
    pca_info_dict=pca_info_dict_original
)
print(f"\nEspectro de threshold reconstruído para zona '{zone_name}':")
print(f"  - Dimensão: {len(threshold_spectrum)} variáveis espectrais")
print(f"  - Range de energias: {threshold_spectrum.index[0]} - {threshold_spectrum.index[-1]}")
print(f"  - Variância explicada pela PC1: {pca_info_dict_original[zone_name]['variance_explained']:.2%}")

zone_df = spectral_zones_original[zone_name]
fig = go.Figure()
x_values = pd.to_numeric(zone_df.columns, errors='coerce')

colors = {'A': 'gold', 'B': 'blue'}
for idx, row in zone_df.iterrows():
    class_label = ycalclass.iloc[idx] if idx < len(ycalclass) else 'Unknown'
    fig.add_trace(go.Scatter(
        x=x_values, y=row.values, mode='lines',
        line=dict(color=colors.get(class_label, 'rgba(128,128,128,0.3)'), width=0.5),
        name=f'Class {class_label}', showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=x_values, y=threshold_spectrum.values, mode='lines',
    line=dict(color='red', width=4, dash='dash'),
    name=f'Threshold Spectrum ({threshold_spectrum.name})'
))

fig.update_layout(
    title=f"Zona '{zone_name}' com Threshold Multivariado "
          f"(Predicado: {lrc_summed_df_cov_natural.iloc[n]['Node_Natural']})",
    xaxis_title='Energia / Comprimento de Onda',
    yaxis_title='Intensidade',
    template='plotly_white', showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig.write_html(SCRIPT_DIR / 'threshold_cov_plot.html')

# ── Perturbation-based LRC (multiple seeds) ──────────────────────────────────
all_results_pert = {}

for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando semente (pert): {seed}")
    print(f"{'='*70}\n")

    bags_result_seed = exp.bagging_predicates(
        zone_sums_df=zone_scores_df,
        y_predicted_numeric=y_pred_conticted_numeric,
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

    pert_results_seed = exp.calculate_predicate_perturbation(
        estimator=pls_model,
        Xcalclass_prep=Xcalclass_prep,
        folds_struct=bags_result_seed,
        predicates_df=predicates_quantiles[0],
        spectral_cuts=spectral_cuts,
        perturbation_mode='median',
        stats_source='full',
        aim=aim,
        metric='mean_abs_diff',
        verbose=True
    )

    all_results_pert[seed] = {
        'bags_result': bags_result_seed,
        'pert_results_dict': pert_results_seed
    }

# Build graphs (perturbation)
graphs_pert_by_seed = {}
for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando Grafo (pert) - Semente: {seed}")
    print(f"{'='*70}\n")
    DG = exp.build_predicate_graph(
        bags_result=all_results_pert[seed]['bags_result'],
        predicate_ranking_dict=all_results_pert[seed]['pert_results_dict'],
        metric_column='Perturbation',
        random_state=seed,
        show_details=True,
        var_exp=True,
        pca_info_dict=pca_info_dict
    )
    graphs_pert_by_seed[seed] = DG

# LRC (perturbation)
lrc_pert_by_seed = {}
for seed in random_seeds:
    DG = graphs_pert_by_seed[seed]
    lrc_pert_df_seed = exp.calculate_lrc_single_graph(DG, predicates_quantiles[0])
    lrc_pert_df_seed['Seed'] = seed
    lrc_pert_by_seed[seed] = lrc_pert_df_seed

lrc_pert_all_seeds_df = pd.DataFrame()
for seed in random_seeds:
    lrc_pert_df_seed = lrc_pert_by_seed[seed].rename(
        columns={'Node': f'Predicate_pert_Seed_{seed}'}
    )
    lrc_pert_all_seeds_df = pd.concat(
        [lrc_pert_all_seeds_df, lrc_pert_df_seed[[f'Predicate_pert_Seed_{seed}']]],
        axis=1
    )

lrc_pert_unique_by_seed = {}
for seed, lrc_df in lrc_pert_by_seed.items():
    lrc_pert_unique_df = lrc_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
    lrc_pert_unique_df = lrc_pert_unique_df.sort_values(
        by='Local_Reaching_Centrality', ascending=False
    ).reset_index(drop=True)
    lrc_pert_unique_by_seed[seed] = lrc_pert_unique_df

print(lrc_pert_all_seeds_df)

# ── Aggregate perturbation LRC across seeds ──────────────────────────────────
lrc_combined_list = []
for seed in random_seeds:
    lrc_combined_list.append(lrc_pert_by_seed[seed].copy())

lrc_all_seeds = pd.concat(lrc_combined_list, ignore_index=True)

lrc_summed_df_pert = lrc_all_seeds.groupby('Node').agg({
    'Local_Reaching_Centrality': 'mean',
    'Zone': 'first',
    'Threshold': 'first',
    'Operator': 'first'
}).reset_index()

lrc_summed_df_pert = lrc_summed_df_pert.sort_values(
    by='Local_Reaching_Centrality', ascending=False
).reset_index(drop=True)

lrc_summed_unique_df_pert = lrc_summed_df_pert.drop_duplicates(
    subset=['Zone'], keep='first'
).reset_index(drop=True)
lrc_summed_unique_df_pert = lrc_summed_unique_df_pert.sort_values(
    by='Local_Reaching_Centrality', ascending=False
).reset_index(drop=True)
print(lrc_summed_unique_df_pert)

# ── Map perturbation thresholds to natural scale ─────────────────────────────
lrc_summed_df_pert_natural = exp.map_thresholds_to_natural(
    lrc_df=lrc_summed_df_pert,
    zone_sums_preprocessed=zone_scores_df,
    zone_sums_natural=zones_original[0]
)
print(lrc_summed_df_pert_natural)

# ── Perturbation threshold reconstruction plot ───────────────────────────────
n = 0
zone_name = lrc_summed_df_pert_natural.iloc[n]['Zone']
threshold_score = float(lrc_summed_df_pert_natural.iloc[n]['Threshold_Natural'])
pca_info_dict_original = zones_original[1]

threshold_spectrum = exp.reconstruct_threshold_to_spectrum(
    threshold_value=threshold_score,
    zone_name=zone_name,
    pca_info_dict=pca_info_dict_original
)
print(f"\nEspectro de threshold reconstruído para zona '{zone_name}':")
print(f"  - Dimensão: {len(threshold_spectrum)} variáveis espectrais")
print(f"  - Variância explicada pela PC1: {pca_info_dict_original[zone_name]['variance_explained']:.2%}")

zone_df = spectral_zones_original[zone_name]
fig = go.Figure()
x_values = pd.to_numeric(zone_df.columns, errors='coerce')

colors = {'A': 'gold', 'B': 'blue'}
for idx, row in zone_df.iterrows():
    class_label = ycalclass.iloc[idx] if idx < len(ycalclass) else 'Unknown'
    fig.add_trace(go.Scatter(
        x=x_values, y=row.values, mode='lines',
        line=dict(color=colors.get(class_label, 'rgba(128,128,128,0.3)'), width=0.5),
        name=f'Class {class_label}', showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=x_values, y=threshold_spectrum.values, mode='lines',
    line=dict(color='red', width=4, dash='dash'),
    name=f'Threshold Spectrum ({threshold_spectrum.name})'
))

fig.update_layout(
    title=f"Zona '{zone_name}' com Threshold Multivariado "
          f"(Predicado: {lrc_summed_df_pert_natural.iloc[n]['Node_Natural']})",
    xaxis_title='Energia / Comprimento de Onda',
    yaxis_title='Intensidade',
    template='plotly_white', showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig.write_html(SCRIPT_DIR / 'threshold_pert_plot.html')

# ── Permutation importance ───────────────────────────────────────────────────
n_repeats = 10
rng = np.random.RandomState(42)
baseline_pred = pls_model.predict(Xcalclass_prep)
importance_list = []
X_arr = Xcalclass_prep.copy()

for col in Xcalclass_prep.columns:
    diffs = []
    for _ in range(n_repeats):
        X_perm = X_arr.copy()
        X_perm[col] = rng.permutation(X_perm[col].values)
        perm_pred = pls_model.predict(X_perm)
        diffs.append(np.mean(np.abs(baseline_pred - perm_pred)))
    importance_list.append(np.mean(diffs))

permutation_df = pd.DataFrame({
    'energy': Xcalclass_prep.columns,
    'Permutation_importance': importance_list
})
permutation_df.sort_values(by='Permutation_importance', ascending=False, inplace=True)

energy_to_zone_vip = {}
for zone_name, start, end in spectral_cuts:
    for e in permutation_df['energy']:
        ef = float(e)
        if start <= ef <= end:
            energy_to_zone_vip[e] = zone_name
permutation_df['Zone'] = permutation_df['energy'].map(energy_to_zone_vip)

permutation_unique_df = permutation_df.drop_duplicates(
    subset=['Zone'], keep='first'
).reset_index(drop=True)
permutation_unique_df = permutation_unique_df.sort_values(
    by='Permutation_importance', ascending=False
)
print(permutation_unique_df)

# ── Feature importance summary ───────────────────────────────────────────────
lengths = [
    len(permutation_unique_df['Zone']),
    len(lrc_summed_unique_df_pert['Zone']),
    len(lrc_summed_unique_df_cov['Zone'])
]
if IS_DEBUGGING:
    lengths += [
        len(vip_scores_unique_df['Zone']),
        len(reg_vet_unique_df['Zone']),
        len(shap_unique_df['Zone']),
    ]
max_len = max(lengths)


def pad_list(lst, length):
    return list(lst) + [None] * (length - len(lst))


fi_data = {
    'Permutation': pad_list(permutation_unique_df['Zone'], max_len),
    'LRC_perturbation': pad_list(lrc_summed_unique_df_pert['Zone'], max_len),
    'LRC_covariance': pad_list(lrc_summed_unique_df_cov['Zone'], max_len),
}
if IS_DEBUGGING:
    fi_data['VIP_Score'] = pad_list(vip_scores_unique_df['Zone'], max_len)
    fi_data['Reg_Coefficient'] = pad_list(reg_vet_unique_df['Zone'], max_len)
    fi_data['Shap'] = pad_list(shap_unique_df['Zone'], max_len)
features_importance = pd.DataFrame(fi_data)

features_importance.to_csv(SCRIPT_DIR / 'feature_importance.csv', index=False, sep=';')
print(features_importance)

# ── RBO rank comparison (debugging) ──────────────────────────────────────────
if IS_DEBUGGING:
    dbg.rbo_rank_comparison(features_importance, SCRIPT_DIR / 'rbo_rank.csv')

# ── Save LRC natural-scale CSVs ─────────────────────────────────────────────
lrc_summed_df_cov_natural.to_csv(SCRIPT_DIR / 'lrc_cov_natural.csv', index=False, sep=';')
lrc_summed_df_pert_natural.to_csv(SCRIPT_DIR / 'lrc_pert_natural.csv', index=False, sep=';')

print("\nDone – all artifacts saved.")
