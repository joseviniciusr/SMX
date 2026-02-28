"""
Debugging utilities – optional analysis helpers (VIP, regression coefficients,
SHAP per zone, RBO rank comparison).

Toggle execution with the ``IS_DEBUGGING`` flag in the caller script.
"""

import pandas as pd
import rbo


def _map_energy_to_zone(energy_series, spectral_cuts):
    """Map energy values to their spectral zone names."""
    mapping = {}
    for zone_name, start, end in spectral_cuts:
        for e in energy_series:
            ef = float(e)
            if start <= ef <= end:
                mapping[e] = zone_name
    return mapping


def vip_scores_per_zone(vip_scores_mat, spectral_cuts):
    """Compute VIP scores and return a zone-deduplicated DataFrame."""
    vip_scores_df = pd.DataFrame({
        'energy': vip_scores_mat.T.index,
        'VIP_Score': vip_scores_mat.T.iloc[:, 0].values
    })
    vip_scores_df = vip_scores_df.sort_values(
        by='VIP_Score', ascending=False
    ).reset_index(drop=True)

    vip_scores_df['Zone'] = vip_scores_df['energy'].map(
        _map_energy_to_zone(vip_scores_df['energy'], spectral_cuts)
    )

    vip_scores_unique_df = vip_scores_df.drop_duplicates(
        subset=['Zone'], keep='first'
    ).reset_index(drop=True)
    vip_scores_unique_df = vip_scores_unique_df.sort_values(
        by='VIP_Score', ascending=False
    ).reset_index(drop=True)
    return vip_scores_unique_df


def regression_coefficients_per_zone(pls_model, spectral_cuts):
    """Compute regression coefficient ranking and return a zone-deduplicated DataFrame."""
    reg_vet = pd.DataFrame(
        pls_model.coef_, columns=pls_model.feature_names_in_
    ).T
    reg_vet.insert(0, 'energy', reg_vet.index)
    reg_vet = reg_vet.reset_index(drop=True)
    reg_vet.columns = ['energy', 'Reg_coef']
    reg_vet['Abs_Reg_coef'] = reg_vet['Reg_coef'].abs()
    reg_vet = reg_vet.sort_values(
        by='Abs_Reg_coef', ascending=False
    ).reset_index(drop=True)

    reg_vet['Zone'] = reg_vet['energy'].map(
        _map_energy_to_zone(reg_vet['energy'], spectral_cuts)
    )

    reg_vet_unique_df = reg_vet.drop_duplicates(
        subset=['Zone'], keep='first'
    ).reset_index(drop=True)
    reg_vet_unique_df = reg_vet_unique_df.sort_values(
        by='Abs_Reg_coef', ascending=False
    ).reset_index(drop=True)
    return reg_vet_unique_df


def shap_per_zone(shap_csv_path, spectral_cuts):
    """Load SHAP CSV and return a zone-deduplicated DataFrame."""
    shap_global_importance = pd.read_csv(shap_csv_path, sep=';')

    shap_global_importance['Zone'] = shap_global_importance['energy'].map(
        _map_energy_to_zone(shap_global_importance['energy'], spectral_cuts)
    )

    shap_unique_df = shap_global_importance.drop_duplicates(
        subset=['Zone'], keep='first'
    ).reset_index(drop=True)
    shap_unique_df = shap_unique_df.sort_values(
        by='Mean_Abs_SHAP', ascending=False
    ).reset_index(drop=True)
    print(shap_unique_df)
    return shap_unique_df


def rbo_rank_comparison(features_importance, output_path):
    """Compute pairwise RBO scores between feature-importance rankings and save to CSV."""
    rbo_comparison = pd.DataFrame(columns=['Method_1', 'Method_2', 'RBO_Score'])
    methods = features_importance.columns.tolist()

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method_1 = methods[i]
            method_2 = methods[j]
            list_1 = [x for x in features_importance[method_1].tolist() if x is not None]
            list_2 = [x for x in features_importance[method_2].tolist() if x is not None]
            min_len = min(len(list_1), len(list_2))
            list_1_trunc = list_1[:min_len]
            list_2_trunc = list_2[:min_len]
            rbo_score = rbo.RankingSimilarity(list_1_trunc, list_2_trunc).rbo(p=0.7, k=10)
            rbo_comparison = pd.concat([rbo_comparison, pd.DataFrame({
                'Method_1': [method_1],
                'Method_2': [method_2],
                'RBO_Score': [rbo_score]
            })], ignore_index=True)

    rbo_comparison.sort_values(by='RBO_Score', ascending=False, inplace=True)
    rbo_comparison.to_csv(output_path, index=False, sep=';')
    print(rbo_comparison)
    return rbo_comparison
