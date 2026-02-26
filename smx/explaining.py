from typing import Dict, List, Tuple, Optional, Union, Callable, Literal
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def extract_spectral_zones(Xcal, cuts):
    """
    Extract spectral zones from a DataFrame based on specified cuts.
    
    Parameters
    ----------
    - **Xcal** : pd.DataFrame
        DataFrame with spectral data, where columns are wavelengths/energies.
    - **cuts** : list of tuples/lists or dicts
        Each item defines a spectral zone to extract.
        - If tuple/list: (start, end) or (name, start, end)
        - If dict: {'name': str, 'start': float, 'end': float}
    
    Returns
    -------
    - **zones** : dict
        Dictionary where keys are zone names and values are DataFrames with the extracted spectral zones.
    """
    import numpy as np
    import pandas as pd

    # convert the column names to numeric when possible (NaN when not convertible)
    col_nums = pd.to_numeric(Xcal.columns.astype(str), errors='coerce')
    zones = {} # dictionary to store extracted zones

    for cut in cuts:
        # normalize cut format
        if isinstance(cut, dict): # if dict
            name = cut.get('name', f"{cut.get('start')}-{cut.get('end')}") # default name if not provided
            start = cut.get('start') # getting start value
            end = cut.get('end') # getting end value
        elif isinstance(cut, (list, tuple)): # if list/tuple
            if len(cut) == 2: 
                start, end = cut # getting start and end values
                name = f"{start}-{end}" # default name
            elif len(cut) == 3: # if name provided
                name, start, end = cut # getting name, start and end values
            else:
                raise ValueError("Cuts in tuple/list format must have 2 or 3 elements.")
        else:
            raise ValueError("Each cut must be a dict or a tuple/list.")

        # validate start and end
        try:
            s = float(start)
            e = float(end)
        except Exception: # Exception for conversion errors
            raise ValueError("star and end must be numeric values (int/float or convertible strings).")

        if s > e: # swap if necessary
            s, e = e, s

        # to select columns whose numeric value is in the interval [s, e]
        mask = (~np.isnan(col_nums)) & (col_nums >= s) & (col_nums <= e)
        selected_cols = list(Xcal.columns[mask])

        # piecing the zone DataFrame into the dictionary
        zones[name] = Xcal.loc[:, selected_cols]

    return zones

def aggregate_spectral_zones(spectral_zones_dict, aggregator='sum'):
    """
    Aggregate spectral zone values using different aggregation functions.
    
    This function processes each spectral zone (DataFrame with multiple energy columns)
    and reduces each row (sample) to a single numerical value using the specified
    aggregation function.
    
    Parameters
    ----------
    - **spectral_zones_dict** : dict
        Dictionary returned by extract_spectral_zones, where:
        - keys = spectral zone names (e.g., 'Ca ka', 'Fe ka')
        - values = DataFrames with spectral data (rows=samples, columns=energies)
    
    - **aggregator** : str, optional (default='sum')
        Aggregation function to apply across the columns of each zone. Options:
        - **'sum'**: Sum of all values in the zone (default)
        - **'mean'**: Arithmetic mean of the values
        - **'median'**: Median of the values
        - **'max'**: Maximum value in the zone
        - **'min'**: Minimum value in the zone
        - **'std'**: Standard deviation of the values
        - **'var'**: Variance of the values
        - **'extreme'**: Value of greatest magnitude (most intense) in the zone, i.e.,
          selects the value with the largest absolute value per sample (may be positive or negative)
    
    Returns
    -------
    - **aggregated_df** : pd.DataFrame
        DataFrame with aggregated values, where:
        - rows = samples (same index as the original DataFrames)
        - columns = spectral zones
        - values = aggregation result (same format as .sum(axis=1))
    
    Raises
    ------
    - ValueError
        If the specified aggregator is not recognized.
    """
    import pandas as pd
    import numpy as np
    
    # INPUT VALIDATION
    valid_aggregators = ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'extreme']
    
    if aggregator not in valid_aggregators:
        raise ValueError(
            f"Aggregator '{aggregator}' not recognized.\n"
            f"Valid options: {', '.join(valid_aggregators)}"
        )
    
    # AGGREGATOR MAPPING
    # Dictionary mapping strings to pandas aggregation functions
    aggregation_functions = {
        'sum': lambda df: df.sum(axis=1),        # sum across columns
        'mean': lambda df: df.mean(axis=1),      # arithmetic mean
        'median': lambda df: df.median(axis=1),  # median
        'max': lambda df: df.max(axis=1),        # maximum value
        'min': lambda df: df.min(axis=1),        # minimum value
        'std': lambda df: df.std(axis=1),        # standard deviation
        'var': lambda df: df.var(axis=1),        # variance
        # 'extreme': selects the value with the greatest magnitude (abs), preserving the sign
        'extreme': lambda df: df.apply(
            lambda row: (row.loc[row.abs().idxmax()] if row.notna().any() else np.nan),
            axis=1
        ),
    }
    
    # SPECTRAL ZONE AGGREGATION
    aggregated_dict = {}  # dictionary to store results
    
    for zone_name, zone_df in spectral_zones_dict.items():
        # Apply the selected aggregation function
        # The result is a Series (same structure as .sum(axis=1))
        aggregated_series = aggregation_functions[aggregator](zone_df)
        
        # Store in dictionary
        aggregated_dict[zone_name] = aggregated_series
    
    # FINAL DATAFRAME CONSTRUCTION
    # Each key becomes a column, preserving the original indices
        aggregated_df = pd.DataFrame(aggregated_dict)    
    return aggregated_df

def predicates_by_quantiles(zone_sums_df, quantiles):
    """
    Generate predicates based on specified quantiles for each column in a DataFrame
    and create a predicate indicator matrix.
    
    Parameters
    ----------
    - **zone_sums_df** : pd.DataFrame
        DataFrame with summed values for spectral zones.
    - **quantiles** : list of float
        List of quantiles (between 0 and 1) to generate predicates for.
    
    Returns
    -------
    - **predicates_df** : pd.DataFrame
        DataFrame containing the generated predicates with columns:
        'predicate', 'rule', 'zone', 'thresholds', 'operator'.
    - **predicate_indicator_df** : pd.DataFrame
        Binary indicator matrix (samples × predicates) where 1 indicates
        the sample satisfies the predicate, 0 otherwise.
    """
    import pandas as pd
    import numpy as np

    # calculating the quantiles for each column of zone_sums_df
    zone_quantiles = zone_sums_df.quantile(quantiles)
    
    zone_predicate_list = []
    predicate_num = 1
    for zone in zone_sums_df.columns:
        for q in quantiles:
            q_value = zone_quantiles.loc[q, zone]
            # <= Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} <= {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': "<="
            })
            predicate_num += 1
            # > Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} > {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': ">"
            })
            predicate_num += 1
    
    predicates_df = pd.DataFrame(zone_predicate_list)
    
    # Removing duplicate predicates based on 'rule' column
    # Some zones may have the same quantile values, creating duplicate rules
    initial_count = len(predicates_df)
    predicates_df = predicates_df.drop_duplicates(subset=['rule'], keep='first').reset_index(drop=True)
    final_count = len(predicates_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate predicates. Remaining: {final_count}")
    
    # Renumbering predicates after removing duplicates
    predicates_df['predicate'] = [f'P{i+1}' for i in range(len(predicates_df))]

    # Generating the predicate indicator DataFrame
    
    # function to evaluate a predicate for a given value
    def eval_predicate(value, thresholds, operator):
        if operator == "<=":
            return float(value <= float(thresholds))
        elif operator == ">":
            return float(value > float(thresholds))
        else:
            return np.nan
    
    # compute all columns first, then concatenate them at once
    columns_dict = {}
    
    # iterating over each predicate
    for _, row in predicates_df.iterrows():
        pred = row['predicate']
        zone = row['zone']
        thresholds = row['thresholds']
        operator = row['operator']
        columns_dict[pred] = zone_sums_df[zone].apply(
            lambda v: eval_predicate(v, thresholds, operator)
        ).astype(int)
    
    # create DataFrame from all columns at once
    predicate_indicator_df = pd.DataFrame(columns_dict, index=zone_sums_df.index)
    
    # setting column names to rules for better readability
    predicate_indicator_df.columns = predicates_df['rule'].tolist()
    
    # computing co-occurrence matrix
    co_occurrence_matrix = np.dot(predicate_indicator_df.T, predicate_indicator_df)
    co_occurrence_matrix_df = pd.DataFrame(co_occurrence_matrix, index=predicate_indicator_df.columns, columns=predicate_indicator_df.columns) 

    return predicates_df, predicate_indicator_df, co_occurrence_matrix_df

def create_predicate_info_dict(predicates_df, predicate_indicator_df, zone_aggregated_df, y_predicted_numeric):
    """
    Create a dictionary containing detailed information for each predicate.
    
    For each predicate, the following data are stored:
    - The aggregated spectral zone values (from the samples satisfying the predicate)
    - The model-predicted values (from the same samples)
    - Optionally: sample indices, predicted class, etc.
    
    Parameters
    ----------
    - **predicates_df** : pd.DataFrame
        DataFrame with predicates generated by `predicates_by_quantiles()` or similar.
        Required columns: ['predicate', 'rule', 'zone', 'thresholds', 'operator']
        
    - **predicate_indicator_df** : pd.DataFrame
        Binary indicator matrix (samples × predicates) returned by `predicates_by_quantiles()`.
        Columns correspond to predicate rules (e.g., "Ca ka <= 25.5").
        Values: 1 = sample satisfies the predicate, 0 = does not satisfy
        
    - **zone_aggregated_df** : pd.DataFrame
        DataFrame with aggregated spectral zone values (returned by `aggregate_spectral_zones()`).
        Rows = samples, Columns = spectral zones
        Values = aggregation result (sum, mean, median, std, etc.)
        
    - **y_predicted_numeric** : pd.Series, pd.DataFrame, or np.ndarray
        Model-predicted values (continuous).
        - For PLS-DA: values between 0 and 1 (e.g., `plsda_results[5].iloc[:, -1]`)
        - For PLS-R: continuous response variable values
        - Must have the same number of rows as `zone_aggregated_df`
    
    Returns
    -------
    - **predicate_info_dict** : dict
        Dictionary structured as:
        {
            'Ca ka <= 25.5': DataFrame({
                'Zone_Aggregated': [aggregated values of the Ca ka zone],
                'Predicted_Y': [model-predicted values],
                'Sample_Index': [original sample indices]
            }),
            'Fe ka > 10.2': DataFrame({...}),
            ...
        }
        
        - Keys: Predicate rules (strings)
        - Values: DataFrames with 3 columns:
            - **Zone_Aggregated**: Aggregated spectral zone values (may be sum, mean, median, etc.)
            - **Predicted_Y**: Model-predicted values for these samples
            - **Sample_Index**: Original sample indices (for traceability)
    
    Raises
    ------
    - ValueError
        If the input DataFrames have an incompatible number of samples
    - KeyError
        If any required column is missing
    """
    import pandas as pd
    import numpy as np
    
    # INPUT VALIDATION
    
    # Verify required columns in predicates_df
    required_cols = ['predicate', 'rule', 'zone', 'thresholds', 'operator']
    missing_cols = [col for col in required_cols if col not in predicates_df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in predicates_df: {missing_cols}")
    
    # Convert y_predicted_numeric to Series if necessary
    if isinstance(y_predicted_numeric, pd.DataFrame):
        y_predicted_numeric = y_predicted_numeric.iloc[:, -1]  # last column
    elif isinstance(y_predicted_numeric, np.ndarray):
        y_predicted_numeric = pd.Series(y_predicted_numeric)
    
    # Verify sample count compatibility
    n_samples_zones = len(zone_aggregated_df)
    n_samples_predicted = len(y_predicted_numeric)
    n_samples_indicators = len(predicate_indicator_df)
    
    if not (n_samples_zones == n_samples_predicted == n_samples_indicators):
        raise ValueError(
            f"Incompatible number of samples:\n"
            f"  zone_aggregated_df: {n_samples_zones}\n"
            f"  y_predicted_numeric: {n_samples_predicted}\n"
            f"  predicate_indicator_df: {n_samples_indicators}\n"
            f"All must have the same number of rows."
        )
    
    # INFORMATION DICTIONARY CONSTRUCTION
    
    predicate_info_dict = {}  # dictionary to store results
    n_predicates_processed = 0  # counter of processed predicates
    n_predicates_empty = 0  # counter of predicates with no samples
    
    # Iterate over each predicate
    for _, row in predicates_df.iterrows():
        
        pred_rule = row['rule']  # predicate rule (e.g., "Ca ka <= 25.5")
        zone_name = row['zone']  # spectral zone name (e.g., "Ca ka")
        
        # 1. IDENTIFY SAMPLES SATISFYING THE PREDICATE
        # Use the indicator matrix to filter samples
        # predicate_indicator_df has columns corresponding to predicate rules
        
        if pred_rule not in predicate_indicator_df.columns:
            # Predicate does not exist in the indicator matrix (should not occur)
            continue
        
        # Boolean mask: True = sample satisfies the predicate
        mask_satisfied = predicate_indicator_df[pred_rule] == 1
        
        # Indices of samples satisfying the predicate
        # Use np.where() for compatibility with all index types
        satisfied_indices = np.where(mask_satisfied)[0].tolist()
        
        # 2. VERIFY WHETHER ANY SAMPLES ARE SATISFIED
        if not satisfied_indices:  # empty list
            n_predicates_empty += 1
            continue  # skip this predicate (not added to the dictionary)
        
        # 3. EXTRACT AGGREGATED SPECTRAL ZONE VALUES
        # Aggregated values (sum, mean, median, std, etc.) of the corresponding zone
        zone_aggregated_values = zone_aggregated_df.loc[satisfied_indices, zone_name]
        
        # 4. EXTRACT MODEL-PREDICTED VALUES
        predicted_values = y_predicted_numeric.iloc[satisfied_indices]
        
        # 5. CREATE DATAFRAME WITH PREDICATE INFORMATION
        df_predicate_info = pd.DataFrame({
            'Zone_Aggregated': zone_aggregated_values.reset_index(drop=True),  # aggregated values
            'Predicted_Y': predicted_values.reset_index(drop=True),  # predicted values
            'Sample_Index': satisfied_indices  # original indices (for traceability)
        })
        
        # 6. STORE IN DICTIONARY
        predicate_info_dict[pred_rule] = df_predicate_info
        n_predicates_processed += 1
    
    return predicate_info_dict

def bagging_predicates(zone_sums_df, y_predicted_numeric, predicates_df, 
                          n_bags=50, n_predicates_per_bag=20, n_samples_per_bag=80, 
                          min_samples_per_predicate=5, replace=True, random_seed=42,
                          sample_bagging=True, predicate_bagging=True):
    """
    Perform predicate bagging with granular control over the sampling strategy.
    
    Bagging Strategy (Configurable):
    ================================
    1. **Row Sampling (Samples):**
       - sample_bagging=True: Randomly draws N samples for each bag
       - sample_bagging=False: Uses ALL samples in every bag
    
    2. **Column Sampling (Predicates):**
       - predicate_bagging=True: Randomly draws M predicates for each bag
       - predicate_bagging=False: Uses ALL predicates in every bag
    
    3. **Filtering and Validation:**
       - For each selected predicate, filters the samples that satisfy it
       - Discards predicates with insufficient coverage (when sample_bagging=True)
    
    Parameters
    ----------
    zone_sums_df : pd.DataFrame
        DataFrame with spectral zone sums (rows=samples, columns=zones).
        
    y_predicted_numeric : pd.Series or np.ndarray
        Model-predicted values (continuous, between 0 and 1 for PLS-DA).
        
    predicates_df : pd.DataFrame
        DataFrame with predicates. Required columns:
        - 'rule': Human-readable rule (e.g., "Ca ka <= 25.5")
        - 'zone': Spectral zone name
        - 'thresholds': Threshold value
        - 'operator': "<=" or ">"
        
    n_bags : int, default=50
        Number of bags (iterations) to create.
        
    n_predicates_per_bag : int, default=20
        Number of predicates to sample per bag.
        **Ignored if predicate_bagging=False.**
        
    n_samples_per_bag : int, default=80
        Number of samples to draw per bag.
        **Ignored if sample_bagging=False.**
        
    min_samples_per_predicate : int, default=5
        Minimum number of samples that must satisfy a predicate for it to be considered valid.
        **Applied only when sample_bagging=True.**
        
    replace : bool, default=True
        - True: Bootstrap (sampling with replacement)
        - False: Subsampling (without replacement)
        **Applied only when sample_bagging=True.**
        
    random_seed : int, default=42
        Random seed for reproducibility.
        
    sample_bagging : bool, default=True
        - True: Performs row subsampling (samples vary across bags)
        - False: Uses all samples in every bag
        
    predicate_bagging : bool, default=True
        - True: Performs column subsampling (predicates vary across bags)
        - False: Uses all predicates in every bag
    
    Returns
    -------
    bags_dict : dict
        Dictionary structured as:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
    
    """
    import numpy as np
    import pandas as pd
    
    # INITIALIZATION
    np.random.seed(random_seed)
    
    n_total_samples = len(zone_sums_df)
    predicate_rules = predicates_df['rule'].tolist()
    bags_dict = {}
    
    # MAIN LOOP: BAG CREATION
    for bag_num in range(1, n_bags + 1):
        
        # 1. SAMPLE SELECTION (ROWS) - Controlled via `sample_bagging`
        if sample_bagging:
            # Draw N samples (bootstrap or subsampling)
            bag_sample_indices = np.random.choice(
                range(n_total_samples),
                size=n_samples_per_bag,
                replace=replace  # True=bootstrap, False=subsampling
            )
        else:
            # Use ALL available samples
            bag_sample_indices = np.arange(n_total_samples)
        
        # 2. PREDICATE SELECTION (COLUMNS) - Controlled via `predicate_bagging`
        if predicate_bagging:
            # Draw M predicates randomly (without replacement)
            selected_predicate_rules = np.random.choice(
                predicate_rules,
                size=min(n_predicates_per_bag, len(predicate_rules)),
                replace=False
            )
        else:
            # Use ALL available predicates
            selected_predicate_rules = predicate_rules
        
        # 3. PREDICATE FILTERING AND VALIDATION
        bag_predicate_dict = {}
        n_discarded = 0
        
        for pred_rule in selected_predicate_rules:
            
            # Retrieve predicate metadata
            pred_row_filtered = predicates_df[predicates_df['rule'] == pred_rule]
            if len(pred_row_filtered) == 0:
                continue  # Predicate not found, skip
            pred_row = pred_row_filtered.iloc[0]
            zone = pred_row['zone']
            threshold = float(pred_row['thresholds'])
            operator = pred_row['operator']
            
            # Extract zone values for the bag samples
            zone_values_bag = zone_sums_df.loc[bag_sample_indices, zone].values
            
            # Apply the predicate rule
            if operator == "<=":
                mask_satisfied = zone_values_bag <= threshold
            elif operator == ">":
                mask_satisfied = zone_values_bag > threshold
            else:
                continue  # Invalid operator, skip
            
            # Filter samples satisfying the predicate
            satisfied_indices_in_bag = bag_sample_indices[mask_satisfied]
            
            # Minimum coverage validation (only when sample_bagging=True)
            if sample_bagging and len(satisfied_indices_in_bag) < min_samples_per_predicate:
                n_discarded += 1
                continue
            
            # Basic validation (always discard empty predicates)
            if len(satisfied_indices_in_bag) == 0:
                n_discarded += 1
                continue
            
            # Store valid predicate data
            df_predicate_info = pd.DataFrame({
                'Zone_Sum': zone_sums_df.loc[satisfied_indices_in_bag, zone].values,
                'Predicted_Y': y_predicted_numeric.iloc[satisfied_indices_in_bag].values,
                'Sample_Index': satisfied_indices_in_bag
            })
            
            bag_predicate_dict[pred_rule] = df_predicate_info
        
        # 4. BAG STORAGE
        if len(bag_predicate_dict) > 0:
            bags_dict[f'Bag_{bag_num}'] = bag_predicate_dict
            
            # Informative log
            samp_str = "Yes" if sample_bagging else "No"
            pred_str = f"Yes ({n_predicates_per_bag})" if predicate_bagging else "No (All)"
            print(f"Bag_{bag_num} | Samples: {samp_str} | Predicates: {pred_str} | "
                  f"Valid: {len(bag_predicate_dict)} | Discarded: {n_discarded}")
        else:
            print(f"Bag_{bag_num}: EMPTY (all predicates discarded)")
    
    return bags_dict

def calculate_predicate_metrics(bags_result, metric='mutual_info', threshold=0.1, n_neighbors=10):
    """
    Compute association metrics between aggregated spectral zone values 
    and model predictions for each predicate in each bag.
    
    This function processes all bags generated by `bagging_predicates()` and computes
    the strength of association between spectral zone values and the model's continuous
    predictions. Two metrics are supported: Mutual Information and Covariance.
    
    Parameters
    ----------
    - **bags_result** : dict
        Dictionary returned by `bagging_predicates_v3()`, structured as:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
        
    - **metric** : str, optional (default='mutual_info')
        Association metric to compute. Options:
        - **'mutual_info'**: Mutual Information (MI) - Measures non-linear dependence
        - **'covariance'**: Covariance - Measures linear dependence
        
    - **threshold** : float, optional (default=0.1)
        Minimum metric value for a predicate to be considered relevant.
        Predicates with metric < threshold are FILTERED from the result.
        - For MI: typical values between 0.0 and 1.0+ (higher = more informative)
        - For Covariance: values depend on data scale (absolute values are used)
        
    - **n_neighbors** : int, optional (default=10)
        Number of neighbors for Mutual Information computation.
        **Used only when metric='mutual_info'. Ignored for covariance.**
        - Low values (3-5): more sensitive to local noise
        - Medium values (10-20): balance between sensitivity and robustness (recommended)
        - High values (>30): smoother, less sensitive to local variations
    
    Returns
    -------
    - **metrics_results_dict** : dict
        Dictionary structured as:
        {
            'Bag_1': DataFrame({
                'Predicate': ['Ca ka <= 25.5', 'Fe ka > 10.2', ...],
                'Mutual_Info': [0.45, 0.32, ...]  # or 'Covariance' if metric='covariance'
            }),
            'Bag_2': DataFrame({...}),
            ...
        }
        
        Each DataFrame contains:
        - **Predicate**: Predicate rule (string)
        - **Mutual_Info** or **Covariance**: Computed metric value
        - Sorted in DESCENDING order by metric (highest values first)
        - Filtered to retain only predicates with metric > threshold
    
    Raises
    ------
    - ValueError
        If metric is not 'mutual_info' or 'covariance'
        
    - KeyError
        If any bag does not contain the expected columns ('Zone_Sum', 'Predicted_Y')
    
    Notes
    -----
    - **Mutual Information (MI):**
        - Captures both LINEAR and NON-LINEAR dependencies between X and Y
        - Values are always >= 0 (0 = independence, >0 = dependence)
        - More robust to outliers than covariance
        - Computationally more expensive
        - Suitable for complex/non-linear relationships
    
    - **Covariance:**
        - Captures only LINEAR dependencies
        - Values can be positive or negative (absolute values are used)
        - Sensitive to outliers and data scale
        - Computationally less expensive
        - Suitable for simple linear relationships
    
    - **Threshold:**
        - Defines the "relevance cutoff" for filtering weak predicates
        - Very low values: retains many predicates (some irrelevant)
        - Very high values: may discard useful predicates
        - Recommendation: start with 0.1 for MI, adjust as needed
    """
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    
    # INPUT VALIDATION    
    valid_metrics = ['mutual_info', 'covariance']
    if metric not in valid_metrics:
        raise ValueError(
            f"Metric '{metric}' not recognized.\n"
            f"Valid options: {', '.join(valid_metrics)}"
        )
    
    if not isinstance(bags_result, dict):
        raise TypeError("bags_result must be a dictionary returned by bagging_predicates_v3()")
    
    # INITIALIZATION    
    metrics_results_dict = {}  # dictionary to store results
    metric_name = 'Mutual_Info' if metric == 'mutual_info' else 'Covariance'
    
    total_bags = len(bags_result)
    total_predicates_processed = 0
    total_predicates_filtered = 0
    
    print(f"Computing {metric_name} for Predicates")
    print(f"Metric: {metric}")
    print(f"Threshold: {threshold}")
    
    # MAIN LOOP: PROCESS EACH BAG    
    for bag_name, predicates_dict in bags_result.items():
        
        if len(predicates_dict) == 0:
            print(f"{bag_name}: EMPTY (skipping)")
            continue
        
        # 1. COMPUTE METRIC FOR EACH PREDICATE IN THE BAG        
        metrics = {}  # temporary dictionary {predicate_rule: metric_value}
        
        for pred_rule, df_info in predicates_dict.items():
            
            # Validate required columns
            required_cols = ['Zone_Sum', 'Predicted_Y']
            missing_cols = [col for col in required_cols if col not in df_info.columns]
            if missing_cols:
                raise KeyError(
                    f"Bag '{bag_name}', Predicate '{pred_rule}': "
                    f"Missing columns: {missing_cols}"
                )
            
            # Extract data
            X_zone = df_info['Zone_Sum'].values.reshape(-1, 1)  # zone values (2D for sklearn)
            y_pred = df_info['Predicted_Y'].values  # predicted values (1D)
            
            # Verify sufficient data availability
            if len(X_zone) < 2:
                metrics[pred_rule] = 0.0  # insufficient data to compute metric
                continue
            
            # Compute the selected metric
            if metric == 'mutual_info':
                # Mutual Information (non-linear)
                mi_score = mutual_info_regression(
                    X_zone, 
                    y_pred, 
                    discrete_features=False,  # X is continuous
                    n_neighbors=n_neighbors,
                    random_state=42  # reproducibility
                )
                metrics[pred_rule] = mi_score[0]  # MI returns an array of 1 element
                
            elif metric == 'covariance':
                # Covariance (linear) - absolute value is used
                # np.cov returns a 2x2 matrix: [[var(X), cov(X,Y)], [cov(Y,X), var(Y)]]
                # The target is cov(X,Y) = element [0,1] or [1,0]
                cov_matrix = np.cov(X_zone.flatten(), y_pred)
                covariance = cov_matrix[0, 1]  # X-Y covariance
                metrics[pred_rule] = np.abs(covariance)  # absolute value
        
        total_predicates_processed += len(metrics)
        
        # 2. CONVERT TO DATAFRAME AND SORT        
        metrics_df = pd.DataFrame.from_dict(
            metrics, 
            orient='index',  # keys = indices, values = column
            columns=[metric_name]
        )
        
        # Add predicate column
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Sort in DESCENDING order (higher values = more informative)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # 3. FILTER BY THRESHOLD        
        n_before_filter = len(metrics_df)
        metrics_df = metrics_df[metrics_df[metric_name] > threshold].reset_index(drop=True)
        n_after_filter = len(metrics_df)
        n_filtered = n_before_filter - n_after_filter
        
        total_predicates_filtered += n_filtered
        
        # 4. STORE RESULT        
        metrics_results_dict[bag_name] = metrics_df
    
    return metrics_results_dict

def calculate_lrc(graphs_by_seed, predicates_df):
    """
    Compute Local Reaching Centrality (LRC) for all nodes in the graphs.
    
    The LRC measures the importance of each node based on its ability to reach
    other nodes in the graph, weighted by edge weights. Nodes with higher LRC
    are more central/important within the graph structure.
    
    Parameters
    ----------
    - **graphs_by_seed** : dict
        Dictionary of NetworkX graphs indexed by seed (returned by build_predicate_graphs).
        Structure: {seed1: nx.DiGraph(), seed2: nx.DiGraph(), ...}
        
    - **predicates_df** : pd.DataFrame
        DataFrame with predicate information. Required columns:
        - 'rule': Predicate rule (e.g., "Ca ka <= 25.5")
        - 'zone': Spectral zone name
        - 'thresholds': Threshold value
        - 'operator': "<=" or ">"
    
    Returns
    -------
    - **lrc_by_seed** : dict
        Dictionary of LRC DataFrames for each seed:
        {
            seed1: DataFrame(['Node', 'Local_Reaching_Centrality', 'Zone', 'Threshold', 'Operator', 'Seed']),
            seed2: DataFrame([...]),
            ...
        }
        
        Each DataFrame contains:
        - **Node**: Node name (predicate rule or 'Class_eut'/'Class_dist')
        - **Local_Reaching_Centrality**: LRC value (higher = more important)
        - **Zone**: Spectral zone name (None for terminal nodes)
        - **Threshold**: Threshold value (None for terminal nodes)
        - **Operator**: Rule operator (None for terminal nodes)
        - **Seed**: Random seed used
        
        **Sorting**: Descending by LRC (most important nodes first)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
        
    # LRC COMPUTATION
    
    lrc_by_seed = {}
    
    for seed, DG in graphs_by_seed.items():
        print(f"\nProcessing LRC - Seed: {seed}")
        
        # 1. COMPUTE LRC FOR EACH NODE
        local_reaching_centrality = {
            node: nx.local_reaching_centrality(DG, node, weight='weight')
            for node in DG.nodes()
        }
        
        # Sort by LRC (descending)
        sorted_lrc = sorted(
            local_reaching_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 2. CREATE DATAFRAME WITH LRC
        lrc_df_seed = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
        
        # 3. EXTRACT PREDICATE METADATA
        zones = []
        thresholds = []
        operators = []
        
        for node in lrc_df_seed['Node']:
            if node.startswith('Class_'):
                # Terminal node
                zones.append(None)
                thresholds.append(None)
                operators.append(None)
            else:
                # Predicate: retrieve metadata from predicates_df
                pred_row_filtered = predicates_df[predicates_df['rule'] == node]
                
                if len(pred_row_filtered) == 0:
                    # Predicate not found (should not occur)
                    zones.append('Unknown')
                    thresholds.append(None)
                    operators.append(None)
                else:
                    pred_row = pred_row_filtered.iloc[0]
                    zones.append(pred_row['zone'])
                    thresholds.append(pred_row['thresholds'])
                    operators.append(pred_row['operator'])
        
        # Add columns to the DataFrame
        lrc_df_seed['Zone'] = zones
        lrc_df_seed['Threshold'] = thresholds
        lrc_df_seed['Operator'] = operators
        lrc_df_seed['Seed'] = seed
        
        # Store result
        lrc_by_seed[seed] = lrc_df_seed
    
        return lrc_by_seed

def calculate_lrc_single_graph(graph, predicates_df):
    """
    Compute Local Reaching Centrality (LRC) for all nodes of a single graph.
    
    The LRC measures the importance of each node based on its ability to reach
    other nodes in the graph, weighted by edge weights. Nodes with higher LRC
    are more central/important within the graph structure.
    
    Parameters
    ----------
    - **graph** : nx.DiGraph
        NetworkX directed graph (returned by build_fold_predicate_graph or similar).
        
    - **predicates_df** : pd.DataFrame
        DataFrame with predicate information. Required columns:
        - 'rule': Predicate rule (e.g., "Ca ka <= 25.5")
        - 'zone': Spectral zone name
        - 'thresholds': Threshold value
        - 'operator': "<=" or ">"
    
    Returns
    -------
    - **lrc_df** : pd.DataFrame
        DataFrame with the following columns:
        - **Node**: Node name (predicate rule or 'Class_A'/'Class_B')
        - **Local_Reaching_Centrality**: LRC value (higher = more important)
        - **Zone**: Spectral zone name (None for terminal nodes)
        - **Threshold**: Threshold value (None for terminal nodes)
        - **Operator**: Rule operator (None for terminal nodes)
        
        **Sorting**: Descending by LRC (most important nodes first)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    
    print(f"\nProcessing graph LRC...")
    
    # 1. COMPUTE LRC FOR EACH NODE
    local_reaching_centrality = {}
    for node in graph.nodes():
        try:
            lrc_val = nx.local_reaching_centrality(graph, node, weight='weight')
        except ZeroDivisionError:
            # Occurs when the internal NetworkX computation attempts division by zero.
            # In this case, LRC is set to 0.0 to maintain execution and consistency.
            lrc_val = 0.0
        local_reaching_centrality[node] = lrc_val
    
    # Sort by LRC (descending)
    sorted_lrc = sorted(
        local_reaching_centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 2. CREATE DATAFRAME WITH LRC
    lrc_df = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
    
    # 3. EXTRACT PREDICATE METADATA
    zones = []
    thresholds = []
    operators = []
    
    for node in lrc_df['Node']:
        if node.startswith('Class_'):
            # Terminal node
            zones.append(None)
            thresholds.append(None)
            operators.append(None)
        else:
            # Predicate: retrieve metadata from predicates_df
            pred_row_filtered = predicates_df[predicates_df['rule'] == node]
            
            if len(pred_row_filtered) == 0:
                # Predicate not found (should not occur)
                zones.append('Unknown')
                thresholds.append(None)
                operators.append(None)
            else:
                pred_row = pred_row_filtered.iloc[0]
                zones.append(pred_row['zone'])
                thresholds.append(pred_row['thresholds'])
                operators.append(pred_row['operator'])
    
    # Add columns to the DataFrame
    lrc_df['Zone'] = zones
    lrc_df['Threshold'] = thresholds
    lrc_df['Operator'] = operators
    
    return lrc_df    

def get_zone_columns_from_predicate(
    predicate_rule: str,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    Xcal_columns: pd.Index
) -> List[str]:
    """
    Retrieve the spectral columns corresponding to the zone of a given predicate.
    
    This function identifies which spectral zone is associated with a predicate
    and returns the list of columns (variables) composing that zone.
    
    Parameters
    ----------
    predicate_rule : str
        Predicate rule (e.g., 'F1 <= 10.5')
    predicates_df : pd.DataFrame
        DataFrame with predicate information (columns: 'rule', 'zone', etc.)
    spectral_cuts : list of tuples
        List of spectral cuts in the format [(name, start, end), ...]
    Xcal_columns : pd.Index
        Column index of the calibration DataFrame (energies)
    
    Returns
    -------
    list
        List of column names (strings) composing the spectral zone
    
    Raises
    ------
    ValueError
        If the zone is not found in spectral_cuts
    KeyError
        If the predicate does not exist in predicates_df
    
    Example
    -------
    >>> zone_cols = get_zone_columns_from_predicate('F1 <= 10.5', predicates_df, spectral_cuts, Xcal.columns)
    >>> print(f"Zone contains {len(zone_cols)} columns: {zone_cols[:3]}...")
    """
    # 1. Find the zone associated with the predicate
    mask = predicates_df['rule'] == predicate_rule
    if not mask.any():
        raise KeyError(f"Predicate '{predicate_rule}' not found in predicates_df")
    
    zone_name = predicates_df.loc[mask, 'zone'].values[0]
    
    # 2. Find the zone boundaries in spectral_cuts
    zone_start, zone_end = None, None
    for cut in spectral_cuts:
        if len(cut) == 3:
            name, start, end = cut
        elif len(cut) == 2:
            start, end = cut
            name = f"{start}-{end}"
        else:
            continue
        
        if name == zone_name:
            zone_start, zone_end = float(start), float(end)
            break
    
    if zone_start is None or zone_end is None:
        raise ValueError(f"Zone '{zone_name}' not found in spectral_cuts")
    
    # 3. Select columns within the interval
    # Convert column names to numeric when possible
    col_numeric = pd.to_numeric(Xcal_columns.astype(str), errors='coerce')
    
    # Mask for columns within the interval [zone_start, zone_end]
    mask_cols = (~np.isnan(col_numeric)) & (col_numeric >= zone_start) & (col_numeric <= zone_end)
    
    zone_columns = list(Xcal_columns[mask_cols])
    
    return zone_columns

def spectral_perturbation_importance(model, X, y_pred_original, spectral_cuts, 
                                      perturbation_value=0, metric='mean_abs_diff'):
    """
    Perturb spectral regions and evaluate the impact on model predictions.
    
    Parameters
    ----------
    model : estimator
        Trained model (e.g., PLS-DA)
    X : pd.DataFrame
        Original spectral data (samples x wavelengths)
    y_pred_original : array-like
        Original model predictions
    spectral_cuts : list of tuples
        List of tuples (zone_name, start, end) defining spectral regions
    perturbation_value : float, default=0
        Value to use for perturbation (0 to zero out, 1 to set to 1, etc.)
    metric : str, default='mean_abs_diff'
        Metric to compute importance: 'mean_abs_diff', 'mean_diff', 'mean_relative_dev'.
        - 'mean_abs_diff': mean of the absolute difference
        - 'mean_diff': mean of the signed difference
        - 'mean_relative_dev': mean of the relative deviation (caution with division by zero)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with spectral zone and importance (mean prediction difference)
    """
    import pandas as pd
    import numpy as np
    
    results = []
    
    for zone_name, start, end in spectral_cuts:
        # Create a copy of the data for perturbation
        X_perturbed = X.copy()
        # Identify columns within the spectral zone range
        cols_to_perturb = [col for col in X.columns if start <= float(col) <= end]
        # Perturb the columns (set to the specified value)
        X_perturbed[cols_to_perturb] = perturbation_value
        # Predict with perturbed data
        y_pred_perturbed = model.predict(X_perturbed)
        # Compute the difference between predictions
        if metric == 'mean_abs_diff':
            importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
        elif metric == 'mean_diff':
            importance = np.mean(y_pred_original - y_pred_perturbed)
        elif metric == 'mean_relative_dev':
            y_pred_original_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
            rel_dev = (y_pred_perturbed - y_pred_original) / y_pred_original_safe
            importance = np.nanmean(rel_dev)
        else:
            raise ValueError(f"Metric '{metric}' not supported. Use 'mean_abs_diff', 'mean_diff', or 'mean_relative_dev'.")
        
        if metric == 'mean_relative_dev' or metric == 'mean_diff':
            pass  # importance already retains the sign
            # Store results
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'Abs_Importance': np.abs(importance),
                'N_Features': len(cols_to_perturb)
            })
        else:
            # Store results
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'N_Features': len(cols_to_perturb)
            })

    # Create DataFrame and sort by importance
    results_df = pd.DataFrame(results)
    if metric == 'mean_relative_dev' or metric == 'mean_diff':
        results_df = results_df.sort_values(by='Abs_Importance', ascending=False).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return results_df

def calculate_predicate_perturbation(
    estimator,
    Xcalclass_prep: pd.DataFrame,
    folds_struct: Dict,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    y_calclass: Union[pd.Series, np.ndarray] = None,
    aim: str = 'regression',
    perturbation_value: float = 0,
    perturbation_mode: str = 'constant',
    stats_source: str = 'full',
    metric: str = 'mean_abs_diff',
    verbose: bool = False,
    save_detailed_results: bool = True
) -> Dict:
    """
    Compute the importance of each predicate using Spectral Perturbation.
    
    This function serves as an alternative to permutation-based approaches. Instead of
    permuting values, it replaces the spectral zone values with a fixed value (e.g., 0)
    or with a column-wise statistic such as the mean, median, maximum, or minimum.
    It then measures the impact (change) on the model predictions.
    
    Both REGRESSION and CLASSIFICATION tasks are supported via the `aim` parameter.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Trained model with a predict() method.
        For classification with certain metrics, it may also require:
        - predict_proba(): for the 'probability_shift' metric
        - decision_function(): for the 'decision_function_shift' metric
        
    Xcalclass_prep : pd.DataFrame
        Pre-processed calibration dataset (n_samples × n_features)
        
    folds_struct : dict
        Fold structure in the format:
        {'Fold_1': {'rule1': DataFrame, 'rule2': DataFrame, ...}, ...}
        
    predicates_df : pd.DataFrame
        DataFrame with predicate information (columns: 'rule', 'zone', etc.)
        
    spectral_cuts : list of tuples
        List of spectral cuts: [(name, start, end), ...]
        
    y_calclass : pd.Series or np.ndarray, optional
        True sample labels. Required for classification metrics
        that compare against ground truth (e.g., 'accuracy_drop', 'f1_drop').
        
    aim : str, default='regression'
        Task type:
        - 'regression': uses predict() and continuous numerical metrics
        - 'classification': uses predict(), predict_proba(), or decision_function()
                           depending on the chosen metric
        
    perturbation_value : float, default=0
        Value used to perturb the zone when perturbation_mode='constant'
        
    perturbation_mode : str, default='constant'
        Perturbation mode:
        - 'constant': uses perturbation_value for all columns (original behavior)
        - 'mean': replaces each column by its mean
        - 'median': replaces each column by its median
        - 'min': replaces each column by its minimum value
        - 'max': replaces each column by its maximum value
        
    stats_source : str, default='full'
        Data source for computing statistics:
        - 'full': uses the entire dataset (Xcalclass_prep)
        - 'predicate': uses only samples belonging to the predicate
        
    metric : str, default='mean_abs_diff'
        Metric for computing importance. Available metrics depend on `aim`:
        
        **For aim='regression':**
        - 'mean_abs_diff': Mean of the absolute difference between predictions (|y_orig - y_pert|)
        - 'mean_diff': Mean of the signed difference (y_orig - y_pert)
        - 'mean_relative_dev': Mean of the relative deviation ((y_pert - y_orig) / y_orig)
        
        **For aim='classification':**
        - 'prediction_change_rate': Proportion of samples that changed class after
          perturbation. Values from 0 to 1, where 1 = all changed. Does not require y_calclass.
          Uses: estimator.predict()
          
        - 'accuracy_drop': Accuracy decrease after perturbation (acc_orig - acc_pert).
          Positive values indicate performance degradation. Requires y_calclass.
          Uses: estimator.predict()
          
        - 'f1_drop': F1-score decrease after perturbation (f1_orig - f1_pert).
          Positive values indicate performance degradation. Requires y_calclass.
          Uses: estimator.predict()
          
        - 'probability_shift': Mean of the absolute difference in predicted probabilities.
          Measures how probabilities change after perturbation. Does not require y_calclass.
          Uses: estimator.predict_proba() - REQUIRES a model with predict_proba (e.g., SVC with probability=True)
          
        - 'decision_function_shift': Mean of the absolute difference in decision function
          values. Useful for SVM and linear models. Does not require y_calclass.
          Uses: estimator.decision_function() - REQUIRES a model with decision_function (e.g., SVC, LinearSVC)
        
    verbose : bool, default=False
        If True, prints progress details
        
    save_detailed_results : bool, default=True
        If True, saves detailed results
    
    Returns
    -------
    dict
        Dictionary in a format compatible with calculate_predicate_metrics_permutation:
        {'Fold_1': DataFrame({'Predicate': [...], 'Perturbation': [...]}), ...}
        
    Notes
    -----
    Metric compatibility with sklearn models:
    
    | Model            | predict_change_rate | accuracy_drop | probability_shift | decision_function_shift |
    |------------------|---------------------|---------------|-------------------|-------------------------|
    | SVC              | ✓                   | ✓             | ✓ (probability=True) | ✓                    |
    | LinearSVC        | ✓                   | ✓             | ✗                 | ✓                       |
    | RandomForest     | ✓                   | ✓             | ✓                 | ✗                       |
    | LogisticRegression| ✓                  | ✓             | ✓                 | ✓                       |
    | KNeighbors       | ✓                   | ✓             | ✓                 | ✗                       |
    | PLSRegression*   | ✓                   | ✓             | ✗                 | ✗                       |
    
    *PLSRegression for classification uses a threshold on continuous predict() output.
    
    Examples
    --------
    >>> # Example with REGRESSION (PLSRegression)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=pls_model,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     aim='regression',
    ...     metric='mean_abs_diff'
    ... )
    
    >>> # Example with CLASSIFICATION (SVC)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=svc_model,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     y_calclass=y_true,
    ...     aim='classification',
    ...     metric='prediction_change_rate'
    ... )
    
    >>> # Example with probability_shift (SVC with probability=True)
    >>> svc_proba = SVC(kernel='rbf', probability=True)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=svc_proba,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     aim='classification',
    ...     metric='probability_shift'
    ... )
    """
    # INPUT VALIDATION
    
    # Verify that the estimator has a predict method
    if not hasattr(estimator, 'predict'):
        # Raise error if the model lacks a predict method
        raise ValueError(f"The estimator must have a predict() method. Type: {type(estimator)}")
    
    # Verify that folds_struct is a dictionary
    if not isinstance(folds_struct, dict):
        # Raise error if the fold structure is not a dictionary
        raise TypeError("folds_struct must be a dictionary")
    
    # Verify required columns in predicates_df
    required_cols = ['rule', 'zone']  # Minimum required columns
    missing_cols = [c for c in required_cols if c not in predicates_df.columns]
    if missing_cols:
        # Raise error if any required column is missing
        raise KeyError(f"Missing columns in predicates_df: {missing_cols}")
    
    # Validate aim
    valid_aims = {'regression', 'classification'}
    if aim not in valid_aims:
        raise ValueError(f"aim must be one of {valid_aims}. Received: {aim}")
    
    # Define valid metrics for each aim
    regression_metrics = {'mean_abs_diff', 'mean_diff', 'mean_relative_dev'}
    classification_metrics = {
        'prediction_change_rate', 
        'accuracy_drop', 
        'f1_drop',
        'probability_shift', 
        'decision_function_shift'
    }
    
    # Validate metric according to aim
    if aim == 'regression':
        if metric not in regression_metrics:
            raise ValueError(
                f"For aim='regression', metric must be one of {regression_metrics}. "
                f"Received: '{metric}'"
            )
    else:  # classification
        if metric not in classification_metrics:
            raise ValueError(
                f"For aim='classification', metric must be one of {classification_metrics}. "
                f"Received: '{metric}'"
            )
        
        # Verify specific requirements for each classification metric
        if metric == 'probability_shift':
            if not hasattr(estimator, 'predict_proba'):
                raise ValueError(
                    f"The 'probability_shift' metric requires an estimator with predict_proba(). "
                    f"Received type: {type(estimator)}. "
                    f"Hint: for SVC, use SVC(probability=True)"
                )
        
        if metric == 'decision_function_shift':
            if not hasattr(estimator, 'decision_function'):
                raise ValueError(
                    f"The 'decision_function_shift' metric requires an estimator with decision_function(). "
                    f"Received type: {type(estimator)}. "
                    f"Compatible models: SVC, LinearSVC, LogisticRegression, etc."
                )
        
        if metric in ['accuracy_drop', 'f1_drop']:
            if y_calclass is None:
                raise ValueError(
                    f"The '{metric}' metric requires y_calclass (true labels). "
                    f"Provide y_calclass as a parameter."
                )
    
    # Convert y_calclass to Series if necessary
    if y_calclass is not None:
        if isinstance(y_calclass, np.ndarray):
            y_calclass = pd.Series(y_calclass)
    
    # INITIALIZATION
    
    # Dictionary to store final results (compatible with existing pipeline)
    metrics_results_dict = {}
    
    # Dictionary to store detailed results
    detailed_results = {}
    
    # Metric column name in the output DataFrame
    metric_name = 'Perturbation'
    
    # Counters for statistics
    total_folds = len(folds_struct)  # Total folds to process
    total_predicates_processed = 0   # Counter of processed predicates
    total_predicates_skipped = 0     # Counter of skipped predicates
    
    # Validate perturbation_mode
    valid_modes = {'constant', 'mean', 'median', 'min', 'max'}
    if perturbation_mode not in valid_modes:
        raise ValueError(f"perturbation_mode must be one of {valid_modes}. Received: {perturbation_mode}")
    
    # Validate stats_source
    valid_sources = {'full', 'predicate'}
    if stats_source not in valid_sources:
        raise ValueError(f"stats_source must be one of {valid_sources}. Received: {stats_source}")
    
    # Initial log if verbose
    if verbose:
        print("=" * 70)
        print("PERTURBATION IMPORTANCE FOR PREDICATES")
        print("=" * 70)
        print(f"Task type (aim): {aim}")
        print(f"Perturbation mode: {perturbation_mode}")
        if perturbation_mode == 'constant':
            print(f"Perturbation value: {perturbation_value}")
        else:
            print(f"Statistics source: {stats_source}")
        print(f"Metric: {metric}")
        print(f"Total folds: {total_folds}")
        if aim == 'classification' and y_calclass is not None:
            print(f"Classes in y_calclass: {y_calclass.unique().tolist()}")
        print()
    
    # MAIN LOOP: PROCESS EACH FOLD
    
    # Iterate over each fold in the structure
    for fold_idx, (fold_name, predicates_dict) in enumerate(folds_struct.items()):
        
        # Log current fold
        if verbose:
            print(f"\n[{fold_name}] Processing {len(predicates_dict)} predicates...")
        
        # Check if the fold is empty
        if len(predicates_dict) == 0:
            # If empty, create an empty DataFrame and skip to the next fold
            if verbose:
                print(f"  EMPTY - skipping")
            metrics_results_dict[fold_name] = pd.DataFrame({
                'Predicate': [],
                metric_name: []
            })
            continue
        
        # Temporary dictionary for metrics of this fold
        fold_metrics = {}
        
        # Temporary dictionary for detailed results of this fold
        fold_detailed = {}
        
        # LOOP: PROCESS EACH PREDICATE IN THE FOLD
        
        # Iterate over each predicate in the fold
        for pred_rule, df_info in predicates_dict.items():
            
            # Increment processed predicates counter
            total_predicates_processed += 1
            
            # 1. OBTAIN SAMPLE INDICES OF THE PREDICATE
            
            # Extract indices of samples belonging to this predicate
            sample_indices = df_info['Sample_Index'].values.tolist()
            
            # Number of samples in the predicate
            n_samples = len(sample_indices)
            
            # Log current predicate
            if verbose:
                print(f"  Predicate: {pred_rule} (n={n_samples})")
            
            # 2. CHECK EDGE CASES
            
            # If there are no samples, importance cannot be computed
            if n_samples == 0:
                if verbose:
                    print(f"    SKIP: n_samples=0 (no samples)")
                # Assign zero importance
                fold_metrics[pred_rule] = 0.0
                # Save details
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'n_samples = 0'
                }
                # Increment skip counter
                total_predicates_skipped += 1
                continue
            
            # 3. OBTAIN SPECTRAL ZONE INFORMATION
            
            # Attempt to retrieve the spectral zone columns of the predicate
            try:
                # Use auxiliary function to obtain zone columns
                zone_cols = get_zone_columns_from_predicate(
                    predicate_rule=pred_rule,
                    predicates_df=predicates_df,
                    spectral_cuts=spectral_cuts,
                    Xcal_columns=Xcalclass_prep.columns
                )
            except (KeyError, ValueError) as e:
                # If error when obtaining zone, assign zero importance
                if verbose:
                    print(f"    ERROR retrieving zone: {e}")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': str(e)
                }
                total_predicates_skipped += 1
                continue
            
            # Check if the zone has columns
            if len(zone_cols) == 0:
                # If zone is empty, assign zero importance
                if verbose:
                    print(f"    SKIP: empty spectral zone")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'empty zone'
                }
                total_predicates_skipped += 1
                continue
            
            # Log zone columns
            if verbose:
                print(f"    Zone: {len(zone_cols)} columns")
            
            # 4. OBTAIN ZONE BOUNDARIES FOR PERTURBATION
            
            # Find the zone name associated with the predicate
            mask_pred = predicates_df['rule'] == pred_rule
            zone_name = predicates_df.loc[mask_pred, 'zone'].values[0]
            
            # Find zone boundaries (start, end) in spectral_cuts
            zone_start, zone_end = None, None
            for cut in spectral_cuts:
                # Extract name and boundaries from the cut
                if len(cut) == 3:
                    name, start, end = cut
                elif len(cut) == 2:
                    start, end = cut
                    name = f"{start}-{end}"
                else:
                    continue
                
                # Check if it is the correct zone
                if name == zone_name:
                    zone_start, zone_end = float(start), float(end)
                    break
            
            # If boundaries were not found, skip
            if zone_start is None or zone_end is None:
                if verbose:
                    print(f"    SKIP: zone boundaries not found")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': zone_cols,
                    'skip_reason': 'boundaries not found'
                }
                total_predicates_skipped += 1
                continue
            
            # 5. EXTRACT PREDICATE SAMPLE DATA
            
            # Extract data subset for predicate samples
            X_eval = Xcalclass_prep.iloc[sample_indices].copy()
            
            # Extract true labels if available (for classification metrics)
            if y_calclass is not None:
                y_true_eval = y_calclass.iloc[sample_indices]
            else:
                y_true_eval = None
            
            # 6. PERTURB SPECTRAL ZONE
            
            # Create a copy of the data for perturbation
            X_perturbed = X_eval.copy()
            
            # Apply perturbation according to the chosen mode
            if perturbation_mode == 'constant':
                # Original behavior: fixed value for all columns
                X_perturbed[zone_cols] = perturbation_value
            else:
                # Compute column-wise statistics
                # Choose data source for statistics
                if stats_source == 'full':
                    stats_data = Xcalclass_prep[zone_cols]
                else:  # 'predicate'
                    stats_data = X_eval[zone_cols]
                
                # Compute statistic according to the mode
                if perturbation_mode == 'mean':
                    col_stats = stats_data.mean(axis=0)
                elif perturbation_mode == 'median':
                    col_stats = stats_data.median(axis=0)
                elif perturbation_mode == 'min':
                    col_stats = stats_data.min(axis=0)
                elif perturbation_mode == 'max':
                    col_stats = stats_data.max(axis=0)
                
                # Replace each column by its statistic
                for col in zone_cols:
                    X_perturbed[col] = col_stats[col]
            
            # 7. COMPUTE IMPORTANCE BASED ON AIM AND SELECTED METRIC
            
            if aim == 'regression':
                # REGRESSION MODE: uses predict() and continuous numerical metrics
                
                # Perform prediction with original data
                y_pred_original = estimator.predict(X_eval)
                y_pred_original = np.array(y_pred_original).flatten()
                
                # Perform prediction with perturbed data
                y_pred_perturbed = estimator.predict(X_perturbed)
                y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                
                # Compute importance according to the selected metric
                if metric == 'mean_abs_diff':
                    # Mean absolute difference between predictions
                    importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
                elif metric == 'mean_diff':
                    # Mean difference (signed)
                    importance = np.mean(y_pred_original - y_pred_perturbed)
                elif metric == 'mean_relative_dev':
                    # Mean relative deviation (caution with division by zero)
                    y_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
                    rel_dev = (y_pred_perturbed - y_pred_original) / y_safe
                    importance = np.nanmean(rel_dev)
                
                # For ranking, use absolute value for signed metrics
                if metric in ['mean_diff', 'mean_relative_dev']:
                    importance_for_ranking = np.abs(importance)
                else:
                    importance_for_ranking = importance
                    
            else:  # aim == 'classification'
                # CLASSIFICATION MODE: uses predict(), predict_proba(), or decision_function()
                
                if metric == 'prediction_change_rate':
                    # -----------------------------------------------------------------
                    # PREDICTION CHANGE RATE: proportion of samples that changed class
                    # Uses: estimator.predict()
                    # -----------------------------------------------------------------
                    
                    # Perform prediction with original data
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Perform prediction with perturbed data
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Compute proportion of samples that changed class
                    # Values range from 0 to 1, where 1 = all samples changed
                    importance = np.mean(y_pred_original != y_pred_perturbed)
                    importance_for_ranking = importance
                    
                elif metric == 'accuracy_drop':
                    # -----------------------------------------------------------------
                    # ACCURACY DROP: decrease in accuracy after perturbation
                    # Uses: estimator.predict() + y_true
                    # -----------------------------------------------------------------
                    
                    # Perform prediction with original data
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Perform prediction with perturbed data
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Compute accuracy before and after perturbation
                    acc_original = accuracy_score(y_true_eval, y_pred_original)
                    acc_perturbed = accuracy_score(y_true_eval, y_pred_perturbed)
                    
                    # Accuracy drop (positive value indicates degradation)
                    importance = acc_original - acc_perturbed
                    importance_for_ranking = np.abs(importance)
                    
                elif metric == 'f1_drop':
                    # -----------------------------------------------------------------
                    # F1 DROP: decrease in F1-score after perturbation
                    # Uses: estimator.predict() + y_true
                    # -----------------------------------------------------------------
                    
                    # Perform prediction with original data
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Perform prediction with perturbed data
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Compute F1-score before and after perturbation
                    # Uses average='weighted' to support multiclass scenarios
                    f1_original = f1_score(y_true_eval, y_pred_original, average='weighted')
                    f1_perturbed = f1_score(y_true_eval, y_pred_perturbed, average='weighted')
                    
                    # F1-score drop (positive value indicates degradation)
                    importance = f1_original - f1_perturbed
                    importance_for_ranking = np.abs(importance)
                    
                elif metric == 'probability_shift':
                    # -----------------------------------------------------------------
                    # PROBABILITY SHIFT: difference in predicted probabilities
                    # Uses: estimator.predict_proba()
                    # -----------------------------------------------------------------
                    
                    # Obtain original probabilities
                    prob_original = estimator.predict_proba(X_eval)
                    
                    # Obtain probabilities after perturbation
                    prob_perturbed = estimator.predict_proba(X_perturbed)
                    
                    # Compute difference in predicted probabilities
                    # IMPORTANT: For classification, predict_proba returns (n_samples, n_classes)
                    # where each row sums to 1.0. To avoid counting redundant changes,
                    # the change is computed PER SAMPLE (sum of absolute differences per row)
                    # and subsequently averaged across all samples.
                    #
                    # Binary example: [0.7, 0.3] -> [0.6, 0.4]
                    # - Without correction: mean(|0.7-0.6| + |0.3-0.4|) = mean(0.1 + 0.1) = 0.2 
                    # - With correction: mean(|0.7-0.6| + |0.3-0.4|) / 2 = 0.1
                    #
                    # For k classes, division by k normalizes the values to ensure
                    # comparability between binary and multiclass problems.
                    
                    n_classes = prob_original.shape[1]
                    
                    # Compute total shift per sample (sum over classes)
                    shift_per_sample = np.sum(np.abs(prob_original - prob_perturbed), axis=1)
                    
                    # Normalize by the number of classes (avoids counting redundant changes)
                    # Division by 2 accounts for the symmetry of probability changes summing to 1
                    shift_per_sample_normalized = shift_per_sample / 2.0
                    
                    # Average over all samples
                    importance = np.mean(shift_per_sample_normalized)
                    importance_for_ranking = importance
                    
                    # For verbose output, also save class predictions
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    
                elif metric == 'decision_function_shift':
                    # -----------------------------------------------------------------
                    # DECISION FUNCTION SHIFT: difference in decision function values
                    # Uses: estimator.decision_function()
                    # Applicable to SVM and linear models
                    # -----------------------------------------------------------------
                    
                    # Obtain original decision function values
                    df_original = estimator.decision_function(X_eval)
                    df_original = np.array(df_original)
                    
                    # Flatten if necessary (for binary classification)
                    if df_original.ndim == 1:
                        df_original = df_original.flatten()
                    
                    # Obtain decision function values after perturbation
                    df_perturbed = estimator.decision_function(X_perturbed)
                    df_perturbed = np.array(df_perturbed)
                    
                    # Flatten if necessary
                    if df_perturbed.ndim == 1:
                        df_perturbed = df_perturbed.flatten()
                    
                    # Compute mean absolute difference
                    importance = np.mean(np.abs(df_original - df_perturbed))
                    importance_for_ranking = importance
                    
                    # For verbose output, also save class predictions
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_perturbed = estimator.predict(X_perturbed)
            
            # 8. STORE RESULTS
            
            # Store importance for ranking
            fold_metrics[pred_rule] = importance_for_ranking
            
            # Save complete details
            fold_detailed[pred_rule] = {
                'importance': importance,
                'importance_abs': np.abs(importance) if isinstance(importance, (int, float)) else importance,
                'n_samples': n_samples,
                'zone_columns': zone_cols,
                'n_zone_features': len(zone_cols),
                'zone_name': zone_name,
                'zone_start': zone_start,
                'zone_end': zone_end,
                'perturbation_mode': perturbation_mode,
                'stats_source': stats_source if perturbation_mode != 'constant' else None,
                'aim': aim,
                'metric': metric
            }
            
            # Log computed importance
            if verbose:
                print(f"    Importance: {importance:.6f}")
        
        # CONVERT TO DATAFRAME (compatible with existing pipeline)
        
        # Create DataFrame from metrics dictionary
        metrics_df = pd.DataFrame.from_dict(
            fold_metrics,
            orient='index',
            columns=[metric_name]
        )
        
        # Add predicate column
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        
        # Reset index to numeric index
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Sort in DESCENDING order (higher values = more important)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        
        # Reset index after sorting
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Store fold result
        metrics_results_dict[fold_name] = metrics_df
        
        # Store detailed fold results
        detailed_results[fold_name] = fold_detailed
    
    # FINAL SUMMARY
    
    # Print summary if verbose
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Task type (aim): {aim}")
        print(f"Metric used: {metric}")
        print(f"Folds processed: {total_folds}")
        print(f"Predicates processed: {total_predicates_processed}")
        print(f"Predicates skipped: {total_predicates_skipped}")
        print()
        # Display per-fold summary
        for fold_name, df in metrics_results_dict.items():
            # Skip special detailed results key
            if fold_name.startswith('__'):
                continue
            print(f"  {fold_name}: {len(df)} predicates")
    
    # SAVE DETAILED RESULTS (OPTIONAL)
    
    # If requested, create DataFrame with all details
    if save_detailed_results:
        # List to store rows for the detailed DataFrame
        detailed_rows = []
        
        # Iterate over folds and predicates
        for fold_name, fold_data in detailed_results.items():
            for pred_rule, pred_data in fold_data.items():
                # Append row with predicate information
                detailed_rows.append({
                    'fold': fold_name,
                    'predicate': pred_rule,
                    'importance': pred_data['importance'],
                    'importance_abs': pred_data.get('importance_abs', np.abs(pred_data['importance'])),
                    'n_samples': pred_data['n_samples'],
                    'n_zone_features': pred_data.get('n_zone_features', 0),
                    'zone_name': pred_data.get('zone_name', None),
                    'skip_reason': pred_data.get('skip_reason', None),
                    'aim': pred_data.get('aim', aim),
                    'metric': pred_data.get('metric', metric)
                })
        
        # Create detailed results DataFrame
        detailed_df = pd.DataFrame(detailed_rows)
        
        # Attach as a special key in the results dictionary
        metrics_results_dict['__detailed_perturbation_results__'] = detailed_df
    
    # Return dictionary with results
    return metrics_results_dict


def map_thresholds_to_natural(
    lrc_df,                    # DataFrame with Zone and Threshold as columns (preprocessed space)
    zone_sums_preprocessed,    # zone_sums_df (preprocessed)
    zone_sums_natural          # zone_sums_df_original (natural)
):
    """
    Maps thresholds from the preprocessed space to the natural space
    using the nearest sample as a reference.

    Returns:
        DataFrame with additional columns: 'Threshold_Natural', 'Sample_Index', 'Approximation_Error', 'Node', 'Operator', 'Node_Natural'
    """
    result_df = lrc_df.copy()

    natural_thresholds = []
    sample_indices = []
    approximation_errors = []
    node_natural_list = []

    for idx, row in result_df.iterrows():
        zone_name = row['Zone']
        threshold_val = row['Threshold']
        operator = row['Operator']
        node = row['Node']

        # Skip None values
        if zone_name is None or threshold_val is None or zone_name not in zone_sums_preprocessed.columns:
            natural_thresholds.append(None)
            sample_indices.append(None)
            approximation_errors.append(None)
            node_natural_list.append(None)
            continue

        threshold = float(threshold_val)

        # Find the index of the nearest sample in the preprocessed space
        zone_values_prep = zone_sums_preprocessed[zone_name]
        distances = (zone_values_prep - threshold).abs()
        closest_idx = distances.idxmin()  # index of the nearest sample

        # Retrieve the corresponding value in the natural space
        natural_value = zone_sums_natural.loc[closest_idx, zone_name]

        # Compute approximation error (in the preprocessed space)
        error = distances.loc[closest_idx]

        # Construct Node_Natural (e.g., "Zone > 0.123")
        if operator is not None and natural_value is not None:
            node_natural = f"{zone_name} {operator} {natural_value:.6f}"
        else:
            node_natural = None

        natural_thresholds.append(natural_value)
        sample_indices.append(closest_idx)
        approximation_errors.append(error)
        node_natural_list.append(node_natural)

    result_df['Threshold_Natural'] = natural_thresholds
    result_df['Reference_Sample_Index'] = sample_indices
    result_df['Approximation_Error'] = approximation_errors
    result_df['Node'] = lrc_df.get('Node')
    result_df['Operator'] = lrc_df.get('Operator')
    result_df['Node_Natural'] = node_natural_list

    return result_df

def aggregate_spectral_zones_pca(spectral_zones_dict):
    """
    Aggregates spectral zones using PCA with a single principal component.
    
    For each spectral zone, a PCA with one component is fitted to extract:
    - Scores: projection of information onto the direction of maximum variance
    - Loadings: weights of each variable on PC1
    - Mean: mean vector of the zone (for reconstruction)
    - Variance Explained: fraction of the variance captured by PC1
    
    Parameters
    ----------
    spectral_zones_dict : dict
        Dictionary returned by extract_spectral_zones.
        Keys = zone names, Values = DataFrames with spectral data.
    
    Returns
    -------
    scores_df : pd.DataFrame
        DataFrame with PC1 scores for each zone (samples x zones).
    pca_info_dict : dict
        Dictionary with PCA information for each zone:
        - 'loadings': PC1 loadings vector
        - 'mean': zone mean vector
        - 'variance_explained': fraction of variance explained
        - 'columns': original column names (for reconstruction)
    """
    from sklearn.decomposition import PCA
    import pandas as pd

    scores_dict = {}  # stores scores for each zone
    pca_info_dict = {}  # stores information for reconstruction
    
    for zone_name, zone_df in spectral_zones_dict.items():
        # 1: Data preparation
        X_zone = zone_df.values  # convert to numpy array
        
        # 2: Fit PCA with 1 component
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X_zone)  # PC1 scores (n_samples, 1)
        
        # 3: Extract information
        loadings = pca.components_[0]  # PC1 loadings (d_m,)
        mean_vector = pca.mean_  # mean vector (d_m,)
        variance_explained = pca.explained_variance_ratio_[0]  # variance fraction
        
        # 4: Storage
        scores_dict[zone_name] = scores.flatten()  # convert to 1D
        
        pca_info_dict[zone_name] = {
            'loadings': loadings,
            'mean': mean_vector,
            'variance_explained': variance_explained,
            'columns': zone_df.columns.tolist(),  # original column names
            'pca_model': pca  # complete PCA model (for future use)
        }
        
        # Informative log
        print(f"Zone '{zone_name}': VE = {variance_explained:.2%}, "
              f"dim = {len(loadings)} variables")
    
    # Create DataFrame with all scores
    scores_df = pd.DataFrame(scores_dict)
    
    return scores_df, pca_info_dict

def reconstruct_threshold_to_spectrum(threshold_value, zone_name, pca_info_dict):
    """
    Reconstructs a scalar threshold (in score space) to the original spectral
    space, yielding a multivariate "threshold spectrum".
    
    Mathematical formula:
        τ = mean + threshold_value * loadings
    
    Parameters
    ----------
    threshold_value : float
        Threshold value in the PC1 score space.
    zone_name : str
        Name of the spectral zone.
    pca_info_dict : dict
        Dictionary with PCA information (returned by aggregate_spectral_zones_pca).
    
    Returns
    -------
    threshold_spectrum : pd.Series
        Threshold spectrum with index = original energies/wavelengths.
    """
    import pandas as pd
    # Retrieve PCA information
    pca_info = pca_info_dict[zone_name]
    loadings = pca_info['loadings']
    mean_vector = pca_info['mean']
    columns = pca_info['columns']
    
    # Reconstruction: τ = mean + q * loadings
    threshold_spectrum = mean_vector + threshold_value * loadings
    
    # Convert to Series with original index
    threshold_spectrum = pd.Series(threshold_spectrum, index=columns, name=f'threshold_{threshold_value:.4f}')
    
    return threshold_spectrum

def extract_predicate_info(predicate_rule):
    """
    Extracts information from a predicate rule.
    
    Parameters
    ----------
    predicate_rule : str
        Rule in the format "zone_name <= threshold" or "zone_name > threshold"
    
    Returns
    -------
    dict : {'zone': str, 'operator': str, 'threshold': float}
    """
    if '<=' in predicate_rule:
        parts = predicate_rule.split('<=')
        operator = '<='
    elif '>' in predicate_rule:
        parts = predicate_rule.split('>')
        operator = '>'
    else:
        raise ValueError(f"Unrecognized operator in: {predicate_rule}")
    
    zone_name = parts[0].strip()
    threshold_value = float(parts[1].strip())
    
    return {
        'zone': zone_name,
        'operator': operator,
        'threshold': threshold_value
    }

def extract_zone_from_predicate(predicate_rule):
    """
    Extracts the spectral zone name from a predicate rule.
    
    Parameters
    ----------
    predicate_rule : str
        Rule in the format "zone_name <= threshold" or "zone_name > threshold"
    
    Returns
    -------
    str : Spectral zone name
    
    Examples
    --------
    >>> extract_zone_from_predicate("Ca ka <= 25.5")
    'Ca ka'
    >>> extract_zone_from_predicate("Fe ka > 10.2")
    'Fe ka'
    """
    if '<=' in predicate_rule:
        return predicate_rule.split('<=')[0].strip()
    elif '>' in predicate_rule:
        return predicate_rule.split('>')[0].strip()
    else:
        raise ValueError(f"Unrecognized operator in: {predicate_rule}")


def build_predicate_graph(bags_result, predicate_ranking_dict, 
                            metric_column='Cov',
                            random_state=42, show_details=True,
                            var_exp=False, pca_info_dict=None):
    """
    Constructs a directed graph of predicates where edge weights are based
    on the Covariance (or another metric) of the SOURCE predicate.
    
    Parameters
    ----------
    - **bags_result** : dict
        Dictionary with predicate bags:
        {'Bag_1': {'Ca ka <= 25.5': DataFrame, ...}, 'Bag_2': {...}, ...}
        
    - **predicate_ranking_dict** : dict
        Dictionary with predicate rankings according to a metric for each bag:
        {'Bag_1': DataFrame(['Predicate', metric_column]), 'Bag_2': ...}
        
    - **metric_column** : str, default='Cov'
        Name of the column in predicate_ranking_dict containing the ranking metric.
        Provides flexibility to use 'Cov', 'Permutation', etc.
        
    - **random_state** : int, default=42
        Seed for random tie-breaking of bidirectional edges.
        
    - **show_details** : bool, default=True
        If True, prints details about bidirectional edge removal.
        
    - **var_exp** : bool, default=False
        If True, multiplies edge weights by the explained variance (PC1)
        of the spectral zone corresponding to the source predicate.
        
    - **pca_info_dict** : dict, optional
        Dictionary with PCA information for each zone (required if var_exp=True).
        Keys = zone names, Values = dict with 'variance_explained'.
    
    Returns
    -------
    - **DG** : nx.DiGraph
        Directed graph with weights based on the accumulated metric.

    """
    import networkx as nx
    import numpy as np
    import pandas as pd
    
    # Validate var_exp parameters
    if var_exp:
        if pca_info_dict is None:
            raise ValueError("pca_info_dict is required when var_exp=True")
    
    # Set seed for reproducibility in tie-breaking
    np.random.seed(random_state)
    
    # PHASE 1: GRAPH INITIALIZATION
    DG = nx.DiGraph()
    DG.add_node('Class_A', node_type='terminal', class_label='A')
    DG.add_node('Class_B', node_type='terminal', class_label='B')
    
    # PHASE 2: PATH CONSTRUCTION AND WEIGHT ACCUMULATION
    for bag_name, bag_predicates_dict in bags_result.items():
        
        # 2.1: Obtain the metric ranking for this bag
        predicate_ranking = predicate_ranking_dict[bag_name]
        ordered_predicates = predicate_ranking['Predicate'].tolist()
        
        # Filter only predicates that exist in this specific bag
        ordered_predicates = [p for p in ordered_predicates if p in bag_predicates_dict.keys()]
        
        if len(ordered_predicates) == 0:
            continue
        
        # 2.2: Create lookup dictionary for metric
        ranking_lookup = dict(zip(predicate_ranking['Predicate'], predicate_ranking[metric_column]))
        
        # 2.3: Construct edges between consecutive predicates
        for i in range(len(ordered_predicates) - 1):
            pred_current = ordered_predicates[i]
            pred_next = ordered_predicates[i + 1]
            
            DG.add_node(pred_current, node_type='predicate')
            DG.add_node(pred_next, node_type='predicate')
            
            ranking_value = float(ranking_lookup[pred_current]) # metric value of the SOURCE predicate
            
            # Weight by explained variance if var_exp=True
            if var_exp:
                zone_name = extract_zone_from_predicate(pred_current) # extract zone from current predicate
                if zone_name in pca_info_dict: # check if zone exists in PCA dictionary
                    ranking_value *= pca_info_dict[zone_name]['variance_explained'] # weight by zone VE
            
            # Accumulate weight if edge already exists
            if DG.has_edge(pred_current, pred_next):
                DG[pred_current][pred_next]['weight'] += ranking_value
            else:
                DG.add_edge(pred_current, pred_next, weight=ranking_value, bag=bag_name)
        
        # 2.4: Connect the LAST predicate to the terminal node
        last_pred = ordered_predicates[-1]
        DG.add_node(last_pred, node_type='predicate')
        
        df_last = bag_predicates_dict[last_pred]
        class_counts = df_last['Class_Predicted'].value_counts()
        majority_class = class_counts.idxmax()
        terminal_node = f'Class_{majority_class}'
        
        ranking_last_value = float(ranking_lookup[last_pred]) # metric value of the last predicate is the edge weight to the terminal
        
        if var_exp:
            zone_name = extract_zone_from_predicate(last_pred)
            if zone_name in pca_info_dict:
                ranking_last_value *= pca_info_dict[zone_name]['variance_explained']
        
        if DG.has_edge(last_pred, terminal_node):
            DG[last_pred][terminal_node]['weight'] += ranking_last_value
        else:
            DG.add_edge(last_pred, terminal_node, weight=ranking_last_value, bag=bag_name)

    # PHASE 3: IDENTIFICATION OF BIDIRECTIONAL EDGES
    bidirectional_pairs = []
    processed = set()
    
    for u, v in DG.edges():
        if DG.has_edge(v, u) and (v, u) not in processed:
            bidirectional_pairs.append({
                'node_A': u, 'node_B': v,
                'weight_A_to_B': float(DG[u][v]['weight']),
                'weight_B_to_A': float(DG[v][u]['weight'])
            })
            processed.add((u, v))
            processed.add((v, u))
    
    print(f"\nTotal bidirectional pairs found: {len(bidirectional_pairs)}")
    
    # PHASE 4: RESOLUTION OF BIDIRECTIONAL EDGES
    n_removed = 0
    
    for pair in bidirectional_pairs:
        u, v = pair['node_A'], pair['node_B']
        w_fwd, w_rev = pair['weight_A_to_B'], pair['weight_B_to_A']
        
        if w_fwd > w_rev:
            DG.remove_edge(v, u)
            if show_details:
                print(f"Removed: {v} -> {u} ({w_rev:.4f}) | Kept: {u} -> {v} ({w_fwd:.4f})")
        elif w_rev > w_fwd:
            DG.remove_edge(u, v)
            if show_details:
                print(f"Removed: {u} -> {v} ({w_fwd:.4f}) | Kept: {v} -> {u} ({w_rev:.4f})")
        else:
            # Tie: random selection
            if np.random.rand() > 0.5:
                DG.remove_edge(v, u)
                if show_details:
                    print(f"Tie! Removed: {v} -> {u} ({w_rev:.4f})")
            else:
                DG.remove_edge(u, v)
                if show_details:
                    print(f"Tie! Removed: {u} -> {v} ({w_fwd:.4f})")
        n_removed += 1

    # PHASE 5: FINAL SUMMARY
    print(f"\n{'='*70}")
    print("CONSTRUCTED GRAPH SUMMARY")
    print(f"{'='*70}")
    print(f"Initial edges: {DG.number_of_edges() + n_removed} | Removed: {n_removed}")
    print(f"Predicate nodes: {len([n for n, attr in DG.nodes(data=True) if attr['node_type'] == 'predicate'])}")
    print(f"Metric: {metric_column}")
    if var_exp:
        print("Weighting by explained variance: ENABLED")
    
    return DG