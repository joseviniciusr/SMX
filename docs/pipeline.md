# Pipeline overview

The `SMX` class wraps the full explanation workflow so it can be run with a
single `fit()` call. It orchestrates:

1. Spectral zone extraction
2. PCA aggregation of each zone
3. Quantile predicate generation
4. Bagging over samples (and predicates if requested)
5. Metric-based predicate ranking
6. Directed predicate graph construction
7. Local Reaching Centrality (LRC) ranking
8. Optional mapping back to natural spectral units

## Core inputs

- `spectral_cuts`: zone definitions for the spectral axis
- `quantiles`: thresholds used to build predicates per zone
- `X_cal_prep`: preprocessed calibration spectra
- `y_pred_cal`: continuous model outputs aligned with `X_cal_prep`
- `X_cal_natural`: original (unpreprocessed) calibration spectra

## Key outputs

After `fit()`, the most used attributes are:

- `lrc_summed_`: mean LRC ranking across seeds
- `lrc_summed_unique_`: one row per zone, sorted by LRC
- `lrc_natural_`: thresholds mapped back to natural units
- `zone_scores_`: PCA scores per zone
- `predicates_df_`: full predicate catalog
- `graphs_by_seed_`: per-seed predicate graphs

## Example

```python
from smx import SMX

smx = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.25, 0.5, 0.75],
    n_repetitions=4,
    n_bags=10,
    metric="perturbation",
    estimator=model,
)

smx.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal)
print(smx.lrc_summed_unique_.head())
```
