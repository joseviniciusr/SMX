# SMX
Spectral Model eXplainer: a XAI tool for spectral-based machine learning models

## Plotting helpers

SMX includes Plotly-based visualization helpers for common explanation views.

### Zone ranking over spectrum

After fitting an `SMX` explainer, you can export the ranked spectral zones over
the reference spectrum:

```python
from smx import SMX

explainer = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.2, 0.4, 0.6, 0.8],
    seeds=[0, 1, 2, 3],
    metric="perturbation",
    estimator=model,
)
explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal_raw)

explainer.plot_zone_ranking_over_spectrum(
    "zone_ranking.html",
    ranking="unique",
)
```

You can also call the standalone plotting function:

```python
from smx import plot_zone_ranking_over_spectrum

plot_zone_ranking_over_spectrum(
    zone_ranking_df=explainer.lrc_summed_unique_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=explainer.zones_natural_,
    output_path="zone_ranking.html",
)
```
