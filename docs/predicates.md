# Predicates and bagging

SMX builds logical predicates from zone scores and uses bagging to stabilize
importance rankings across subsamples.

## Predicate generation

`PredicateGenerator` creates two predicates per quantile per zone:

- `zone <= threshold`
- `zone > threshold`

```python
from smx import PredicateGenerator

generator = PredicateGenerator(quantiles=[0.25, 0.5, 0.75])
generator.fit(zone_scores)

predicates_df = generator.predicates_df_
indicator_df = generator.indicator_df_
```

## Bagging

`PredicateBagger` subsamples rows (and optionally predicates) to build bags
that feed the metric computations:

```python
from smx import PredicateBagger

bagger = PredicateBagger(n_bags=10, n_samples_fraction=0.8, replace=False, random_seed=42)
bags = bagger.run(zone_scores, y_pred_cal, predicates_df)
```

## Metrics

Two main metrics are provided:

- `CovarianceMetric`: covariance or mutual information between zone values and predictions
- `PerturbationMetric`: replace a zone and measure prediction shift

```python
from smx import CovarianceMetric, PerturbationMetric

cov_metric = CovarianceMetric(metric="covariance", threshold=0.01)
rankings = cov_metric.compute(bags)

pert_metric = PerturbationMetric(
    estimator=model,
    Xcalclass_prep=X_cal_prep,
    predicates_df=predicates_df,
    spectral_cuts=spectral_cuts,
    perturbation_mode="median",
    metric="probability_shift",
)
rankings = pert_metric.compute(bags)
```
