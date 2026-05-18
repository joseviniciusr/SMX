# Predicate graph and LRC

SMX turns ranked predicates into a directed graph and uses Local Reaching
Centrality (LRC) to summarize global importance.

## Build a graph

```python
from smx import PredicateGraphBuilder

builder = PredicateGraphBuilder(var_exp=True, pca_info_dict=aggregator.pca_info_)
graph = builder.build(bags, rankings, metric_column="Perturbation")
```

## Compute LRC

```python
from smx import compute_lrc, aggregate_lrc_across_seeds

lrc_df = compute_lrc(graph, predicates_df)
```

When multiple seeds are used, aggregate their LRC rankings:

```python
lrc_summed, lrc_unique = aggregate_lrc_across_seeds(lrc_by_seed, random_seeds)
```

## Map thresholds back to natural units

When you provide `X_cal_natural` to `SMX.fit()`, SMX maps the thresholds back
to natural spectral units:

```python
from smx import map_thresholds_to_natural

lrc_natural = map_thresholds_to_natural(
    lrc_df=lrc_summed,
    zone_sums_preprocessed=zone_scores,
    zone_sums_natural=zone_scores_natural,
)
```

You can also reconstruct the full threshold spectrum for a predicate:

```python
from smx import reconstruct_threshold_to_spectrum

spectrum = reconstruct_threshold_to_spectrum(
    threshold_value=0.42,
    zone_name="F3",
    pca_info_dict=aggregator.pca_info_,
)
```
