# Predicate graph and LRC

SMX turns ranked predicates into a directed graph and uses Local Reaching
Centrality (LRC) to summarize global importance.

## Build a graph

```python
from smx import PredicateGraphBuilder

builder = PredicateGraphBuilder(var_exp=True, pca_info_dict=aggregator.pca_info_)
graph = builder.build(bags, rankings)
```

The rankings must have the columns `Predicate` and `Perturbation`. By default
there is one terminal node per class and the last predicate of each bag points
to the bag's majority predicted class. With `terminal_class="A"` the graph has
a single terminal `Class_A` reached by every bag; this is what the per-class
SMX uses (see below). `verbose=False` silences the summary print.

## Compute LRC

```python
from smx import compute_lrc, aggregate_lrc_across_seeds

lrc_df = compute_lrc(graph, predicates_df)
```

When multiple seeds are used, aggregate their LRC rankings:

```python
lrc, lrc_unique = aggregate_lrc_across_seeds(lrc_by_seed, random_seeds)
```

`compute_lrc` requires finite, non-negative edge weights and raises
`ValueError` otherwise (networkx would silently give every node LRC 0 when
a single weight is negative).

## Per-class graphs

With `SMX(..., per_class="within" | "all")`, `PerturbationMetric` adds one
column `Perturbation_<k>` per class to every bag ranking (the `Perturbation`
column and the row order are unchanged). For each class, the pipeline orders
each bag by that column (dropping NaN: below the `within` guard or a
non-positive value), builds a graph with `terminal_class=k`, and aggregates
the seeds with:

```python
from smx import aggregate_lrc_across_seeds_per_class

lrc_k, lrc_unique_k = aggregate_lrc_across_seeds_per_class(lrc_by_seed_k, seeds)
```

It drops terminal rows (one row per zone in `lrc_unique_k`) and reports
`N_Seeds`, the number of seeds in which each predicate appeared.

## Map thresholds back to natural units

When you provide `X_cal_natural` to `SMX.fit()`, SMX maps the thresholds back
to natural spectral units:

```python
from smx import map_thresholds_to_natural

lrc_natural = map_thresholds_to_natural(
    lrc_df=lrc,
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
