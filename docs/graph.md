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

Only predicates with a positive, finite weight (the ranking value, times the
zone's PC1 explained variance when `var_exp=True`) enter a bag's chain. A
predicate with weight 0 in a bag is left out of that bag's chain, so the
graph never holds a zero-weight edge (it would break the LRC of every node
upstream of it). The predicates left out are recorded on the graph:

- `graph.graph["zero_effect_predicates"]`: evaluated, but with weight 0 in
  every bag in which they appeared. `compute_lrc` reports them with LRC 0.
- `graph.graph["non_finite_predicates"]`: only non-finite values (not
  evaluated); they get no row.
- `graph.graph["n_zero_weight_rows"]`: number of predicate/bag rows left out
  because their weight was 0.

A zone that is constant in the calibration set has an undefined PC1
explained variance; `ZoneAggregator` sets it to 0 (with a warning), so with
`var_exp=True` its predicates have weight 0 and LRC 0.

## Compute LRC

```python
from smx import compute_lrc, aggregate_lrc_across_seeds

lrc_df = compute_lrc(graph, predicates_df)
```

When multiple seeds are used, aggregate their LRC rankings:

```python
lrc, lrc_unique = aggregate_lrc_across_seeds(lrc_by_seed, random_seeds)
```

`compute_lrc` requires finite, strictly positive edge weights and raises
`ValueError` otherwise. It appends the graph's `zero_effect_predicates` with
LRC 0, after the graph's own nodes. A graph without edges gives LRC 0 to
every node.

`aggregate_lrc_across_seeds` averages each predicate over the seeds in which
it has a row (a zero-effect predicate counts as 0 in that seed). The terminal
nodes (`Class_*`) are part of the graphs and of the per-seed tables, but they
are dropped from the aggregated tables, so `SMX.lrc_`, `SMX.lrc_unique_` and
`SMX.lrc_natural_` only hold predicates (one row per zone in `lrc_unique_`).

## Per-class graphs

With `SMX(..., per_class="within" | "all")`, `PerturbationMetric` adds one
column `Perturbation_<k>` per class to every bag ranking (the `Perturbation`
column and the row order are unchanged). For each class, the pipeline orders
each bag by that column, builds a graph with `terminal_class=k`, and
aggregates the seeds with the function below. A NaN value means *not
evaluated* for that class (below the `within` guard, or not finite) and the
predicate is dropped from that bag's ranking; a value of exactly 0 means
*evaluated, no effect*, and the predicate gets LRC 0, as in the global
ranking.

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
