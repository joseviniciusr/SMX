"""
SMX — Spectral Model Explanation
=================================
Public API for the SMX core algorithm library.

Typical usage
-------------
>>> import smx
>>> zones = smx.extract_spectral_zones(Xcal, cuts)
>>> agg = smx.ZoneAggregator(method='pca')
>>> scores_df = agg.fit_transform(zones)
>>> gen = smx.PredicateGenerator(quantiles=[0.25, 0.5, 0.75])
>>> gen.fit(scores_df)
>>> bagger = smx.PredicateBagger()
>>> bags = bagger.run(scores_df, y_pred, gen.predicates_df_)
>>> metric = smx.CovarianceMetric(threshold=0.01)
>>> rankings = metric.compute(bags)
>>> builder = smx.PredicateGraphBuilder()
>>> graph = builder.build(bags, rankings, metric_column='Covariance')
>>> lrc_df = smx.compute_lrc(graph, gen.predicates_df_)
"""

from smx._version import __version__  # noqa: F401

from smx.pipeline import SMX
from smx.zones.extraction import extract_spectral_zones
from smx.zones.aggregation import ZoneAggregator
from smx.predicates.generation import PredicateGenerator
from smx.predicates.bagging import PredicateBagger
from smx.predicates.metrics import (
    BasePredicateMetric,
    CovarianceMetric,
    PerturbationMetric,
)
from smx.graph.builder import PredicateGraphBuilder
from smx.graph.centrality import compute_lrc, aggregate_lrc_across_seeds
from smx.graph.interpretation import (
    map_thresholds_to_natural,
    reconstruct_threshold_to_spectrum,
    extract_predicate_info,
)
from smx.datasets.synthetic import generate_synthetic_spectral_data

__all__ = [
    "__version__",
    # pipeline (high-level facade)
    "SMX",
    # zones
    "extract_spectral_zones",
    "ZoneAggregator",
    # predicates
    "PredicateGenerator",
    "PredicateBagger",
    # metrics
    "BasePredicateMetric",
    "CovarianceMetric",
    "PerturbationMetric",
    # graph
    "PredicateGraphBuilder",
    "compute_lrc",
    "aggregate_lrc_across_seeds",
    # interpretation
    "map_thresholds_to_natural",
    "reconstruct_threshold_to_spectrum",
    "extract_predicate_info",
    # datasets
    "generate_synthetic_spectral_data",
]
