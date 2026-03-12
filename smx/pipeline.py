"""
SMXExplainer: high-level facade for the full SMX explanation pipeline.

This class internalises the seed-loop orchestration that every caller would
otherwise have to rewrite manually (zone extraction → predicate generation →
bagging → metric → graph → LRC → natural-scale mapping across N seeds).

Individual component classes (``ZoneAggregator``, ``PredicateGenerator``, etc.)
remain available for power users who need fine-grained control.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

from smx.zones.extraction import extract_spectral_zones
from smx.zones.aggregation import ZoneAggregator
from smx.predicates.generation import PredicateGenerator
from smx.predicates.bagging import PredicateBagger
from smx.predicates.metrics import CovarianceMetric, PerturbationMetric
from smx.graph.builder import PredicateGraphBuilder
from smx.graph.centrality import compute_lrc, aggregate_lrc_across_seeds
from smx.graph.interpretation import map_thresholds_to_natural

logger = logging.getLogger(__name__)

SpectralCuts = List[tuple]   # list of (name, start, end)


class SMXExplainer:
    """Full SMX explanation pipeline as a single fit/transform object.

    Runs zone extraction → PCA aggregation → predicate generation →
    seed-loop (bagging → metric → graph → LRC) → cross-seed aggregation →
    natural-scale threshold mapping.

    Parameters
    ----------
    spectral_cuts : list of (name, start, end) tuples
        Zone definitions, e.g. ``[("Low", 1.0, 4.0), ("High", 4.0, 10.0)]``.
    quantiles : list of float
        Quantile fractions for predicate generation, e.g. ``[0.25, 0.5, 0.75]``.
    seeds : list of int, default [0, 1, 2, 3]
        Random seeds for the bagging loop.
    n_bags : int, default 10
        Number of bags per seed.
    n_samples_fraction : float, default 0.8
        Fraction of calibration samples drawn per bag.
    min_samples_fraction : float, default 0.2
        Minimum fraction of samples satisfying a predicate for it to be
        included in a bag.
    replace : bool, default False
        Whether to sample bags with replacement.
    metric : {'covariance', 'perturbation'}, default 'perturbation'
        Importance metric to use.
    estimator : sklearn-compatible estimator, optional
        Trained model required when ``metric='perturbation'``.
    perturbation_mode : str, default 'median'
        Replacement strategy for perturbation (``'constant'``, ``'mean'``,
        ``'median'``, ``'min'``, ``'max'``).
    perturbation_metric : str, default 'probability_shift'
        Perturbation importance measure. Common values:
        ``'probability_shift'`` (classifiers with ``predict_proba``),
        ``'mean_abs_diff'`` (regressors).
    normalize_by_zone_size : bool, default True
        Divide raw perturbation importance by zone width.
    zone_size_exponent : float, default 1.0
        Exponent applied to zone size during normalisation.
    covariance_threshold : float, default 0.01
        Minimum covariance value to keep a predicate (covariance metric only).
    var_exp : bool, default False
        Weight graph edges by PC1 explained variance of the source zone.
    show_graph_details : bool, default False
        Print bidirectional-edge details during graph construction.
    class_threshold : float, default 0.5
        Decision boundary for ``Class_Predicted`` annotation on bags.

    Attributes (set after :meth:`fit`)
    ------------------------------------
    lrc_natural_ : pd.DataFrame
        Primary result — LRC with natural-scale thresholds.  Columns:
        ``Node``, ``Local_Reaching_Centrality``, ``Zone``, ``Threshold``,
        ``Operator``, ``Threshold_Natural``.
    lrc_summed_ : pd.DataFrame
        Mean-aggregated LRC across seeds, preprocessed-scale thresholds.
    lrc_summed_unique_ : pd.DataFrame
        Zone-deduplicated version of *lrc_summed_* (one row per zone).
    zone_scores_ : pd.DataFrame
        PCA zone scores on the preprocessed calibration data.
    predicates_df_ : pd.DataFrame
        Full predicate catalogue (generated from *zone_scores_*).
    pca_info_ : dict
        PCA info for the preprocessed zones.
    pca_info_natural_ : dict
        PCA info for the natural (unpreprocessed) zones.
    zones_natural_ : dict
        Raw spectral zone DataFrames from the unpreprocessed data.
    graphs_by_seed_ : dict[int, nx.DiGraph]
        Per-seed directed predicate graphs (useful for debugging).
    valid_seeds_ : list[int]
        Seeds that produced a non-empty graph (subset of *seeds*).
    """

    def __init__(
        self,
        spectral_cuts: SpectralCuts,
        quantiles: List[float],
        seeds: Optional[List[int]] = None,
        n_bags: int = 10,
        n_samples_fraction: float = 0.8,
        min_samples_fraction: float = 0.2,
        replace: bool = False,
        metric: Literal["covariance", "perturbation"] = "perturbation",
        estimator: Optional[Any] = None,
        perturbation_mode: str = "median",
        perturbation_metric: str = "probability_shift",
        normalize_by_zone_size: bool = True,
        zone_size_exponent: float = 1.0,
        covariance_threshold: float = 0.01,
        var_exp: bool = False,
        show_graph_details: bool = False,
        class_threshold: float = 0.5,
    ) -> None:
        if metric not in ("covariance", "perturbation"):
            raise ValueError(f"metric must be 'covariance' or 'perturbation', got '{metric}'.")
        if metric == "perturbation" and estimator is None:
            raise ValueError("estimator is required when metric='perturbation'.")

        self.spectral_cuts = spectral_cuts
        self.quantiles = quantiles
        self.seeds = seeds if seeds is not None else [0, 1, 2, 3]
        self.n_bags = n_bags
        self.n_samples_fraction = n_samples_fraction
        self.min_samples_fraction = min_samples_fraction
        self.replace = replace
        self.metric = metric
        self.estimator = estimator
        self.perturbation_mode = perturbation_mode
        self.perturbation_metric = perturbation_metric
        self.normalize_by_zone_size = normalize_by_zone_size
        self.zone_size_exponent = zone_size_exponent
        self.covariance_threshold = covariance_threshold
        self.var_exp = var_exp
        self.show_graph_details = show_graph_details
        self.class_threshold = class_threshold

        # Result attributes — populated by fit()
        self.lrc_natural_: Optional[pd.DataFrame] = None
        self.lrc_summed_: Optional[pd.DataFrame] = None
        self.lrc_summed_unique_: Optional[pd.DataFrame] = None
        self.zone_scores_: Optional[pd.DataFrame] = None
        self.predicates_df_: Optional[pd.DataFrame] = None
        self.pca_info_: Optional[Dict] = None
        self.pca_info_natural_: Optional[Dict] = None
        self.zones_natural_: Optional[Dict] = None
        self.graphs_by_seed_: Dict[int, nx.DiGraph] = {}
        self.valid_seeds_: List[int] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal_prep: pd.DataFrame,
        y_pred_cal: Union[pd.Series, np.ndarray],
        X_cal_natural: Optional[pd.DataFrame] = None,
    ) -> "SMXExplainer":
        """Run the full SMX explanation pipeline.

        Parameters
        ----------
        X_cal_prep : pd.DataFrame
            Pre-processed calibration spectra (samples × features).
        y_pred_cal : pd.Series or np.ndarray
            Continuous model predictions for the calibration set (aligned
            with *X_cal_prep*).
        X_cal_natural : pd.DataFrame, optional
            Unpreprocessed calibration spectra with the same shape as
            *X_cal_prep*.  Required for ``lrc_natural_`` threshold mapping.
            When ``None``, *X_cal_prep* is used as a fallback (thresholds
            will remain on the preprocessed scale).

        Returns
        -------
        self
        """
        y_pred = (
            pd.Series(y_pred_cal.values)
            if isinstance(y_pred_cal, pd.Series)
            else pd.Series(y_pred_cal)
        )
        n_cal = len(X_cal_prep)

        # ── Step 1: zone extraction + PCA aggregation ─────────────────────
        logger.debug("Extracting spectral zones…")
        zones_prep = extract_spectral_zones(X_cal_prep, self.spectral_cuts)

        aggregator = ZoneAggregator(method="pca")
        zone_scores = aggregator.fit_transform(zones_prep)
        pca_info = aggregator.pca_info_

        self.zone_scores_ = zone_scores
        self.pca_info_ = pca_info

        # ── Step 2: predicate generation ─────────────────────────────────
        logger.debug("Generating predicates…")
        gen = PredicateGenerator(quantiles=self.quantiles)
        gen.fit(zone_scores)
        predicates_df = gen.predicates_df_
        self.predicates_df_ = predicates_df

        n_samples_per_bag = max(1, int(n_cal * self.n_samples_fraction))
        min_samples_per_predicate = max(1, int(n_cal * self.min_samples_fraction))

        metric_column = "Covariance" if self.metric == "covariance" else "Perturbation"

        # ── Step 3: seed loop ────────────────────────────────────────────
        lrc_by_seed: Dict[int, pd.DataFrame] = {}
        graphs_by_seed: Dict[int, nx.DiGraph] = {}

        for seed in self.seeds:
            logger.debug("Seed %d — bagging…", seed)

            # 3a. Bagging
            bagger = PredicateBagger(
                n_bags=self.n_bags,
                n_samples_per_bag=n_samples_per_bag,
                min_samples_per_predicate=min_samples_per_predicate,
                replace=self.replace,
                sample_bagging=True,
                predicate_bagging=False,
                random_seed=seed,
            )
            bags = bagger.run(zone_scores, y_pred, predicates_df)

            # Annotate bags with discrete class prediction
            for pred_dict in bags.values():
                for df_info in pred_dict.values():
                    df_info["Class_Predicted"] = np.where(
                        df_info["Predicted_Y"] >= self.class_threshold, "A", "B"
                    )

            # 3b. Metric
            logger.debug("Seed %d — computing %s metric…", seed, self.metric)
            if self.metric == "covariance":
                metric_obj = CovarianceMetric(
                    metric="covariance",
                    threshold=self.covariance_threshold,
                )
            else:
                metric_obj = PerturbationMetric(
                    estimator=self.estimator,
                    Xcalclass_prep=X_cal_prep,
                    predicates_df=predicates_df,
                    spectral_cuts=self.spectral_cuts,
                    perturbation_mode=self.perturbation_mode,
                    stats_source="full",
                    metric=self.perturbation_metric,
                    normalize_by_zone_size=self.normalize_by_zone_size,
                    zone_size_exponent=self.zone_size_exponent,
                )
            rankings = metric_obj.compute(bags)

            # 3c. Graph
            logger.debug("Seed %d — building predicate graph…", seed)
            builder = PredicateGraphBuilder(
                random_state=seed,
                show_details=self.show_graph_details,
                var_exp=self.var_exp,
                pca_info_dict=pca_info if self.var_exp else None,
            )
            graph = builder.build(bags, rankings, metric_column=metric_column)
            graphs_by_seed[seed] = graph

            # 3d. LRC
            predicate_nodes = [
                n for n, attr in graph.nodes(data=True)
                if attr.get("node_type") == "predicate"
            ]
            if not predicate_nodes:
                logger.warning(
                    "Seed %d produced an empty graph (%s) — skipping.",
                    seed, self.metric,
                )
                continue

            lrc_df_seed = compute_lrc(graph, predicates_df)
            lrc_df_seed["Seed"] = seed
            lrc_by_seed[seed] = lrc_df_seed

        if not lrc_by_seed:
            raise RuntimeError(
                f"All seeds produced empty graphs for metric='{self.metric}'. "
                "The model predictions may be degenerate (e.g. all on one side)."
            )

        self.graphs_by_seed_ = graphs_by_seed
        self.valid_seeds_ = list(lrc_by_seed.keys())

        # ── Step 4: aggregate across seeds ───────────────────────────────
        logger.debug("Aggregating LRC across %d valid seeds…", len(self.valid_seeds_))
        lrc_summed, lrc_summed_unique = aggregate_lrc_across_seeds(
            lrc_by_seed, self.valid_seeds_
        )
        self.lrc_summed_ = lrc_summed
        self.lrc_summed_unique_ = lrc_summed_unique

        # ── Step 5: map thresholds to natural scale ───────────────────────
        X_natural = X_cal_natural if X_cal_natural is not None else X_cal_prep
        logger.debug("Mapping thresholds to natural scale…")
        zones_natural = extract_spectral_zones(X_natural, self.spectral_cuts)
        agg_natural = ZoneAggregator(method="pca")
        zone_scores_natural = agg_natural.fit_transform(zones_natural)

        self.zones_natural_ = zones_natural
        self.pca_info_natural_ = agg_natural.pca_info_

        self.lrc_natural_ = map_thresholds_to_natural(
            lrc_df=lrc_summed,
            zone_sums_preprocessed=zone_scores,
            zone_sums_natural=zone_scores_natural,
        )

        return self
