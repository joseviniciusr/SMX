"""
SMX: high-level facade for the full SMX explanation pipeline.

This class internalises the seed-loop orchestration that every caller would
otherwise have to rewrite manually (zone extraction → predicate generation →
bagging → metric → graph → LRC → natural-scale mapping across N seeds).

Individual component classes (``ZoneAggregator``, ``PredicateGenerator``, etc.)
remain available for power users who need fine-grained control.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

from smx.zones.extraction import extract_spectral_zones
from smx.zones.aggregation import ZoneAggregator
from smx.predicates.generation import PredicateGenerator
from smx.predicates.bagging import PredicateBagger
from smx.predicates.metrics import PerturbationMetric
from smx.graph.builder import PredicateGraphBuilder
from smx.graph.centrality import (
    compute_lrc,
    aggregate_lrc_across_seeds,
    aggregate_lrc_across_seeds_per_class,
)
from smx.graph.interpretation import map_thresholds_to_natural
from smx.evaluation.faithfulness import progressive_masking_faithfulness
from smx.evaluation.normalized_faithfulness import (
    normalized_progressive_masking_faithfulness,
)

logger = logging.getLogger(__name__)

SpectralCuts = List[tuple]   # list of (name, start, end) or (name, start, end, group)

_PER_CLASS_MODES = ("within", "all")


def _resolve_per_class_mode(per_class: Union[bool, str, None]) -> Optional[str]:
    """``False``/``None`` → ``None``; ``True`` → ``'within'``; a mode → itself."""
    if per_class is None or per_class is False:
        return None
    if per_class is True:
        return "within"
    if isinstance(per_class, str) and per_class in _PER_CLASS_MODES:
        return per_class
    raise ValueError(
        f"per_class must be False, True, 'within' or 'all'; got {per_class!r}."
    )


def _validate_per_class_min_samples(value: Union[float, int]) -> None:
    if isinstance(value, bool):
        raise ValueError("per_class_min_samples must be a float in (0, 1] or an int >= 1, not a bool.")
    if isinstance(value, (int, np.integer)):
        if value < 1:
            raise ValueError(f"per_class_min_samples as an int must be >= 1; got {value}.")
        return
    if isinstance(value, (float, np.floating)):
        if not (0.0 < float(value) <= 1.0):
            raise ValueError(
                f"per_class_min_samples as a float must be in (0, 1]; got {value}."
            )
        return
    raise ValueError(
        f"per_class_min_samples must be a float in (0, 1] or an int >= 1; got {value!r}."
    )


class SMX:
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
    n_repetitions : int, default 4
        Number of independent bagging repetitions.  Seeds are generated as
        ``[0, 1, …, n_repetitions-1]``.
    n_bags : int, default 10
        Number of bags per seed.
    n_samples_fraction : float, default 0.8
        Fraction of calibration samples drawn per bag.  The minimum samples
        per predicate is hardcoded to 20 % of the dataset.
    replace : bool, default False
        Whether to sample bags with replacement.
    estimator : sklearn-compatible estimator, required
        Trained model with a ``predict()`` method (and ``predict_proba()``
        when using ``perturbation_metric='probability_shift'``). Supports
        both binary and multi-class classifiers natively.
    perturbation_mode : str, default 'median'
        Replacement strategy for perturbation (``'constant'``, ``'mean'``,
        ``'median'``, ``'min'``, ``'max'``).
    perturbation_value : float, default 0
        Constant replacement value used when ``perturbation_mode='constant'``.
    perturbation_metric : str, default 'probability_shift'
        Perturbation importance measure. Determines how the impact of
        spectral zone perturbation is quantified. Choice depends on the
        estimator type and the desired sensitivity:

        **Classification estimators** (with ``predict_proba``):

        - ``'probability_shift'`` — Mean total variation distance between
          pre- and post-perturbation class probabilities. Sensitive to
          confidence changes across all classes. Works natively with any
          number of classes K ≥ 2 via the Total Variation Distance. Requires
          ``predict_proba()``.
        - ``'prediction_change_rate'`` — Fraction of samples whose predicted
          class label changes after perturbation.
        - ``'accuracy_drop'`` — Drop in accuracy when perturbed predictions
          are compared to original predictions.
        - ``'f1_drop'`` — Weighted F1-score decrease after perturbation.
        - ``'decision_function_shift'`` — Mean absolute change in decision
          function values (e.g. signed distances from hyperplane for SVC).
          Requires ``decision_function()``.

        **Regression estimators** (with ``predict`` returning continuous values):

        - ``'mean_abs_diff'`` — Mean absolute difference between original
          and perturbed predictions.
        - ``'mean_diff'`` — Mean signed difference (bias direction). Positive
          values indicate perturbation increases predictions, negative decreases.
        - ``'mean_relative_dev'`` — Mean relative deviation, normalized by
          original prediction magnitude. Treats zero predictions as NaN.
    normalize_by_zone_size : bool, default True
        Divide raw perturbation importance by zone width.
    zone_size_exponent : float, default 1.0
        Exponent applied to zone size during normalisation.
    var_exp : bool, default True
        Weight graph edges by PC1 explained variance of the source zone.
    show_graph_details : bool, default False
        Print bidirectional-edge details during graph construction.
    per_class : bool or {'within', 'all'}, default False
        Also produce one zone ranking **per class** (``True`` ≡ ``'within'``).
        The class enters through the ranking metric: for each class *k* the
        importance of a predicate is the mean of ``|Δ output_k|`` (same
        perturbation, same predictions, no extra model calls), and a graph
        with a single terminal ``Class_<k>`` is built from the bags ordered
        by that metric (chain "B1b").

        - ``'within'``: the mean is taken over the predicate's samples
          **predicted** as *k* only. Recommended: it answers "what makes the
          model call these samples *k*".
        - ``'all'``: the mean is taken over all the predicate's samples
          (the global population). Mostly driven by the other classes'
          samples; with K = 2 it equals the global ranking.

        Only ``'probability_shift'`` and ``'mean_abs_diff'`` (with an
        ``(n_samples, K)`` output, e.g. PLS on one-hot targets) decompose by
        class; other metrics raise ``ValueError``. The global outputs are
        identical with or without *per_class*.
    per_class_min_samples : float or int, default 0.2
        ``'within'`` only (ignored by ``'all'``, whose population is already
        guarded by the bagger at 20 % of the dataset). Minimum number of
        *distinct* predicted-*k* samples a predicate must have in a bag to
        enter class *k*'s chain. A float in (0, 1] is a fraction of the
        number of calibration samples predicted as *k* (``max(1,
        int(f · n_k))``, mirroring the bagger's 20 % rule); an int ≥ 1 is an
        absolute count.
    per_class_output_labels : sequence, optional
        Class label of each model output column, in order. Defaults to
        ``estimator.classes_``; required for estimators without it (e.g. a
        ``PLSRegression`` used as a classifier).

    Notes
    -----
    *Per-class limitations.* The perturbation replaces a zone by the
    calibration median. In ``'within'`` mode a class whose typical zone value
    coincides with that median (a class defined by exclusion, or the middle
    class of an ordinal axis) shows little change and its relevant zones can
    be under-ranked; ``'all'`` captures them, at the price of mixing in the
    other classes' samples. Predicates with a non-positive per-class value
    are left out of that class's chain, whereas the global chain keeps
    zero-valued predicates.

    *Known limitation of the global output (not yet fixed).* With
    ``var_exp=False`` and a metric that can be exactly 0 (label-based
    metrics), :func:`smx.graph.centrality.compute_lrc` sets the LRC of nodes
    upstream of a zero-weight edge to 0.

    Attributes (set after :meth:`fit`)
    ------------------------------------
    lrc_natural\_ : pd.DataFrame or None
        LRC with natural-scale thresholds (available only when
        ``X_cal_natural`` is provided to :meth:`fit`). Columns:
        ``Node``, ``Local_Reaching_Centrality``, ``Zone``, ``Threshold``,
        ``Operator``, ``Threshold_Natural``.
    lrc\_ : pd.DataFrame
        Mean-aggregated LRC across seeds, preprocessed-scale thresholds. Only
        predicate rows: the terminal nodes (``Class_*``) stay in the graphs
        but not in the final tables (nor in ``lrc_natural_``).
    lrc_unique\_ : pd.DataFrame
        Zone-deduplicated version of *lrc_* (one row per zone).
    zone_scores\_ : pd.DataFrame
        PCA zone scores on the preprocessed calibration data.
    predicates_df\_ : pd.DataFrame
        Full predicate catalogue (generated from *zone_scores_*).
    pca_info\_ : dict
        PCA info for the preprocessed zones.
    pca_info_natural\_ : dict or None
        PCA info for the natural (unpreprocessed) zones (only when
        ``X_cal_natural`` is provided to :meth:`fit`).
    zones_natural\_ : dict or None
        Raw spectral zone DataFrames from the unpreprocessed data (only when
        ``X_cal_natural`` is provided to :meth:`fit`).
    graphs_by_seed\_ : dict[int, nx.DiGraph]
        Per-seed directed predicate graphs (useful for debugging).
    valid_seeds\_ : list[int]
        Seeds that produced a non-empty graph (subset of ``seeds``).
    faithfulness\_ : dict
        Progressive top-k masking evaluation summary produced by
        :meth:`evaluate_faithfulness`.
    per_class_mode\_ : {'within', 'all'} or None
        Resolved *per_class* (``None`` when disabled). All the per-class
        attributes below are ``None`` when it is ``None``.
    per_class_labels\_ : list of str
        Class labels of the per-class outputs (model output order).
    per_class_min_samples\_ : dict[str, int] or None
        Effective ``'within'`` guard per class; ``None`` in ``'all'`` mode.
    lrc_per_class\_ : dict[str, pd.DataFrame]
        Per class: mean LRC across seeds of every predicate (no terminal
        rows), with ``N_Seeds`` (seeds in which the predicate appeared).
        Classes without any valid seed are absent (a warning is logged).
    lrc_unique_per_class\_ : dict[str, pd.DataFrame]
        Per class: one row per zone (its highest-LRC predicate).
    lrc_natural_per_class\_ : dict[str, pd.DataFrame] or None
        *lrc_per_class_* with natural-scale thresholds (only with
        ``X_cal_natural``).
    graphs_by_class_by_seed\_ : dict[str, dict[int, nx.DiGraph]]
        Per-class graphs (single terminal ``Class_<k>``).
    valid_seeds_per_class\_ : dict[str, list[int]]
        Seeds that produced a usable graph for each class.
    per_class_coverage\_ : pd.DataFrame
        One row per (seed, class): bags, predicate/bag rows, rows evaluated
        and excluded (below the guard or non-positive), graph status.
    """

    def __init__(
        self,
        spectral_cuts: SpectralCuts,
        quantiles: List[float],
        n_repetitions: int = 4,
        n_bags: int = 10,
        n_samples_fraction: float = 0.8,
        replace: bool = False,
        estimator: Optional[Any] = None,
        perturbation_mode: str = "median",
        perturbation_value: float = 0,
        perturbation_metric: str = "probability_shift",
        perturbation_stats_source: str = "full",
        normalize_by_zone_size: bool = True,
        zone_size_exponent: float = 1.0,
        var_exp: bool = True,
        show_graph_details: bool = False,
        per_class: Union[bool, str] = False,
        per_class_min_samples: Union[float, int] = 0.2,
        per_class_output_labels: Optional[Sequence[Any]] = None,
    ) -> None:
        if estimator is None:
            raise ValueError(
                "estimator is required. Provide a trained sklearn-compatible model "
                "with predict() and (for probability_shift) predict_proba() methods."
            )
        per_class_mode = _resolve_per_class_mode(per_class)
        _validate_per_class_min_samples(per_class_min_samples)
        if per_class_mode is not None and perturbation_metric not in PerturbationMetric._PER_CLASS_METRICS:
            raise ValueError(
                f"per_class requires a perturbation_metric that decomposes by output column "
                f"({sorted(PerturbationMetric._PER_CLASS_METRICS)}); got '{perturbation_metric}'."
            )
        if per_class_output_labels is not None:
            labels = [str(c) for c in per_class_output_labels]
            if len(labels) == 0 or len(set(labels)) != len(labels):
                raise ValueError(
                    f"per_class_output_labels must be non-empty and unique; got {labels}."
                )

        self.spectral_cuts = spectral_cuts
        self.quantiles = quantiles
        self.n_repetitions = n_repetitions
        self.seeds = list(range(n_repetitions))
        self.n_bags = n_bags
        self.n_samples_fraction = n_samples_fraction
        self.replace = replace
        self.estimator = estimator
        self.perturbation_mode = perturbation_mode
        self.perturbation_value = perturbation_value
        self.perturbation_metric = perturbation_metric
        self.perturbation_stats_source = perturbation_stats_source
        self.normalize_by_zone_size = normalize_by_zone_size
        self.zone_size_exponent = zone_size_exponent
        self.var_exp = var_exp
        self.show_graph_details = show_graph_details
        self.per_class = per_class
        self.per_class_min_samples = per_class_min_samples
        self.per_class_output_labels = per_class_output_labels
        self.per_class_mode_: Optional[str] = per_class_mode

        # Result attributes — populated by fit()
        self.lrc_natural_: Optional[pd.DataFrame] = None
        self.lrc_: Optional[pd.DataFrame] = None
        self.lrc_unique_: Optional[pd.DataFrame] = None
        self.zone_scores_: Optional[pd.DataFrame] = None
        self.predicates_df_: Optional[pd.DataFrame] = None
        self.pca_info_: Optional[Dict] = None
        self.pca_info_natural_: Optional[Dict] = None
        self.zones_natural_: Optional[Dict] = None
        self.graphs_by_seed_: Dict[int, nx.DiGraph] = {}
        self.valid_seeds_: List[int] = []
        self.faithfulness_: Optional[Dict[str, Any]] = None
        self.normalized_faithfulness_: Optional[Dict[str, Any]] = None
        self._reset_per_class_results()

    def _reset_per_class_results(self) -> None:
        self.per_class_labels_: Optional[List[str]] = None
        self.per_class_min_samples_: Optional[Dict[str, int]] = None
        self.lrc_per_class_: Optional[Dict[str, pd.DataFrame]] = None
        self.lrc_unique_per_class_: Optional[Dict[str, pd.DataFrame]] = None
        self.lrc_natural_per_class_: Optional[Dict[str, pd.DataFrame]] = None
        self.graphs_by_class_by_seed_: Optional[Dict[str, Dict[int, nx.DiGraph]]] = None
        self.valid_seeds_per_class_: Optional[Dict[str, List[int]]] = None
        self.per_class_coverage_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal_prep: pd.DataFrame,
        y_class_labels: Union[pd.Series, np.ndarray, list],
        X_cal_natural: Optional[pd.DataFrame] = None,
    ) -> "SMX":
        """Run the full SMX explanation pipeline.

        Parameters
        ----------
        X_cal_prep : pd.DataFrame
            Pre-processed calibration spectra (samples × features).
        y_class_labels : pd.Series, np.ndarray, or list
            Class labels assigned by the trained estimator to each sample in the
            calibration set. Must be aligned row-wise with *X_cal_prep*. Accepts
            any hashable label type (integer class codes, string names, etc.).
            Typically obtained as ``estimator.predict(X_cal_prep)``. Must contain
            at least two distinct class labels.
        X_cal_natural : pd.DataFrame, optional
            Unpreprocessed calibration spectra with the same shape as
            *X_cal_prep*.  Required for ``lrc_natural_`` threshold mapping.
            When ``None``, the natural-scale mapping step is skipped and
            ``lrc_natural_``, ``zones_natural_``, and ``pca_info_natural_``
            remain ``None``.

        Returns
        -------
        self
        """
        # --- Validate and normalise class labels ---
        y_class_labels = pd.Series(y_class_labels).astype(str).reset_index(drop=True)

        if len(y_class_labels) != len(X_cal_prep):
            raise ValueError(
                f"y_class_labels length ({len(y_class_labels)}) must match "
                f"X_cal_prep length ({len(X_cal_prep)})."
            )

        unique_classes = sorted(y_class_labels.unique().tolist())

        if len(unique_classes) < 2:
            raise ValueError(
                f"y_class_labels must contain at least 2 distinct class labels. "
                f"Got {len(unique_classes)}: {unique_classes}."
            )
        n_cal = len(X_cal_prep)

        # ── Per-class set-up (labels of the model outputs + 'within' guard) ─
        self._reset_per_class_results()
        per_class_mode = self.per_class_mode_
        per_class_labels: List[str] = []
        per_class_guard: Optional[Dict[str, int]] = None
        if per_class_mode is not None:
            per_class_labels = self._resolve_output_labels()
            unknown = sorted(set(unique_classes) - set(per_class_labels))
            if unknown:
                raise ValueError(
                    f"y_class_labels contain {unknown}, which are not among the model "
                    f"output labels {per_class_labels} (from "
                    f"{'per_class_output_labels' if self.per_class_output_labels is not None else 'estimator.classes_'}). "
                    f"Labels are compared as strings."
                )
            if per_class_mode == "within":
                counts = y_class_labels.value_counts()
                per_class_guard = {
                    k: self._effective_guard(int(counts.get(k, 0))) for k in per_class_labels
                }
            self.per_class_labels_ = per_class_labels
            self.per_class_min_samples_ = per_class_guard
            per_class_lrc: Dict[str, Dict[int, pd.DataFrame]] = {k: {} for k in per_class_labels}
            per_class_graphs: Dict[str, Dict[int, nx.DiGraph]] = {k: {} for k in per_class_labels}
            coverage_rows: List[Dict[str, Any]] = []

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

        # ── Step 3: seed loop ────────────────────────────────────────────
        lrc_by_seed: Dict[int, pd.DataFrame] = {}
        graphs_by_seed: Dict[int, nx.DiGraph] = {}

        for seed in self.seeds:
            logger.debug("Seed %d — bagging…", seed)

            # 3a. Bagging
            bagger = PredicateBagger(
                n_bags=self.n_bags,
                n_samples_fraction=self.n_samples_fraction,
                replace=self.replace,
                sample_bagging=True,
                predicate_bagging=False,
                random_seed=seed,
            )
            bags = bagger.run(zone_scores, predicates_df)

            # Annotate each bag's predicate DataFrames with the true class label
            # for the samples they contain, derived from y_class_labels.
            for pred_dict in bags.values():
                for df_info in pred_dict.values():
                    sample_indices = df_info["Sample_Index"].values
                    df_info["Class_Predicted"] = y_class_labels.iloc[sample_indices].values

            # 3b. Metric (perturbation-based)
            logger.debug("Seed %d — computing perturbation metric…", seed)
            metric_obj = PerturbationMetric(
                estimator=self.estimator,
                Xcalclass_prep=X_cal_prep,
                predicates_df=predicates_df,
                spectral_cuts=self.spectral_cuts,
                perturbation_mode=self.perturbation_mode,
                perturbation_value=self.perturbation_value,
                stats_source=self.perturbation_stats_source,
                metric=self.perturbation_metric,
                normalize_by_zone_size=self.normalize_by_zone_size,
                zone_size_exponent=self.zone_size_exponent,
                per_class=per_class_mode,
                class_labels=per_class_labels or None,
                per_class_min_samples=per_class_guard,
            )
            rankings = metric_obj.compute(bags)

            # 3c. Graph
            logger.debug("Seed %d — building predicate graph…", seed)
            builder = PredicateGraphBuilder(
                class_labels=unique_classes,       # ← derived from y_class_labels
                random_state=seed,
                show_details=self.show_graph_details,
                var_exp=self.var_exp,
                pca_info_dict=pca_info if self.var_exp else None,
            )
            graph = builder.build(bags, rankings)  # metric_column removed
            graphs_by_seed[seed] = graph

            # 3e. Per-class graphs + LRC (independent of the global graph)
            if per_class_mode is not None:
                self._fit_per_class_seed(
                    seed, bags, rankings, predicates_df, pca_info,
                    metric_obj.per_class_columns,
                    per_class_lrc, per_class_graphs, coverage_rows,
                )

            # 3d. LRC
            predicate_nodes = [
                n for n, attr in graph.nodes(data=True)
                if attr.get("node_type") == "predicate"
            ]
            if len(predicate_nodes) < 1 or graph.number_of_nodes() < 2:
                logger.warning(
                    "Seed %d produced an undersized graph (nodes=%d, predicate_nodes=%d) — skipping.",
                    seed, graph.number_of_nodes(), len(predicate_nodes)
                )
                continue

            lrc_df_seed = compute_lrc(graph, predicates_df)
            if lrc_df_seed.empty:
                logger.warning(
                    "Seed %d produced an empty LRC table after graph processing — skipping.",
                    seed,
                )
                continue
            predicate_lrc = lrc_df_seed.loc[
                lrc_df_seed["Node"].isin(predicate_nodes), "Local_Reaching_Centrality"
            ]
            if len(predicate_lrc) and bool((predicate_lrc == 0).all()):
                logger.warning(
                    "Seed %d: every predicate has LRC 0 in the global graph (e.g. the "
                    "perturbation metric is zero everywhere) — the seed is kept, but its "
                    "ranking carries no information.",
                    seed,
                )
            lrc_df_seed["Seed"] = seed
            lrc_by_seed[seed] = lrc_df_seed

        if not lrc_by_seed:
            raise RuntimeError(
                "All seeds produced empty graphs. "
                "Check that y_class_labels contains enough samples per class and that "
                "the perturbation metric is compatible with the estimator."
            )

        self.graphs_by_seed_ = graphs_by_seed
        self.valid_seeds_ = list(lrc_by_seed.keys())

        # ── Step 4: aggregate across seeds ───────────────────────────────
        logger.debug("Aggregating LRC across %d valid seeds…", len(self.valid_seeds_))
        lrc, lrc_unique = aggregate_lrc_across_seeds(
            lrc_by_seed, self.valid_seeds_
        )
        self.lrc_ = lrc
        self.lrc_unique_ = lrc_unique

        # ── Step 4b: per-class aggregation ───────────────────────────────
        if per_class_mode is not None:
            self.graphs_by_class_by_seed_ = per_class_graphs
            self.valid_seeds_per_class_ = {}
            self.lrc_per_class_ = {}
            self.lrc_unique_per_class_ = {}
            for k in per_class_labels:
                valid = [s for s in self.seeds if s in per_class_lrc[k]]
                self.valid_seeds_per_class_[k] = valid
                if not valid:
                    logger.warning(
                        "Class %r produced no usable per-class graph in any seed — it is "
                        "absent from lrc_per_class_.",
                        k,
                    )
                    continue
                lrc_k, lrc_unique_k = aggregate_lrc_across_seeds_per_class(
                    per_class_lrc[k], valid
                )
                self.lrc_per_class_[k] = lrc_k
                self.lrc_unique_per_class_[k] = lrc_unique_k
            self.per_class_coverage_ = pd.DataFrame(
                coverage_rows,
                columns=["Seed", "Class", "N_Bags", "N_Predicate_Bag_Rows",
                         "N_Evaluated", "N_Excluded", "Graph_Status"],
            )

        # ── Step 5: map thresholds to natural scale (optional) ───────────
        if X_cal_natural is not None:
            logger.debug("Mapping thresholds to natural scale…")
            zones_natural = extract_spectral_zones(X_cal_natural, self.spectral_cuts)
            agg_natural = ZoneAggregator(method="pca")
            zone_scores_natural = agg_natural.fit_transform(zones_natural)

            self.zones_natural_ = zones_natural
            self.pca_info_natural_ = agg_natural.pca_info_

            self.lrc_natural_ = map_thresholds_to_natural(
                lrc_df=lrc,
                zone_sums_preprocessed=zone_scores,
                zone_sums_natural=zone_scores_natural,
            )
            if per_class_mode is not None:
                self.lrc_natural_per_class_ = {
                    k: map_thresholds_to_natural(
                        lrc_df=lrc_k,
                        zone_sums_preprocessed=zone_scores,
                        zone_sums_natural=zone_scores_natural,
                    )
                    for k, lrc_k in self.lrc_per_class_.items()
                }
        else:
            logger.info(
                "X_cal_natural was not provided to SMX.fit(); skipping natural-scale threshold mapping. "
                "Preprocessed-scale outputs (lrc_, lrc_unique_) remain available."
            )
            self.lrc_natural_ = None
            self.zones_natural_ = None
            self.pca_info_natural_ = None

        return self

    # ------------------------------------------------------------------
    # Per-class helpers
    # ------------------------------------------------------------------

    def _resolve_output_labels(self) -> List[str]:
        """Label of each model output column, as strings, in output order."""
        if self.per_class_output_labels is not None:
            return [str(c) for c in self.per_class_output_labels]
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            raise ValueError(
                "per_class needs the class label of each model output column: the "
                f"estimator ({type(self.estimator).__name__}) has no 'classes_', so pass "
                "per_class_output_labels=[...] (e.g. the labels of a PLS one-hot target, "
                "in column order)."
            )
        return [str(c) for c in classes]

    def _effective_guard(self, n_class: int) -> int:
        """'within' guard for a class with *n_class* predicted calibration samples."""
        value = self.per_class_min_samples
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        return max(1, int(n_class * float(value)))

    def _fit_per_class_seed(
        self,
        seed: int,
        bags: Dict[str, Dict[str, pd.DataFrame]],
        rankings: Dict[str, pd.DataFrame],
        predicates_df: pd.DataFrame,
        pca_info: Dict,
        class_columns: Dict[str, str],
        lrc_store: Dict[str, Dict[int, pd.DataFrame]],
        graph_store: Dict[str, Dict[int, nx.DiGraph]],
        coverage_rows: List[Dict[str, Any]],
    ) -> None:
        """Build the class graphs of one seed and store their LRC tables."""
        for label, column in class_columns.items():
            class_rankings: Dict[str, pd.DataFrame] = {}
            n_rows = n_evaluated = 0
            for bag_name, ranking_df in rankings.items():
                if bag_name.startswith("__"):
                    continue  # '__detailed_perturbation_results__'
                n_rows += len(ranking_df)
                ranked = (
                    ranking_df.loc[ranking_df[column].notna(), ["Predicate", column]]
                    .rename(columns={column: "Perturbation"})
                    .sort_values("Perturbation", ascending=False, kind="mergesort")
                    .reset_index(drop=True)
                )
                n_evaluated += len(ranked)
                class_rankings[bag_name] = ranked

            builder = PredicateGraphBuilder(
                terminal_class=label,
                random_state=seed,
                show_details=False,
                var_exp=self.var_exp,
                pca_info_dict=pca_info if self.var_exp else None,
                verbose=False,
            )
            graph = builder.build(bags, class_rankings)
            graph_store[label][seed] = graph

            predicate_nodes = [
                n for n, attr in graph.nodes(data=True) if attr.get("node_type") == "predicate"
            ]
            status = "ok"
            if len(predicate_nodes) < 1 or graph.number_of_nodes() < 2:
                status = "undersized"
            else:
                lrc_df = compute_lrc(graph, predicates_df, verbose=False)
                predicate_lrc = lrc_df.loc[
                    lrc_df["Node"].isin(predicate_nodes), "Local_Reaching_Centrality"
                ]
                if not bool((predicate_lrc > 0).any()):
                    status = "all_zero_lrc"
                else:
                    lrc_df["Seed"] = seed
                    lrc_store[label][seed] = lrc_df
            if status != "ok":
                logger.warning(
                    "Seed %d, class %r: per-class graph is %s — excluded from the "
                    "aggregation of that class.",
                    seed, label, status,
                )
            coverage_rows.append({
                "Seed": seed,
                "Class": label,
                "N_Bags": len(class_rankings),
                "N_Predicate_Bag_Rows": n_rows,
                "N_Evaluated": n_evaluated,
                "N_Excluded": n_rows - n_evaluated,
                "Graph_Status": status,
            })

    def evaluate_faithfulness(
        self,
        X_eval: pd.DataFrame,
        *,
        ranking: Literal["unique", "summed", "natural"] = "unique",
        X_reference: Optional[pd.DataFrame] = None,
        metric: Literal[
            "auto",
            "probability_shift",
            "mean_abs_diff",
            "decision_function_shift",
            "calibration_invariant",
        ] = "auto",
        masking_strategy: Literal["zero", "constant", "mean", "median", "min", "max"] = "zero",
        constant_value: float = 0.0,
        max_k: Optional[int] = None,
        n_random_rankings: int = 100,
        random_state: Optional[int] = 42,
        output_path: Optional[Union[str, Path]] = None,
        plot_title: Optional[str] = None,
        plot_width: int = 1100,
        plot_height: int = 560,
    ) -> Dict[str, Any]:
        """Evaluate SMX faithfulness via progressive top-k zone masking.

        The ranked spectral zones are progressively masked on *X_eval* following
        the selected SMX ranking, and the resulting prediction shift is
        summarized by the area under the masking curve (AUC).

        Parameters
        ----------
        X_eval : pd.DataFrame
            Evaluation spectra to be masked progressively.
        ranking : {'unique', 'summed', 'natural'}, default 'unique'
            Ranking table used to derive the ordered list of spectral zones.
            ``'unique'`` uses the one-zone-per-row ranking in
            :attr:`lrc_unique_`. ``'summed'`` and ``'natural'`` are
            deduplicated internally to one row per zone before masking.
        X_reference : pd.DataFrame, optional
            Reference spectra used to compute replacement values for
            non-zero masking strategies. Defaults to *X_eval*.
        metric : {'auto', 'probability_shift', 'mean_abs_diff', 'decision_function_shift'}, default 'auto'
            Prediction-shift metric to evaluate. ``'auto'`` chooses
            ``'probability_shift'`` when the estimator exposes
            ``predict_proba()``, ``'decision_function_shift'`` when it exposes
            ``decision_function()``, otherwise ``'mean_abs_diff'``.
        masking_strategy : {'zero', 'constant', 'mean', 'median', 'min', 'max'}, default 'zero'
            How masked spectral variables are replaced.
        constant_value : float, default 0.0
            Replacement value used when ``masking_strategy='constant'``.
        max_k : int, optional
            Maximum number of ranked zones to mask. Defaults to all ranked
            zones available in *X_eval*.
        n_random_rankings : int, default 100
            Number of random rankings used to contextualize the observed AUC.
        random_state : int, optional
            Seed controlling the random baseline.
        output_path : str or Path, optional
            If provided, also export a faithfulness plot to this path. The
            extension determines the format (``.html`` or a static image).
        plot_title : str, optional
            Title override used when *output_path* is provided.
        plot_width : int, default 1100
            Plot width in pixels. Used only when *output_path* is provided.
        plot_height : int, default 560
            Plot height in pixels. Used only when *output_path* is provided.

        Returns
        -------
        dict
            Faithfulness summary including ``curve_df``, ``auc``,
            ``auc_normalized``, ``level``, and null-baseline statistics.
        """
        if self.estimator is None:
            raise RuntimeError(
                "SMX requires a fitted estimator to evaluate faithfulness."
            )

        ranking_map = {
            "unique": self.lrc_unique_,
            "summed": self.lrc_,
            "natural": self.lrc_natural_,
        }
        if ranking not in ranking_map:
            raise ValueError("ranking must be 'unique', 'summed', or 'natural'.")

        ranking_df = ranking_map[ranking]
        if ranking_df is None or ranking_df.empty:
            raise RuntimeError(
                f"No ranking data is available for ranking='{ranking}'. Fit SMX before "
                "calling evaluate_faithfulness()."
            )

        result = progressive_masking_faithfulness(
            estimator=self.estimator,
            X_eval=X_eval,
            spectral_cuts=self.spectral_cuts,
            ranking_df=ranking_df,
            X_reference=X_reference,
            metric=metric,
            masking_strategy=masking_strategy,
            constant_value=constant_value,
            max_k=max_k,
            n_random_rankings=n_random_rankings,
            random_state=random_state,
        )
        result["ranking_source"] = ranking
        if output_path is not None:
            from smx.plotting import plot_faithfulness_curve

            plot_faithfulness_curve(
                result,
                output_path=output_path,
                title=plot_title,
                width=plot_width,
                height=plot_height,
            )
            result["plot_path"] = str(output_path)
        self.faithfulness_ = result
        return result

    def evaluate_normalized_faithfulness(
        self,
        X_eval: pd.DataFrame,
        *,
        ranking: Literal["unique", "summed", "natural"] = "unique",
        X_reference: Optional[pd.DataFrame] = None,
        metric: Literal[
            "auto",
            "probability_shift",
            "mean_abs_diff",
            "decision_function_shift",
            "calibration_invariant",
        ] = "auto",
        masking_strategy: Literal["zero", "constant", "mean", "median", "min", "max"] = "zero",
        constant_value: float = 0.0,
        max_k: Optional[int] = None,
        normalization: Literal["auto", "exact", "beam"] = "auto",
        beam_size: int = 5,
        exact_max_zones: int = 9,
        n_random_rankings: int = 100,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Evaluate faithfulness with model/input-specific NAOPC bounds.

        This is an alternative to :meth:`evaluate_faithfulness`. It uses the
        same SMX zone rankings and masking strategies, but computes a
        target-class AOPC per sample and normalizes it by the attainable lower
        and upper bounds for that sample. The legacy faithfulness method is
        not changed by this method.

        ``normalization='exact'`` uses dynamic programming over all masked
        zone subsets. ``'beam'`` uses deterministic upper/lower beam searches;
        ``'auto'`` selects exact normalization when the number of zones is at
        most ``exact_max_zones``.
        """
        if self.estimator is None:
            raise RuntimeError(
                "SMX requires a fitted estimator to evaluate normalized faithfulness."
            )

        ranking_map = {
            "unique": self.lrc_unique_,
            "summed": self.lrc_,
            "natural": self.lrc_natural_,
        }
        if ranking not in ranking_map:
            raise ValueError("ranking must be 'unique', 'summed', or 'natural'.")

        ranking_df = ranking_map[ranking]
        if ranking_df is None or ranking_df.empty:
            raise RuntimeError(
                f"No ranking data is available for ranking='{ranking}'. Fit SMX before "
                "calling evaluate_normalized_faithfulness()."
            )

        result = normalized_progressive_masking_faithfulness(
            estimator=self.estimator,
            X_eval=X_eval,
            spectral_cuts=self.spectral_cuts,
            ranking_df=ranking_df,
            X_reference=X_reference,
            metric=metric,
            masking_strategy=masking_strategy,
            constant_value=constant_value,
            max_k=max_k,
            normalization=normalization,
            beam_size=beam_size,
            exact_max_zones=exact_max_zones,
            n_random_rankings=n_random_rankings,
            random_state=random_state,
        )
        result["ranking_source"] = ranking
        self.normalized_faithfulness_ = result
        return result

    def plot_zone_ranking_over_spectrum(
        self,
        output_path: Optional[Union[str, Path]] = None,
        *,
        ranking: Literal["unique", "natural"] = "unique",
        aggregation: Literal["mean", "median"] = "mean",
        X_natural: Optional[pd.DataFrame] = None,
        y_labels: Optional["pd.Series"] = None,
        **plot_kwargs: Any,
    ):
        """Plot ranked spectral zones over the natural-scale mean spectrum.

        Thin wrapper around :func:`smx.plotting.plot_zone_ranking_over_spectrum`
        using the fitted ranking, spectral cuts and natural-scale zones.

        Parameters
        ----------
        output_path : str or Path, optional
            Export destination; ``.html`` for an interactive figure or
            ``.png`` / ``.svg`` / ``.pdf`` for a static image (requires
            ``kaleido``).  Nothing is written when ``None``.
        ranking : {'unique', 'natural'}, default 'unique'
            Ranking source. ``'unique'`` uses ``lrc_unique_`` (one row per
            zone). ``'natural'`` uses ``lrc_natural_`` and collapses multiple
            predicates per zone to the strongest LRC value.
        aggregation : {'mean', 'median'}, default 'mean'
            Aggregation used to build the reference spectrum from
            ``zones_natural_``.
        X_natural : pd.DataFrame, optional
            Full calibration dataset in natural (unpreprocessed) units.  When
            provided together with *y_labels*, a mean spectrum is drawn for each
            class on top of the overall reference spectrum.
        y_labels : pd.Series, optional
            Class labels aligned with the rows of *X_natural*.
        plot_kwargs
            Forwarded to :func:`smx.plotting.plot_zone_ranking_over_spectrum`
            (e.g. ``title``, ``class_colors``, ``theme``, ``width``,
            ``height``, ``return_df``).

        Returns
        -------
        plotly.graph_objects.Figure or (Figure, pd.DataFrame)
            See :func:`smx.plotting.plot_zone_ranking_over_spectrum`.
        """
        from smx.plotting import plot_zone_ranking_over_spectrum

        if self.lrc_ is None:
            raise RuntimeError(
                "SMX must be fitted before calling plot_zone_ranking_over_spectrum."
            )

        if self.zones_natural_ is None:
            raise RuntimeError(
                "SMX was fitted without X_cal_natural, so no natural-scale reference spectrum is available for plotting. "
                "Re-run SMX.fit(..., X_cal_natural=...) to enable plotting over the natural spectrum."
            )

        if ranking == "unique":
            ranking_df = self.lrc_unique_
        elif ranking == "natural":
            ranking_df = self.lrc_natural_
        else:
            raise ValueError("ranking must be 'unique' or 'natural'.")

        if ranking_df is None or ranking_df.empty:
            raise RuntimeError(
                "No ranking information is available. Fit SMX successfully before plotting."
            )

        if (X_natural is None) != (y_labels is None):
            raise ValueError("X_natural and y_labels must be provided together.")
        class_spectra = None
        if X_natural is not None:
            labels = np.asarray(y_labels)
            class_spectra = {
                str(cls): X_natural[labels == cls]
                for cls in pd.unique(labels)
            }

        return plot_zone_ranking_over_spectrum(
            ranking_df,
            self.spectral_cuts,
            self.zones_natural_,
            output_path=output_path,
            aggregation=aggregation,
            class_spectra=class_spectra,
            **plot_kwargs,
        )

    def plot_faithfulness(
        self,
        output_path: Optional[Union[str, Path]] = None,
        **plot_kwargs: Any,
    ):
        """Plot the progressive masking faithfulness curve saved in ``faithfulness_``.

        Thin wrapper around :func:`smx.plotting.plot_faithfulness_curve`.

        Parameters
        ----------
        output_path : str or Path, optional
            Export destination; ``.html`` for an interactive figure or an
            image extension for static export.  Nothing is written when
            ``None``.
        plot_kwargs
            Forwarded to :func:`smx.plotting.plot_faithfulness_curve`
            (e.g. ``title``, ``theme``, ``width``, ``height``,
            ``show_percentile``, ``return_df``).

        Returns
        -------
        plotly.graph_objects.Figure or (Figure, pd.DataFrame)
            See :func:`smx.plotting.plot_faithfulness_curve`.
        """
        from smx.plotting import plot_faithfulness_curve

        if self.faithfulness_ is None:
            raise RuntimeError(
                "No faithfulness result is available. Call evaluate_faithfulness() "
                "before plot_faithfulness()."
            )

        return plot_faithfulness_curve(
            self.faithfulness_,
            output_path=output_path,
            **plot_kwargs,
        )
