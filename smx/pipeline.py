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
from smx.evaluation.faithfulness import progressive_masking_faithfulness

logger = logging.getLogger(__name__)

SpectralCuts = List[tuple]   # list of (name, start, end) or (name, start, end, group)


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
    var_exp : bool, default True
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
        Seeds that produced a non-empty graph (subset of ``seeds``).
    faithfulness_ : dict
        Progressive top-k masking evaluation summary produced by
        :meth:`evaluate_faithfulness`.
    """

    def __init__(
        self,
        spectral_cuts: SpectralCuts,
        quantiles: List[float],
        n_repetitions: int = 4,
        n_bags: int = 10,
        n_samples_fraction: float = 0.8,
        replace: bool = False,
        metric: Literal["covariance", "perturbation"] = "perturbation",
        estimator: Optional[Any] = None,
        perturbation_mode: str = "median",
        perturbation_metric: str = "probability_shift",
        perturbation_stats_source: str = "full",
        normalize_by_zone_size: bool = True,
        zone_size_exponent: float = 1.0,
        covariance_threshold: float = 0.01,
        var_exp: bool = True,
        show_graph_details: bool = False,
        class_threshold: float = 0.5,
    ) -> None:
        if metric not in ("covariance", "perturbation"):
            raise ValueError(f"metric must be 'covariance' or 'perturbation', got '{metric}'.")
        if metric == "perturbation" and estimator is None:
            raise ValueError("estimator is required when metric='perturbation'.")

        self.spectral_cuts = spectral_cuts
        self.quantiles = quantiles
        self.n_repetitions = n_repetitions
        self.seeds = list(range(n_repetitions))
        self.n_bags = n_bags
        self.n_samples_fraction = n_samples_fraction
        self.replace = replace
        self.metric = metric
        self.estimator = estimator
        self.perturbation_mode = perturbation_mode
        self.perturbation_metric = perturbation_metric
        self.perturbation_stats_source = perturbation_stats_source
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
        self.faithfulness_: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_cal_prep: pd.DataFrame,
        y_pred_cal: Union[pd.Series, np.ndarray],
        X_cal_natural: Optional[pd.DataFrame] = None,
    ) -> "SMX":
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

        metric_column = "Covariance" if self.metric == "covariance" else "Perturbation"

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
                    stats_source=self.perturbation_stats_source,
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
            if len(predicate_nodes) < 1 or graph.number_of_nodes() < 2:
                logger.warning(
                    "Seed %d produced an undersized graph (%s, nodes=%d, predicate_nodes=%d) — skipping.",
                    seed, self.metric, graph.number_of_nodes(), len(predicate_nodes)
                )
                continue

            lrc_df_seed = compute_lrc(graph, predicates_df)
            if lrc_df_seed.empty:
                logger.warning(
                    "Seed %d produced an empty LRC table after graph processing (%s) — skipping.",
                    seed, self.metric,
                )
                continue
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

    def evaluate_faithfulness(
        self,
        X_eval: pd.DataFrame,
        *,
        ranking: Literal["unique", "summed", "natural"] = "unique",
        X_reference: Optional[pd.DataFrame] = None,
        metric: Literal["auto", "probability_shift", "mean_abs_diff"] = "auto",
        masking_strategy: Literal["zero", "constant", "mean", "median", "min", "max"] = "zero",
        constant_value: float = 0.0,
        max_k: Optional[int] = None,
        n_random_rankings: int = 100,
        random_state: Optional[int] = 42,
        output_path: Optional[Union[str, "Path"]] = None,
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
            :attr:`lrc_summed_unique_`. ``'summed'`` and ``'natural'`` are
            deduplicated internally to one row per zone before masking.
        X_reference : pd.DataFrame, optional
            Reference spectra used to compute replacement values for
            non-zero masking strategies. Defaults to *X_eval*.
        metric : {'auto', 'probability_shift', 'mean_abs_diff'}, default 'auto'
            Prediction-shift metric to evaluate. ``'auto'`` chooses
            ``'probability_shift'`` when the estimator exposes
            ``predict_proba()``, otherwise ``'mean_abs_diff'``.
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
            "unique": self.lrc_summed_unique_,
            "summed": self.lrc_summed_,
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
                faithfulness_result=result,
                output_path=output_path,
                title=plot_title,
                width=plot_width,
                height=plot_height,
            )
            result["plot_path"] = str(output_path)
        self.faithfulness_ = result
        return result

    def plot_zone_ranking_over_spectrum(
        self,
        output_path: Union[str, "Path"],
        *,
        ranking: Literal["unique", "natural"] = "unique",
        aggregation: Literal["mean", "median"] = "mean",
        title: Optional[str] = None,
        X_natural: Optional[pd.DataFrame] = None,
        y_labels: Optional["pd.Series"] = None,
        class_colors: Optional[dict] = None,
        width: Optional[int] = 1200,
        height: Optional[int] = 500,
    ) -> pd.DataFrame:
        """Plot ranked spectral zones over a reference spectrum and save to file.

        The output format is inferred from *output_path* — ``.html`` for an
        interactive figure, or ``.png`` / ``.svg`` / ``.pdf`` for a static image
        (requires ``kaleido``).

        Parameters
        ----------
        output_path : str or Path
            Destination ``.html`` file.
        ranking : {'unique', 'natural'}, default 'unique'
            Ranking source. ``'unique'`` uses ``lrc_summed_unique_`` (one row per
            zone). ``'natural'`` uses ``lrc_natural_`` and collapses multiple
            predicates per zone to the strongest LRC value.
        aggregation : {'mean', 'median'}, default 'mean'
            Aggregation used to build the reference spectrum from
            ``zones_natural_``.
        title : str, optional
            Figure title override.
        X_natural : pd.DataFrame, optional
            Full calibration dataset in natural (unpreprocessed) units.  When
            provided together with *y_labels*, a mean spectrum is drawn for each
            class on top of the overall reference spectrum.
        y_labels : pd.Series, optional
            Class labels aligned with the rows of *X_natural*.  Required when
            *X_natural* is given.
        class_colors : dict[str, str], optional
            Mapping from class label to hex/CSS color string.  Missing labels
            fall back to a built-in palette.
        width : int, default 1200
            Figure width in pixels. Used only for static image exports.
        height : int, default 500
            Figure height in pixels. Used only for static image exports.

        Returns
        -------
        pd.DataFrame
            Normalized ranking table used in the figure.
        """
        from smx.plotting import plot_zone_ranking_over_spectrum

        if self.zones_natural_ is None:
            raise RuntimeError(
                "SMX must be fitted before calling plot_zone_ranking_over_spectrum."
            )

        if ranking == "unique":
            ranking_df = self.lrc_summed_unique_
        elif ranking == "natural":
            ranking_df = self.lrc_natural_
        else:
            raise ValueError("ranking must be 'unique' or 'natural'.")

        if ranking_df is None or ranking_df.empty:
            raise RuntimeError(
                "No ranking information is available. Fit SMX successfully before plotting."
            )

        class_spectra = None
        if X_natural is not None and y_labels is not None:
            class_spectra = {
                str(cls): X_natural[y_labels.values == cls]
                for cls in y_labels.unique()
            }

        return plot_zone_ranking_over_spectrum(
            zone_ranking_df=ranking_df,
            spectral_cuts=self.spectral_cuts,
            reference_spectrum=self.zones_natural_,
            output_path=output_path,
            aggregation=aggregation,
            title=title or "SMX zone ranking over spectrum",
            class_spectra=class_spectra,
            class_colors=class_colors,
            width=width,
            height=height,
        )

    def plot_faithfulness(
        self,
        output_path: Union[str, "Path"],
        *,
        title: Optional[str] = None,
        width: int = 1100,
        height: int = 560,
    ) -> pd.DataFrame:
        """Plot the progressive masking faithfulness curve saved in ``faithfulness_``.

        Parameters
        ----------
        output_path : str or Path
            Destination file. Use ``.html`` for an interactive figure or an
            image extension for static export.
        title : str, optional
            Figure title override.
        width : int, default 1100
            Figure width in pixels. Used only for static image exports.
        height : int, default 560
            Figure height in pixels. Used only for static image exports.

        Returns
        -------
        pd.DataFrame
            Faithfulness masking curve used in the figure.
        """
        from smx.plotting import plot_faithfulness_curve

        if self.faithfulness_ is None:
            raise RuntimeError(
                "No faithfulness result is available. Call evaluate_faithfulness() "
                "before plot_faithfulness()."
            )

        return plot_faithfulness_curve(
            faithfulness_result=self.faithfulness_,
            output_path=output_path,
            title=title,
            width=width,
            height=height,
        )
