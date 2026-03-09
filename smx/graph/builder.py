"""
PredicateGraphBuilder: construct a directed predicate graph.

Edges are added between consecutive predicates ordered by a ranking metric
inside each bag, and weights are accumulated across bags.  Bidirectional
edges are resolved by keeping only the higher-weight direction.
"""

from typing import Dict, List, Optional

import numpy as np
import networkx as nx
import pandas as pd

from smx.graph.interpretation import _extract_zone_from_predicate


class PredicateGraphBuilder:
    """Build a directed predicate graph from bags and metric rankings.

    Edge weights derive from the ranking metric of the *source* predicate in
    each bag.  When the same directed edge appears in multiple bags, the
    weights are accumulated (summed).  Bidirectional edges (A→B and B→A)
    are resolved by:

    * keeping the edge with the higher accumulated weight;
    * breaking ties randomly.

    Parameters
    ----------
    random_state : int, default 42
        Seed for random tie-breaking of bidirectional edges.
    show_details : bool, default True
        Print details about identified and removed bidirectional edges.
    var_exp : bool, default False
        When ``True``, multiply edge weights by the PC1 explained variance of
        the source predicate's spectral zone.  Requires *pca_info_dict*.
    pca_info_dict : dict, optional
        ``{zone_name: {'variance_explained': float, ...}}`` as returned by
        :class:`smx.zones.aggregation.ZoneAggregator` (``pca_info_`` attribute).
        Required when ``var_exp=True``.
    """

    def __init__(
        self,
        random_state: int = 42,
        show_details: bool = True,
        var_exp: bool = False,
        pca_info_dict: Optional[Dict] = None,
    ) -> None:
        if var_exp and pca_info_dict is None:
            raise ValueError("pca_info_dict is required when var_exp=True.")
        self.random_state = random_state
        self.show_details = show_details
        self.var_exp = var_exp
        self.pca_info_dict = pca_info_dict

    def build(
        self,
        bags_result: Dict[str, Dict[str, pd.DataFrame]],
        predicate_ranking_dict: Dict[str, pd.DataFrame],
        metric_column: str = "Covariance",
    ) -> nx.DiGraph:
        """Build and return the directed predicate graph.

        Parameters
        ----------
        bags_result : dict
            Bags as returned by :class:`smx.predicates.bagging.PredicateBagger`.
        predicate_ranking_dict : dict
            ``{bag_name: DataFrame(['Predicate', metric_column])}``
            as returned by a :class:`smx.predicates.metrics.BasePredicateMetric`
            subclass.
        metric_column : str, default ``'Covariance'``
            Name of the metric column in *predicate_ranking_dict*.

        Returns
        -------
        nx.DiGraph
            Directed graph with ``'weight'`` edge attributes.
        """
        np.random.seed(self.random_state)

        DG: nx.DiGraph = nx.DiGraph()
        DG.add_node("Class_A", node_type="terminal", class_label="A")
        DG.add_node("Class_B", node_type="terminal", class_label="B")

        # ── Phase 1: accumulate edge weights ─────────────────────────────
        for bag_name, bag_predicates_dict in bags_result.items():
            if bag_name not in predicate_ranking_dict:
                continue
            ranking_df: pd.DataFrame = predicate_ranking_dict[bag_name]
            if ranking_df.empty:
                continue

            ordered = [
                p for p in ranking_df["Predicate"].tolist()
                if p in bag_predicates_dict
            ]
            if not ordered:
                continue

            lookup: Dict[str, float] = dict(
                zip(ranking_df["Predicate"], ranking_df[metric_column])
            )

            for i in range(len(ordered) - 1):
                src = ordered[i]
                dst = ordered[i + 1]
                DG.add_node(src, node_type="predicate")
                DG.add_node(dst, node_type="predicate")
                w = self._edge_weight(src, lookup)
                self._accumulate(DG, src, dst, w, bag_name)

            # Last predicate → terminal
            last = ordered[-1]
            DG.add_node(last, node_type="predicate")
            df_last = bag_predicates_dict[last]
            if "Class_Predicted" in df_last.columns:
                majority = df_last["Class_Predicted"].value_counts().idxmax()
                terminal = f"Class_{majority}"
            else:
                terminal = "Class_A"
            w = self._edge_weight(last, lookup)
            self._accumulate(DG, last, terminal, w, bag_name)

        # ── Phase 2: resolve bidirectional edges ─────────────────────────
        n_removed = self._resolve_bidirectional(DG)
        print(
            f"\n{'='*70}\n"
            f"CONSTRUCTED GRAPH SUMMARY\n"
            f"{'='*70}\n"
            f"Edges (after removing {n_removed} bidirectional): {DG.number_of_edges()}\n"
            f"Predicate nodes: "
            f"{sum(1 for _, a in DG.nodes(data=True) if a.get('node_type') == 'predicate')}\n"
            f"Metric: {metric_column}\n"
            f"Variance-exp weighting: {'ENABLED' if self.var_exp else 'DISABLED'}"
        )
        return DG

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _edge_weight(self, predicate: str, lookup: Dict[str, float]) -> float:
        w = float(lookup.get(predicate, 0.0))
        if self.var_exp and self.pca_info_dict is not None:
            try:
                zone = _extract_zone_from_predicate(predicate)
                if zone in self.pca_info_dict:
                    w *= self.pca_info_dict[zone]["variance_explained"]
            except ValueError:
                pass
        return w

    @staticmethod
    def _accumulate(DG: nx.DiGraph, src: str, dst: str, w: float, bag: str) -> None:
        if DG.has_edge(src, dst):
            DG[src][dst]["weight"] += w
        else:
            DG.add_edge(src, dst, weight=w, bag=bag)

    def _resolve_bidirectional(self, DG: nx.DiGraph) -> int:
        """Remove the weaker direction of every conflicting edge pair."""
        pairs = []
        processed = set()
        for u, v in list(DG.edges()):
            if DG.has_edge(v, u) and (v, u) not in processed:
                pairs.append((u, v, float(DG[u][v]["weight"]), float(DG[v][u]["weight"])))
                processed.add((u, v))
                processed.add((v, u))

        print(f"\nTotal bidirectional pairs found: {len(pairs)}")
        n_removed = 0
        for u, v, w_fwd, w_rev in pairs:
            if not (DG.has_edge(u, v) and DG.has_edge(v, u)):
                continue  # already resolved
            if w_fwd > w_rev:
                DG.remove_edge(v, u)
                if self.show_details:
                    print(f"Removed: {v} → {u} ({w_rev:.4f}) | Kept: {u} → {v} ({w_fwd:.4f})")
            elif w_rev > w_fwd:
                DG.remove_edge(u, v)
                if self.show_details:
                    print(f"Removed: {u} → {v} ({w_fwd:.4f}) | Kept: {v} → {u} ({w_rev:.4f})")
            else:
                if np.random.rand() > 0.5:
                    DG.remove_edge(v, u)
                    if self.show_details:
                        print(f"Tie! Removed: {v} → {u}")
                else:
                    DG.remove_edge(u, v)
                    if self.show_details:
                        print(f"Tie! Removed: {u} → {v}")
            n_removed += 1
        return n_removed
