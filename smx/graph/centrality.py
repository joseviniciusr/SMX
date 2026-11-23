"""
LRC (Local Reaching Centrality) computation and cross-seed aggregation.
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def compute_lrc(
    graph: nx.DiGraph,
    predicates_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute Local Reaching Centrality (LRC) for every node of *graph*.

    LRC measures how well a node can reach other nodes in the graph,
    weighted by edge weights.  Higher LRC → more central / important.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed predicate graph (e.g., from
        :class:`smx.graph.builder.PredicateGraphBuilder`).
    predicates_df : pd.DataFrame
        Predicate catalogue with columns ``'rule'``, ``'zone'``,
        ``'thresholds'``, ``'operator'``.

    verbose : bool, default True
        Print a progress line.

    Returns
    -------
    pd.DataFrame
        Columns: ``Node``, ``Local_Reaching_Centrality``, ``Zone``,
        ``Threshold``, ``Operator``.  Sorted descending by LRC.

    Raises
    ------
    ValueError
        If any edge weight is negative or not finite. networkx would otherwise
        reject the whole graph and every node would silently get LRC 0.

    Notes
    -----
    Known limitation (not yet fixed): an edge of weight exactly 0 stored as a
    Python ``float`` makes networkx raise ``ZeroDivisionError`` for every node
    whose shortest-path search crosses it, and those nodes get LRC 0. The same
    edge stored as ``np.float64`` (the ``var_exp=True`` path) yields an
    infinite distance instead, and the LRC is computed. This only happens with
    metrics that can be exactly 0 (label-based metrics) and ``var_exp=False``.
    """
    if verbose:
        print("\nProcessing graph LRC…")

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["Node", "Local_Reaching_Centrality", "Zone", "Threshold", "Operator"]
        )

    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        try:
            w_float = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"Edge {u!r} -> {v!r} has a non-numeric weight {w!r}.") from None
        if not np.isfinite(w_float) or w_float < 0:
            raise ValueError(
                f"Edge {u!r} -> {v!r} has an invalid weight {w!r}: LRC needs finite, "
                f"non-negative weights (a negative weight would make every node's LRC 0)."
            )

    lrc_values: Dict[str, float] = {}
    for node in graph.nodes():
        try:
            lrc_values[node] = nx.local_reaching_centrality(graph, node, weight="weight")
        except (ZeroDivisionError, nx.NetworkXError):
            lrc_values[node] = 0.0

    sorted_lrc = sorted(lrc_values.items(), key=lambda x: x[1], reverse=True)
    lrc_df = pd.DataFrame(sorted_lrc, columns=["Node", "Local_Reaching_Centrality"])

    zones, thresholds, operators = [], [], []
    for node in lrc_df["Node"]:
        if node.startswith("Class_"):
            zones.append(None)
            thresholds.append(None)
            operators.append(None)
        else:
            row = predicates_df[predicates_df["rule"] == node]
            if row.empty:
                zones.append("Unknown")
                thresholds.append(None)
                operators.append(None)
            else:
                zones.append(row.iloc[0]["zone"])
                thresholds.append(row.iloc[0]["thresholds"])
                operators.append(row.iloc[0]["operator"])

    lrc_df["Zone"] = zones
    lrc_df["Threshold"] = thresholds
    lrc_df["Operator"] = operators
    return lrc_df


def aggregate_lrc_across_seeds(
    lrc_by_seed: Dict[int, pd.DataFrame],
    random_seeds: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-seed LRC DataFrames into a mean-aggregated ranking.

    Parameters
    ----------
    lrc_by_seed : dict
        ``{seed: lrc_df}`` where each *lrc_df* is returned by
        :func:`compute_lrc` (must have column ``'Node'`` plus
        ``'Local_Reaching_Centrality'``, ``'Zone'``, ``'Threshold'``,
        ``'Operator'``).
    random_seeds : list of int
        Seeds to include in the aggregation (keys of *lrc_by_seed*).

    Returns
    -------
    lrc_df : pd.DataFrame
        Mean-aggregated LRC for all predicates, sorted descending.
    lrc_unique_df : pd.DataFrame
        Zone-deduplicated version of *lrc_df* (one row per zone),
        keeping the highest-ranked predicate per zone.
    """
    frames = [lrc_by_seed[seed].copy() for seed in random_seeds if seed in lrc_by_seed]
    if not frames:
        raise ValueError("lrc_by_seed contains none of the requested seeds.")

    lrc_all = pd.concat(frames, ignore_index=True)

    lrc_df = (
        lrc_all.groupby("Node")
        .agg(
            Local_Reaching_Centrality=("Local_Reaching_Centrality", "mean"),
            Zone=("Zone", "first"),
            Threshold=("Threshold", "first"),
            Operator=("Operator", "first"),
        )
        .reset_index()
        .sort_values("Local_Reaching_Centrality", ascending=False)
        .reset_index(drop=True)
    )

    lrc_unique_df = (
        lrc_df.drop_duplicates(subset=["Zone"], keep="first")
        .reset_index(drop=True)
        .sort_values("Local_Reaching_Centrality", ascending=False)
        .reset_index(drop=True)
    )

    return lrc_df, lrc_unique_df
