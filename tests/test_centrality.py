"""compute_lrc and the cross-seed aggregations."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from smx.graph.centrality import (
    aggregate_lrc_across_seeds,
    aggregate_lrc_across_seeds_per_class,
    compute_lrc,
)
from helpers import oracle_lrc_chain

RULES = ["Za <= 1.0", "Zb > 0.5", "Zc <= -0.2", "Zd > 3.0"]
PREDICATES = pd.DataFrame({
    "rule": RULES,
    "zone": ["Za", "Zb", "Zc", "Zd"],
    "thresholds": [1.0, 0.5, -0.2, 3.0],
    "operator": ["<=", ">", "<=", ">"],
})


def _chain(weights, cast=float):
    g = nx.DiGraph()
    nodes = RULES[: len(weights)] + ["Class_A"]
    for i, w in enumerate(weights):
        g.add_edge(nodes[i], nodes[i + 1], weight=cast(w))
    return g, nodes


def _lrc(g, **kwargs):
    return compute_lrc(g, PREDICATES, verbose=False, **kwargs).set_index("Node")


def test_lrc_of_a_chain_matches_the_definition():
    weights = [0.6, 0.3, 0.2, 0.1]
    g, nodes = _chain(weights)
    got = _lrc(g)["Local_Reaching_Centrality"]
    np.testing.assert_allclose(got.loc[nodes].values, oracle_lrc_chain(weights), rtol=1e-12)


def test_lrc_table_is_sorted_and_carries_the_predicate_metadata():
    g, _ = _chain([0.6, 0.3, 0.2])
    g.add_edge("Zq > 9.0", "Class_A", weight=0.05)  # not in the catalogue
    df = compute_lrc(g, PREDICATES, verbose=False)
    assert df["Local_Reaching_Centrality"].is_monotonic_decreasing
    meta = df.set_index("Node")
    assert meta.loc["Zb > 0.5", ["Zone", "Threshold", "Operator"]].tolist() == ["Zb", 0.5, ">"]
    assert meta.loc["Zq > 9.0", "Zone"] == "Unknown"
    assert meta.loc["Class_A", ["Zone", "Threshold", "Operator"]].isna().all()


def test_lrc_is_invariant_to_a_common_scale_of_the_weights():
    a, _ = _chain([0.6, 0.3, 0.2, 0.1])
    b, _ = _chain([7.3 * w for w in [0.6, 0.3, 0.2, 0.1]])
    la, lb = _lrc(a)["Local_Reaching_Centrality"], _lrc(b)["Local_Reaching_Centrality"]
    np.testing.assert_allclose(lb.loc[la.index].values, la.values, rtol=1e-12)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), np.float64(-1e-12),
                                 0.0, np.float64(0.0)])
def test_invalid_weights_raise_instead_of_zeroing_the_graph(bad):
    """N2 + B1: a weight must be finite and > 0. A negative one makes networkx
    reject the graph; a zero one breaks every upstream LRC (as a Python float,
    all of them became 0). The builder never creates such edges."""
    g, _ = _chain([0.6, 0.3, 0.2])
    g[RULES[1]][RULES[2]]["weight"] = bad
    with pytest.raises(ValueError, match="invalid weight"):
        compute_lrc(g, PREDICATES, verbose=False)


def test_empty_graph_and_graph_without_edges():
    empty = compute_lrc(nx.DiGraph(), PREDICATES, verbose=False)
    assert empty.empty and list(empty.columns) == [
        "Node", "Local_Reaching_Centrality", "Zone", "Threshold", "Operator"]
    g = nx.DiGraph()
    g.add_node("Class_A", node_type="terminal")
    g.add_node("Class_B", node_type="terminal")
    assert (_lrc(g)["Local_Reaching_Centrality"] == 0.0).all()


def test_zero_effect_predicates_are_reported_with_lrc_zero():
    """B1: evaluated predicates left out of the chain come back with LRC 0,
    with the catalogue metadata, after every node of the graph."""
    g, nodes = _chain([0.6, 0.3])
    g.graph["zero_effect_predicates"] = [RULES[3], RULES[2]]
    df = compute_lrc(g, PREDICATES, verbose=False)
    assert list(df["Node"]) == [RULES[0], RULES[1], "Class_A", RULES[3], RULES[2]]
    meta = df.set_index("Node")
    assert meta.loc[RULES[2], ["Local_Reaching_Centrality", "Zone", "Threshold", "Operator"]
                    ].tolist() == [0.0, "Zc", -0.2, "<="]
    # Positive LRC values are the chain's own, untouched by the extra rows.
    np.testing.assert_allclose(meta.loc[nodes, "Local_Reaching_Centrality"].values,
                               oracle_lrc_chain([0.6, 0.3]), rtol=1e-12)


def test_only_zero_effect_predicates_gives_an_all_zero_table():
    """A model insensitive to every zone: terminals only, every predicate LRC 0."""
    g = nx.DiGraph()
    g.add_node("Class_A", node_type="terminal")
    g.graph["zero_effect_predicates"] = RULES[:2]
    df = compute_lrc(g, PREDICATES, verbose=False)
    assert set(df["Node"]) == {"Class_A", *RULES[:2]}
    assert (df["Local_Reaching_Centrality"] == 0.0).all()


def test_verbose_flag(capsys):
    g, _ = _chain([0.6, 0.3])
    compute_lrc(g, PREDICATES, verbose=False)
    assert capsys.readouterr().out == ""
    compute_lrc(g, PREDICATES)
    assert "Processing graph LRC" in capsys.readouterr().out


@pytest.mark.parametrize("cast", [float, np.float64])
def test_zero_weight_edge_raises_whatever_the_number_type(cast):
    """Q6/B1: a zero-weight edge used to zero every upstream LRC (Python float)
    or give infinite, arbitrarily tied path lengths (np.float64)."""
    g, _ = _chain([0.5, 0.0, 0.2, 0.1], cast)
    with pytest.raises(ValueError, match="invalid weight"):
        compute_lrc(g, PREDICATES, verbose=False)


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def _seed_table(rows):
    df = pd.DataFrame(rows, columns=["Node", "Local_Reaching_Centrality"])
    meta = PREDICATES.set_index("rule")
    df["Zone"] = [None if n.startswith("Class_") else meta.loc[n, "zone"] for n in df["Node"]]
    df["Threshold"] = [None if n.startswith("Class_") else meta.loc[n, "thresholds"] for n in df["Node"]]
    df["Operator"] = [None if n.startswith("Class_") else meta.loc[n, "operator"] for n in df["Node"]]
    return df


@pytest.fixture()
def lrc_by_seed():
    return {
        0: _seed_table([("Za <= 1.0", 0.5), ("Zb > 0.5", 0.3), ("Class_A", 0.0)]),
        1: _seed_table([("Zd > 3.0", 0.9), ("Za <= 1.0", 0.7), ("Class_A", 0.0)]),
        2: _seed_table([("Zb > 0.5", 0.1), ("Class_A", 0.0)]),
    }


def test_per_class_aggregation_means_over_present_seeds_and_counts_them(lrc_by_seed):
    lrc, unique = aggregate_lrc_across_seeds_per_class(lrc_by_seed, [0, 1, 2])
    got = lrc.set_index("Node")
    assert list(lrc["Node"]) == ["Zd > 3.0", "Za <= 1.0", "Zb > 0.5"]
    assert got["Local_Reaching_Centrality"].tolist() == pytest.approx([0.9, 0.6, 0.2])
    assert got["N_Seeds"].tolist() == [1, 2, 2]
    assert not lrc["Node"].str.startswith("Class_").any()
    # One row per zone, no terminal row, best predicate of each zone.
    assert list(unique["Zone"]) == ["Zd", "Za", "Zb"] and unique["Zone"].is_unique
    assert list(unique.columns) == ["Node", "Local_Reaching_Centrality", "Zone",
                                    "Threshold", "Operator", "N_Seeds"]


def test_per_class_aggregation_keeps_only_requested_seeds(lrc_by_seed):
    lrc, _ = aggregate_lrc_across_seeds_per_class(lrc_by_seed, [0])
    assert lrc.set_index("Node")["N_Seeds"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="none of the requested seeds"):
        aggregate_lrc_across_seeds_per_class(lrc_by_seed, [7])


def test_global_aggregation_means_over_present_seeds(lrc_by_seed):
    lrc, _ = aggregate_lrc_across_seeds(lrc_by_seed, [0, 1, 2])
    got = lrc.set_index("Node")["Local_Reaching_Centrality"]
    assert got.loc[["Zd > 3.0", "Za <= 1.0", "Zb > 0.5"]].tolist() == pytest.approx([0.9, 0.6, 0.2])
    assert "N_Seeds" not in lrc.columns


def test_global_tables_have_no_terminal_and_one_row_per_zone(lrc_by_seed):
    """B3: terminals are dropped after aggregation; other rows keep their order."""
    lrc, unique = aggregate_lrc_across_seeds(lrc_by_seed, [0, 1, 2])
    assert not lrc["Node"].str.startswith("Class_").any()
    assert not unique["Node"].str.startswith("Class_").any()
