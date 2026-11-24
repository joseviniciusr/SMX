"""compute_lrc and the cross-seed aggregations.

Known, not-yet-fixed defects are pinned with ``xfail(strict=True)``: the day
one of them is fixed, the test "unexpectedly passes", the strict flag turns
that into a failure, and the marker must be removed consciously.
"""

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


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), np.float64(-1e-12)])
def test_invalid_weights_raise_instead_of_zeroing_the_graph(bad):
    """N2: networkx rejects negative weights for the whole graph (every LRC -> 0)."""
    g, _ = _chain([0.6, 0.3, 0.2])
    g[RULES[1]][RULES[2]]["weight"] = bad
    with pytest.raises(ValueError, match="invalid weight"):
        compute_lrc(g, PREDICATES, verbose=False)


def test_empty_graph_and_all_zero_graph():
    empty = compute_lrc(nx.DiGraph(), PREDICATES, verbose=False)
    assert empty.empty and list(empty.columns) == [
        "Node", "Local_Reaching_Centrality", "Zone", "Threshold", "Operator"]
    g, _ = _chain([0.0, 0.0])
    assert (_lrc(g)["Local_Reaching_Centrality"] == 0.0).all()


def test_verbose_flag(capsys):
    g, _ = _chain([0.6, 0.3])
    compute_lrc(g, PREDICATES, verbose=False)
    assert capsys.readouterr().out == ""
    compute_lrc(g, PREDICATES)
    assert "Processing graph LRC" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason="Q6 (deferred): a zero-weight Python-float edge "
                   "zeroes the LRC of every upstream node; np.float64 does not.")
def test_zero_weight_edge_does_not_depend_on_the_number_type():
    weights = [0.5, 0.0, 0.2, 0.1]
    as_float = _lrc(_chain(weights, float)[0])["Local_Reaching_Centrality"]
    as_f64 = _lrc(_chain(weights, np.float64)[0])["Local_Reaching_Centrality"]
    assert as_f64.loc[RULES[0]] > 0            # holds today
    assert as_float.loc[RULES[0]] > 0          # fails today: silently 0.0
    np.testing.assert_allclose(as_float.loc[as_f64.index].values, as_f64.values, rtol=1e-12)


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
