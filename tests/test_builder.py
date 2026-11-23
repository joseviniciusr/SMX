"""PredicateGraphBuilder: chain, exact weights, var_exp, bidirectional edges, terminals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smx.graph.builder import PredicateGraphBuilder

P1, P2, P3 = "Za <= 1.00", "Zb > 0.50", "Zc <= -0.20"


def _bag(rules, predicted="A"):
    return {r: pd.DataFrame({"Zone_Sum": [0.0, 0.0, 0.0], "Sample_Index": [0, 1, 2],
                             "Class_Predicted": list(predicted) * (3 // len(predicted))})
            for r in rules}


def _ranking(pairs):
    return pd.DataFrame(pairs, columns=["Predicate", "Perturbation"])


def _build(bags, rankings, **kwargs):
    params = dict(class_labels=["A", "B"], random_state=0, show_details=False, verbose=False)
    params.update(kwargs)
    return PredicateGraphBuilder(**params).build(bags, rankings)


def _edges(g):
    return {(u, v): d["weight"] for u, v, d in g.edges(data=True)}


def test_chain_follows_the_ranking_and_weights_are_the_source_metric():
    bags = {"Bag_1": _bag([P1, P2, P3])}
    rankings = {"Bag_1": _ranking([(P2, 0.9), (P1, 0.4), (P3, 0.1)])}
    assert _edges(_build(bags, rankings)) == {(P2, P1): 0.9, (P1, P3): 0.4, (P3, "Class_A"): 0.1}


def test_predicates_absent_from_the_bag_are_skipped_in_the_chain():
    bags = {"Bag_1": _bag([P1, P3])}
    rankings = {"Bag_1": _ranking([(P2, 0.9), (P1, 0.4), (P3, 0.1)])}
    assert _edges(_build(bags, rankings)) == {(P1, P3): 0.4, (P3, "Class_A"): 0.1}


def test_weights_accumulate_exactly_across_bags():
    bags = {"Bag_1": _bag([P1, P2]), "Bag_2": _bag([P1, P2]), "Bag_3": _bag([P1, P2])}
    rankings = {b: _ranking([(P1, w), (P2, 0.05)]) for b, w in
                zip(bags, [0.30, 0.25, 0.125])}
    edges = _edges(_build(bags, rankings))
    assert edges[(P1, P2)] == 0.30 + 0.25 + 0.125
    assert edges[(P2, "Class_A")] == 0.05 * 3


def test_var_exp_multiplies_by_the_source_zone_pc1_variance():
    pca_info = {"Za": {"variance_explained": np.float64(0.5)},
                "Zb": {"variance_explained": np.float64(0.25)}}
    bags = {"Bag_1": _bag([P1, P2])}
    rankings = {"Bag_1": _ranking([(P1, 0.8), (P2, 0.4)])}
    plain = _edges(_build(bags, rankings))
    weighted = _edges(_build(bags, rankings, var_exp=True, pca_info_dict=pca_info))
    assert weighted[(P1, P2)] == plain[(P1, P2)] * 0.5
    assert weighted[(P2, "Class_A")] == plain[(P2, "Class_A")] * 0.25


def test_var_exp_requires_pca_info():
    with pytest.raises(ValueError, match="pca_info_dict is required"):
        PredicateGraphBuilder(var_exp=True)


def test_bidirectional_pair_keeps_the_heavier_direction():
    bags = {"Bag_1": _bag([P1, P2]), "Bag_2": _bag([P1, P2]), "Bag_3": _bag([P1, P2])}
    rankings = {"Bag_1": _ranking([(P1, 0.3), (P2, 0.1)]),
                "Bag_2": _ranking([(P2, 0.2), (P1, 0.1)]),
                "Bag_3": _ranking([(P2, 0.2), (P1, 0.1)])}
    edges = _edges(_build(bags, rankings))
    assert (P2, P1) in edges and (P1, P2) not in edges
    assert edges[(P2, P1)] == pytest.approx(0.4)


def test_tied_bidirectional_pair_is_resolved_deterministically_per_seed():
    bags = {"Bag_1": _bag([P1, P2]), "Bag_2": _bag([P1, P2])}
    rankings = {"Bag_1": _ranking([(P1, 0.2), (P2, 0.1)]),
                "Bag_2": _ranking([(P2, 0.2), (P1, 0.1)])}
    outcomes = set()
    for seed in range(12):
        a, b = _edges(_build(bags, rankings, random_state=seed)), _edges(_build(bags, rankings, random_state=seed))
        assert a == b
        assert ((P1, P2) in a) != ((P2, P1) in a)  # exactly one direction survives
        outcomes.add((P1, P2) in a)
    assert outcomes == {True, False}  # the tie-break really depends on the seed


def test_last_predicate_points_to_the_majority_predicted_class():
    bags = {"Bag_1": {P1: pd.DataFrame({"Zone_Sum": 0.0, "Sample_Index": range(5),
                                         "Class_Predicted": ["A", "B", "B", "B", "A"]})}}
    edges = _edges(_build(bags, {"Bag_1": _ranking([(P1, 0.5)])}))
    assert edges == {(P1, "Class_B"): 0.5}


@pytest.mark.parametrize("labels, expected", [
    (["A", "B", "C"], {"Class_A", "Class_B", "Class_C"}),
    ([0, 1, 2], {"Class_0", "Class_1", "Class_2"}),
    (None, {"Class_A", "Class_B"}),  # legacy binary fallback
])
def test_one_terminal_per_class_label(labels, expected):
    g = _build({}, {}, class_labels=labels)
    assert {n for n, a in g.nodes(data=True) if a["node_type"] == "terminal"} == expected


def test_terminal_class_gives_a_single_terminal_reached_by_every_bag():
    bags = {"Bag_1": _bag([P1, P2], "B"), "Bag_2": _bag([P2, P3], "C")}
    rankings = {"Bag_1": _ranking([(P1, 0.3), (P2, 0.2)]),
                "Bag_2": _ranking([(P2, 0.6), (P3, 0.1)])}
    g = _build(bags, rankings, class_labels=["A", "B", "C"], terminal_class="A")
    terminals = [n for n, a in g.nodes(data=True) if a["node_type"] == "terminal"]
    assert terminals == ["Class_A"]
    assert _edges(g) == {(P1, P2): 0.3, (P2, "Class_A"): 0.2, (P2, P3): 0.6, (P3, "Class_A"): 0.1}


def test_terminal_class_is_cast_to_string():
    g = _build({}, {}, terminal_class=2)
    assert list(g.nodes) == ["Class_2"]


def test_verbose_false_prints_nothing_and_true_prints_the_summary(capsys):
    bags = {"Bag_1": _bag([P1, P2]), "Bag_2": _bag([P1, P2])}
    rankings = {"Bag_1": _ranking([(P1, 0.3), (P2, 0.1)]), "Bag_2": _ranking([(P2, 0.4), (P1, 0.1)])}
    _build(bags, rankings, verbose=False, show_details=True)
    assert capsys.readouterr().out == ""
    _build(bags, rankings, verbose=True, show_details=True)
    out = capsys.readouterr().out
    assert "CONSTRUCTED GRAPH SUMMARY" in out and "Removed:" in out


def test_metric_column_keyword_is_rejected():
    with pytest.raises(TypeError, match="metric_column"):
        PredicateGraphBuilder(class_labels=["A"]).build({}, {}, metric_column="X")
