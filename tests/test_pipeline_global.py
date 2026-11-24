"""SMX.fit — global outputs: schema, validation, determinism, warnings."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import pytest

from smx import SMX
from smx.graph import builder as builder_module
from helpers import ConstantProbaStub, LinearSoftmaxStub, fit_smx, make_known_truth, quiet

LRC_COLUMNS = ["Node", "Local_Reaching_Centrality", "Zone", "Threshold", "Operator"]


@pytest.mark.parametrize("classes, labels", [
    (("A", "B"), ("A", "B")),
    (("A", "B", "C"), (0, 1, 2)),          # integer predictions -> 'Class_0', ...
    (("A", "B", "C", "D", "E"), ("A", "B", "C", "D", "E")),
])
def test_global_output_schema_for_k_classes(classes, labels):
    data = make_known_truth(seed=3, classes=classes)
    est = LinearSoftmaxStub(classes=labels, n_ch=data["n_ch"], n_zones=data["n_zones"])
    s = fit_smx(data, est)

    assert s.valid_seeds_ == [0, 1]
    for table in (s.lrc_, s.lrc_unique_):
        assert list(table.columns) == LRC_COLUMNS
        assert table["Local_Reaching_Centrality"].is_monotonic_decreasing
        assert table["Node"].is_unique
    zones = set(s.lrc_["Zone"].dropna())
    assert zones <= set(data["zones"])
    assert set(s.lrc_["Node"]) <= set(s.predicates_df_["rule"])   # no terminal rows (B3)
    assert set(s.lrc_unique_["Zone"]) <= set(data["zones"])
    assert s.lrc_unique_["Zone"].is_unique
    for g in s.graphs_by_seed_.values():
        terminals = {n for n, a in g.nodes(data=True) if a["node_type"] == "terminal"}
        assert terminals == {f"Class_{c}" for c in labels}
    assert list(s.lrc_natural_.columns) == LRC_COLUMNS + [
        "Threshold_Natural", "Reference_Sample_Index", "Approximation_Error", "Node_Natural"]
    pd.testing.assert_frame_equal(s.lrc_natural_[LRC_COLUMNS], s.lrc_)


def test_natural_outputs_are_none_without_x_cal_natural(known_truth):
    s = fit_smx(known_truth, known_truth["est"], natural=False)
    assert s.lrc_natural_ is None and s.zones_natural_ is None and s.pca_info_natural_ is None
    assert s.lrc_ is not None


@pytest.mark.parametrize("case, match", [
    ("one_label_short", "must match"),
    ("single_class", "at least 2 distinct"),
])
def test_fit_validates_class_labels(known_truth, case, match):
    n = len(known_truth["X"])
    y = known_truth["est"].predict(known_truth["X"])[:-1] if case == "one_label_short" else ["A"] * n
    s = SMX(spectral_cuts=known_truth["cuts"], quantiles=[0.5], estimator=known_truth["est"])
    with pytest.raises(ValueError, match=match):
        s.fit(known_truth["X"], y)


def test_fit_is_deterministic_and_idempotent(known_truth, fitted_global):
    twin = fit_smx(known_truth, known_truth["est"])
    again = copy.deepcopy(twin)
    with quiet():
        again.fit(known_truth["X"], known_truth["est"].predict(known_truth["X"]),
                  X_cal_natural=known_truth["X"] * 1.7 + 3.0)
    for other in (twin, again):
        pd.testing.assert_frame_equal(other.lrc_, fitted_global.lrc_, check_exact=True)
        pd.testing.assert_frame_equal(other.lrc_unique_, fitted_global.lrc_unique_, check_exact=True)


def test_fit_leaves_numpy_global_random_state_untouched(known_truth, fitted_global):
    """B6: SMX draws from local generators, never from numpy's global one."""
    np.random.seed(12345)
    before = np.random.get_state()
    s = fit_smx(known_truth, known_truth["est"], per_class="within")
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    # ... and the result does not depend on what the caller did to it.
    pd.testing.assert_frame_equal(s.lrc_, fitted_global.lrc_, check_exact=True)


def test_bags_are_annotated_with_the_given_labels(known_truth, monkeypatch):
    """The pipeline annotates every bag frame with y_class_labels[Sample_Index]."""
    seen = []
    original = builder_module.PredicateGraphBuilder.build

    def spy(self, bags, rankings):
        seen.append(bags)
        return original(self, bags, rankings)

    monkeypatch.setattr(builder_module.PredicateGraphBuilder, "build", spy)
    y = np.array([f"L{i % 3}" for i in range(len(known_truth["X"]))])  # arbitrary labels
    with quiet():
        SMX(spectral_cuts=known_truth["cuts"], quantiles=[0.5], n_repetitions=1, n_bags=2,
            estimator=known_truth["est"]).fit(known_truth["X"], y)
    assert seen
    for bags in seen:
        for preds in bags.values():
            for df in preds.values():
                np.testing.assert_array_equal(df["Class_Predicted"].values,
                                              y[df["Sample_Index"].values])


def test_warns_when_every_global_predicate_has_zero_lrc(known_truth, caplog):
    """N3: an insensitive model gives zero perturbation everywhere."""
    with caplog.at_level(logging.WARNING, logger="smx.pipeline"):
        s = fit_smx(known_truth, ConstantProbaStub())
    msgs = [r.getMessage() for r in caplog.records if "every predicate has LRC 0" in r.getMessage()]
    assert len(msgs) == len(s.valid_seeds_) == 2
    # B1: the seeds stay valid; every evaluated predicate has a row with LRC 0
    # (no graph edge at all), every zone is present and there is no terminal row.
    assert all(g.number_of_edges() == 0 for g in s.graphs_by_seed_.values())
    assert (s.lrc_["Local_Reaching_Centrality"] == 0).all()
    assert set(s.lrc_["Node"]) <= set(s.predicates_df_["rule"])
    assert set(s.lrc_unique_["Zone"]) == set(known_truth["zones"])


def test_no_zero_lrc_warning_for_an_informative_model(known_truth, caplog):
    with caplog.at_level(logging.WARNING, logger="smx.pipeline"):
        fit_smx(known_truth, known_truth["est"])
    assert not [r for r in caplog.records if "every predicate has LRC 0" in r.getMessage()]


@pytest.mark.parametrize("var_exp", [True, False])
def test_zero_effect_zone_gets_lrc_zero_and_does_not_poison_the_others(known_truth, var_exp):
    """B1: with var_exp=False the zero effect of Zn used to be a Python-float
    zero edge, which set the LRC of every predicate upstream of it to 0."""
    s = fit_smx(known_truth, known_truth["est"], var_exp=var_exp)
    zone_lrc = s.lrc_unique_.set_index("Zone")["Local_Reaching_Centrality"]
    assert zone_lrc.loc["Zn"] == 0.0
    assert (zone_lrc.drop("Zn") > 0).all()
    for g in s.graphs_by_seed_.values():
        assert all(d["weight"] > 0 for *_, d in g.edges(data=True))
        assert any(p.startswith("Zn ") for p in g.graph["zero_effect_predicates"])


def test_warns_about_predicates_with_a_non_finite_metric(known_truth, caplog, monkeypatch):
    """Non-finite metric = not evaluated: left out, no row, and a warning."""
    from smx.predicates import metrics as metrics_module

    original = metrics_module.PerturbationMetric.compute

    def with_nan(self, bags):
        out = original(self, bags)
        for name, rk in out.items():
            if not name.startswith("__"):
                rk.loc[rk["Predicate"].str.startswith("Zc "), "Perturbation"] = np.nan
        return out

    monkeypatch.setattr(metrics_module.PerturbationMetric, "compute", with_nan)
    with caplog.at_level(logging.WARNING, logger="smx.pipeline"):
        s = fit_smx(known_truth, known_truth["est"])
    assert "Zc" not in set(s.lrc_["Zone"])
    assert any("non-finite perturbation" in r.getMessage() for r in caplog.records)


def test_normalized_faithfulness_works_after_a_real_fit(known_truth, fitted_global):
    """B4: it used to read lrc_summed_unique_/lrc_summed_, which fit() never sets."""
    s = copy.deepcopy(fitted_global)
    result = s.evaluate_normalized_faithfulness(known_truth["X"], normalization="beam",
                                                n_random_rankings=3)
    assert isinstance(result, dict) and "naopc" in str(sorted(result)).lower()
