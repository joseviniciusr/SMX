"""SMX.fit with per_class: attributes, guard, tables, graphs, coverage, labels."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import pytest

from helpers import LinearRegressorStub, LinearSoftmaxStub, fit_smx, make_known_truth, quiet

PER_CLASS_ATTRS = ("per_class_labels_", "per_class_min_samples_", "lrc_per_class_",
                   "lrc_unique_per_class_", "lrc_natural_per_class_", "graphs_by_class_by_seed_",
                   "valid_seeds_per_class_", "per_class_coverage_")
TABLE_COLUMNS = ["Node", "Local_Reaching_Centrality", "Zone", "Threshold", "Operator", "N_Seeds"]


def test_per_class_attributes_stay_none_when_disabled(fitted_global):
    assert fitted_global.per_class_mode_ is None
    for attr in PER_CLASS_ATTRS:
        assert getattr(fitted_global, attr) is None, attr


def test_within_guard_is_20_percent_of_each_predicted_class(known_truth, fitted_within):
    counts = pd.Series(known_truth["est"].predict(known_truth["X"])).value_counts()
    assert fitted_within.per_class_labels_ == ["A", "B", "C"]
    assert fitted_within.per_class_min_samples_ == {
        k: max(1, int(0.2 * counts[k])) for k in ["A", "B", "C"]}


def test_within_guard_accepts_an_absolute_count(known_truth):
    s = fit_smx(known_truth, known_truth["est"], per_class="within", per_class_min_samples=7)
    assert s.per_class_min_samples_ == {"A": 7, "B": 7, "C": 7}


def test_all_mode_has_no_own_guard(fitted_all):
    assert fitted_all.per_class_mode_ == "all"
    assert fitted_all.per_class_min_samples_ is None


@pytest.mark.parametrize("fixture", ["fitted_within", "fitted_all"])
def test_per_class_tables(request, fixture):
    s = request.getfixturevalue(fixture)
    rules = set(s.predicates_df_["rule"])
    assert set(s.lrc_per_class_) == set(s.lrc_unique_per_class_) == {"A", "B", "C"}
    for k in ["A", "B", "C"]:
        lrc, unique = s.lrc_per_class_[k], s.lrc_unique_per_class_[k]
        for table in (lrc, unique):
            assert list(table.columns) == TABLE_COLUMNS
            assert table["Local_Reaching_Centrality"].is_monotonic_decreasing
            assert (table["Local_Reaching_Centrality"] >= 0).all()
            assert (table["Local_Reaching_Centrality"] > 0).any()
            assert table["N_Seeds"].between(1, len(s.valid_seeds_per_class_[k])).all()
        assert set(lrc["Node"]) <= rules                 # no terminal rows
        assert unique["Zone"].is_unique and set(unique["Zone"]) <= set(s.lrc_["Zone"].dropna())
        # unique = best predicate of each zone
        best = lrc.groupby("Zone")["Local_Reaching_Centrality"].max()
        assert unique.set_index("Zone")["Local_Reaching_Centrality"].to_dict() == best.to_dict()
        natural = s.lrc_natural_per_class_[k]
        assert list(natural["Node"]) == list(lrc["Node"])
        assert natural["Threshold_Natural"].notna().all()


@pytest.mark.parametrize("fixture", ["fitted_within", "fitted_all"])
def test_zero_effect_zone_has_lrc_zero_per_class_and_globally(request, fixture):
    """B1: the noise zone Zn has exactly zero effect in the stub. It is reported
    with LRC 0 (evaluated, no effect), per class exactly as globally."""
    s = request.getfixturevalue(fixture)
    tables = [s.lrc_unique_] + [s.lrc_unique_per_class_[k] for k in ["A", "B", "C"]]
    for t in tables:
        zone_lrc = t.set_index("Zone")["Local_Reaching_Centrality"]
        assert zone_lrc.loc["Zn"] == 0.0
        assert (zone_lrc.drop("Zn") > 0).all()
        assert t.iloc[-1]["Zone"] == "Zn"


@pytest.mark.parametrize("fixture", ["fitted_within", "fitted_all"])
def test_class_graphs_have_a_single_terminal(request, fixture):
    s = request.getfixturevalue(fixture)
    assert set(s.graphs_by_class_by_seed_) == {"A", "B", "C"}
    for k, by_seed in s.graphs_by_class_by_seed_.items():
        assert set(by_seed) == set(s.seeds)
        for g in by_seed.values():
            terminals = [n for n, a in g.nodes(data=True) if a["node_type"] == "terminal"]
            assert terminals == [f"Class_{k}"]
            assert all(d["weight"] > 0 for *_, d in g.edges(data=True))
            assert g.in_degree(f"Class_{k}") >= 1 and g.out_degree(f"Class_{k}") == 0


@pytest.mark.parametrize("fixture", ["fitted_within", "fitted_all"])
def test_coverage_is_consistent(request, fixture):
    s = request.getfixturevalue(fixture)
    cov = s.per_class_coverage_
    assert len(cov) == len(s.seeds) * 3
    assert list(cov.columns) == ["Seed", "Class", "N_Bags", "N_Predicate_Bag_Rows",
                                 "N_Evaluated", "N_Below_Guard", "N_Zero", "Graph_Status"]
    assert (cov["N_Evaluated"] + cov["N_Below_Guard"] == cov["N_Predicate_Bag_Rows"]).all()
    assert (cov["N_Zero"] <= cov["N_Evaluated"]).all()
    assert (cov["N_Zero"] > 0).all()            # the noise zone Zn: effect exactly 0
    assert (cov["N_Bags"] == s.n_bags).all()
    for _, row in cov.iterrows():
        valid = row["Seed"] in s.valid_seeds_per_class_[row["Class"]]
        assert valid == (row["Graph_Status"] in ("ok", "all_zero"))


def test_within_excludes_more_than_all(fitted_within, fitted_all):
    """'within' adds the class guard on top of the bagger's: never fewer exclusions."""
    w = fitted_within.per_class_coverage_.set_index(["Seed", "Class"])
    a = fitted_all.per_class_coverage_.set_index(["Seed", "Class"])
    assert (w["N_Below_Guard"] >= a["N_Below_Guard"]).all()
    assert (w["N_Below_Guard"] > a["N_Below_Guard"]).any()


def test_refit_resets_per_class_results(known_truth, fitted_within):
    s = copy.deepcopy(fitted_within)
    with quiet():
        s.fit(known_truth["X"], known_truth["est"].predict(known_truth["X"]),
              X_cal_natural=known_truth["X"] * 1.7 + 3.0)
    for k in ["A", "B", "C"]:
        pd.testing.assert_frame_equal(s.lrc_per_class_[k], fitted_within.lrc_per_class_[k],
                                      check_exact=True)
    pd.testing.assert_frame_equal(s.per_class_coverage_, fitted_within.per_class_coverage_)


def test_natural_per_class_is_none_without_x_cal_natural(known_truth):
    s = fit_smx(known_truth, known_truth["est"], per_class="within", natural=False)
    assert s.lrc_natural_per_class_ is None and s.lrc_per_class_


# ---------------------------------------------------------------------------
# Output labels (N4)
# ---------------------------------------------------------------------------

def test_estimator_without_classes_needs_explicit_output_labels(known_truth, fitted_within):
    bare = LinearSoftmaxStub(n_ch=5, n_zones=4, expose_classes=False)
    with pytest.raises(ValueError, match="has no 'classes_'"):
        fit_smx(known_truth, bare, per_class="within")
    s = fit_smx(known_truth, bare, per_class="within", per_class_output_labels=["A", "B", "C"])
    for k in ["A", "B", "C"]:
        pd.testing.assert_frame_equal(s.lrc_per_class_[k], fitted_within.lrc_per_class_[k],
                                      check_exact=True)


def test_predicted_labels_must_be_among_the_output_labels(known_truth):
    with pytest.raises(ValueError, match="not among the model output labels"):
        fit_smx(known_truth, known_truth["est"], per_class="all",
                per_class_output_labels=["x", "y", "z"])


def test_integer_classes_are_matched_as_strings(known_truth):
    est = LinearSoftmaxStub(classes=[0, 1, 2], n_ch=5, n_zones=4)
    s = fit_smx(known_truth, est, per_class="within")
    assert s.per_class_labels_ == ["0", "1", "2"]
    assert set(s.lrc_unique_per_class_) == {"0", "1", "2"}


def test_pls_like_regressor_with_two_dimensional_output(known_truth):
    est = LinearRegressorStub(n_ch=5, n_zones=4)
    s = fit_smx(known_truth, est, y=est.predict_labels(known_truth["X"]),
                perturbation_metric="mean_abs_diff", per_class="within",
                per_class_output_labels=["A", "B", "C"])
    # In the linear stub only zone j moves output j: each class sees its own zone
    # only; the other zones were evaluated with effect 0 and get LRC 0.
    for k, zone in zip("ABC", ["Za", "Zb", "Zc"]):
        zone_lrc = s.lrc_unique_per_class_[k].set_index("Zone")["Local_Reaching_Centrality"]
        assert set(zone_lrc[zone_lrc > 0].index) == {zone}
        assert set(zone_lrc.index) == {"Za", "Zb", "Zc", "Zn"}


def test_class_never_predicted_is_absent_in_within_and_present_in_all(known_truth, caplog):
    class FourOutputs(LinearSoftmaxStub):
        """Fourth class 'D' has a tiny, input-dependent probability: never predicted."""

        def predict_proba(self, X):
            p = super().predict_proba(X)
            d = 1e-6 * p[:, :1]
            return np.hstack([p * (1 - d), d])

    est = FourOutputs(n_ch=5, n_zones=4)
    est.classes_ = np.array(["A", "B", "C", "D"])
    with caplog.at_level(logging.WARNING, logger="smx.pipeline"):
        within = fit_smx(known_truth, est, per_class="within")
    assert within.per_class_min_samples_["D"] == 1
    assert "D" not in within.lrc_per_class_ and within.valid_seeds_per_class_["D"] == []
    assert any("Class 'D' produced no usable per-class graph" in r.getMessage() for r in caplog.records)
    assert {"A", "B", "C"} <= set(within.lrc_per_class_)
    every = fit_smx(known_truth, est, per_class="all")
    assert "D" in every.lrc_per_class_
