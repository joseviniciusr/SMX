"""PerturbationMetric: exact values against naive oracles and exact identities.

The estimators are training-free stubs (``helpers.py``), so every expected
number is recomputed independently here instead of being trusted from the
library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smx.predicates.bagging import PredicateBagger
from smx.predicates.generation import PredicateGenerator
from smx.predicates.metrics import PerturbationMetric
from smx.zones.aggregation import ZoneAggregator
from smx.zones.extraction import extract_spectral_zones
from helpers import (
    LinearRegressorStub,
    LinearSoftmaxStub,
    make_known_truth,
    oracle_abs_delta,
    quiet,
    zone_columns,
)

RTOL = 1e-12


def _setup(data, labels_of):
    zones = extract_spectral_zones(data["X"], data["cuts"])
    scores = ZoneAggregator(method="pca").fit_transform(zones)
    predicates = PredicateGenerator(quantiles=[0.25, 0.5, 0.75]).fit(scores).predicates_df_
    with quiet():
        bags = PredicateBagger(random_seed=0, n_bags=3).run(scores, predicates)
    y_pred = pd.Series(labels_of(data["X"])).astype(str).values
    for preds in bags.values():
        for df in preds.values():
            df["Class_Predicted"] = y_pred[df["Sample_Index"].values]
    cut_of = {name: (s, e) for name, s, e in data["cuts"]}
    zone_cols = {r["rule"]: zone_columns(data["X"], *cut_of[r["zone"]]) for _, r in predicates.iterrows()}
    return {"data": data, "predicates": predicates, "bags": bags, "y_pred": y_pred, "zone_cols": zone_cols}


@pytest.fixture(scope="module")
def s1():
    data = make_known_truth(seed=7)
    est = LinearSoftmaxStub(n_ch=data["n_ch"], n_zones=data["n_zones"])
    ctx = _setup(data, est.predict)
    ctx["est"] = est
    return ctx


def _compute(ctx, est=None, metric="probability_shift", bags=None, **kwargs):
    params = dict(
        estimator=est if est is not None else ctx["est"],
        Xcalclass_prep=ctx["data"]["X"],
        predicates_df=ctx["predicates"],
        spectral_cuts=ctx["data"]["cuts"],
        perturbation_mode="median",
        stats_source="full",
        metric=metric,
        normalize_by_zone_size=True,
    )
    params.update(kwargs)
    return PerturbationMetric(**params).compute(bags if bags is not None else ctx["bags"])


def _guard(ctx, frac=0.2):
    counts = pd.Series(ctx["y_pred"]).value_counts()
    return {k: max(1, int(frac * counts.get(k, 0))) for k in ["A", "B", "C"]}


def _rows(results):
    for bag, df in results.items():
        if bag.startswith("__"):
            continue
        for _, row in df.iterrows():
            yield bag, row


# ---------------------------------------------------------------------------
# Global value
# ---------------------------------------------------------------------------

def test_global_probability_shift_matches_oracle(s1):
    res = _compute(s1)
    n_checked = 0
    for bag, row in _rows(res):
        df = s1["bags"][bag][row["Predicate"]]
        cols = s1["zone_cols"][row["Predicate"]]
        D = oracle_abs_delta(s1["est"], s1["data"]["X"], df["Sample_Index"].values, cols)
        expected = (D.sum(axis=1) / 2.0).mean() / len(cols)
        assert row["Perturbation"] == pytest.approx(expected, rel=RTOL, abs=1e-300)
        n_checked += 1
    assert n_checked > 50


def test_each_bag_is_sorted_descending_by_global_value(s1):
    for bag, df in _compute(s1, per_class="all", class_labels=["A", "B", "C"]).items():
        if not bag.startswith("__"):
            assert df["Perturbation"].is_monotonic_decreasing


@pytest.mark.parametrize("mode", ["within", "all"])
def test_global_column_and_row_order_are_unchanged_by_per_class(s1, mode):
    base = _compute(s1)
    kwargs = dict(per_class=mode, class_labels=["A", "B", "C"])
    if mode == "within":
        kwargs["per_class_min_samples"] = _guard(s1)
    other = _compute(s1, **kwargs)
    assert base.keys() == other.keys()
    for bag in base:
        if bag.startswith("__"):
            pd.testing.assert_frame_equal(base[bag], other[bag], check_exact=True)
            continue
        assert list(other[bag].columns) == ["Predicate", "Perturbation",
                                            "Perturbation_A", "Perturbation_B", "Perturbation_C"]
        pd.testing.assert_frame_equal(base[bag], other[bag][["Predicate", "Perturbation"]],
                                      check_exact=True)
    assert all(list(df.columns) == ["Predicate", "Perturbation"]
               for bag, df in base.items() if not bag.startswith("__"))


# ---------------------------------------------------------------------------
# 'all' mode: exact identities
# ---------------------------------------------------------------------------

def test_all_mode_sums_to_twice_the_tvd(s1):
    """sum_k mean|dp_k| = 2 * mean TVD, predicate by predicate (P2)."""
    res = _compute(s1, per_class="all", class_labels=["A", "B", "C"])
    checked = 0
    for _, row in _rows(res):
        # In 'all' mode NaN can only mean an exact 0 (|dp_k| >= 0, no guard).
        per_class = row[["Perturbation_A", "Perturbation_B", "Perturbation_C"]].astype(float).fillna(0.0)
        assert per_class.sum() == pytest.approx(2.0 * row["Perturbation"], rel=RTOL, abs=1e-300)
        checked += row["Perturbation"] > 0
    assert checked > 30


def test_all_mode_matches_oracle_column_by_column(s1):
    res = _compute(s1, per_class="all", class_labels=["A", "B", "C"])
    for bag, row in _rows(res):
        df = s1["bags"][bag][row["Predicate"]]
        cols = s1["zone_cols"][row["Predicate"]]
        D = oracle_abs_delta(s1["est"], s1["data"]["X"], df["Sample_Index"].values, cols)
        for j, k in enumerate("ABC"):
            expected = D[:, j].mean() / len(cols)
            got = row[f"Perturbation_{k}"]
            if expected > 0:
                assert got == pytest.approx(expected, rel=RTOL)
            else:
                assert np.isnan(got)


def test_all_mode_mean_abs_diff_2d_sums_to_k_times_global():
    data = make_known_truth(seed=7)
    est = LinearRegressorStub(n_ch=data["n_ch"], n_zones=data["n_zones"])
    ctx = _setup(data, est.predict_labels)
    res = _compute(ctx, est=est, metric="mean_abs_diff", per_class="all", class_labels=["A", "B", "C"])
    checked = 0
    for _, row in _rows(res):
        per_class = row[["Perturbation_A", "Perturbation_B", "Perturbation_C"]].astype(float).fillna(0.0)
        assert per_class.sum() == pytest.approx(3.0 * row["Perturbation"], rel=RTOL, abs=1e-300)
        checked += row["Perturbation"] > 0
    assert checked > 30


def test_binary_all_mode_equals_the_global_value():
    """With K = 2, |dp_A| = |dp_B| = TVD, so 'all' carries no class information."""
    data = make_known_truth(seed=11, classes=("A", "B"))
    est = LinearSoftmaxStub(classes=("A", "B"), n_ch=data["n_ch"], n_zones=data["n_zones"])
    ctx = _setup(data, est.predict)
    res = _compute(ctx, est=est, per_class="all", class_labels=["A", "B"])
    checked = 0
    for _, row in _rows(res):
        if row["Perturbation"] == 0:
            assert np.isnan(row["Perturbation_A"]) and np.isnan(row["Perturbation_B"])
            continue
        assert row["Perturbation_A"] == pytest.approx(row["Perturbation"], rel=1e-10)
        assert row["Perturbation_B"] == pytest.approx(row["Perturbation"], rel=1e-10)
        checked += 1
    assert checked > 20


# ---------------------------------------------------------------------------
# 'within' mode
# ---------------------------------------------------------------------------

def test_within_mode_matches_oracle_including_the_guard(s1):
    guard = _guard(s1)
    res = _compute(s1, per_class="within", class_labels=["A", "B", "C"], per_class_min_samples=guard)
    n_value = n_guarded = 0
    for bag, row in _rows(res):
        df = s1["bags"][bag][row["Predicate"]]
        cols = s1["zone_cols"][row["Predicate"]]
        idx = df["Sample_Index"].values
        D = oracle_abs_delta(s1["est"], s1["data"]["X"], idx, cols)
        predicted = s1["y_pred"][idx]
        for j, k in enumerate("ABC"):
            got = row[f"Perturbation_{k}"]
            rows = predicted == k
            if len(np.unique(idx[rows])) < guard[k]:
                assert np.isnan(got)
                n_guarded += 1
                continue
            expected = D[rows, j].mean() / len(cols)
            if expected > 0:
                assert got == pytest.approx(expected, rel=RTOL)
                n_value += 1
            else:
                assert np.isnan(got)
    assert n_value > 50 and n_guarded > 0


def _hand_bag(ctx, sample_index, predicted):
    rule = ctx["predicates"]["rule"].iloc[0]
    df = pd.DataFrame({"Zone_Sum": 0.0, "Sample_Index": np.asarray(sample_index),
                       "Class_Predicted": list(predicted)})
    return rule, {"Bag_1": {rule: df}}


def test_within_guard_counts_distinct_samples_not_rows(s1):
    # Four rows but only two distinct samples, all predicted 'A'.
    rule, bags = _hand_bag(s1, [5, 5, 5, 9], "AAAA")
    common = dict(per_class="within", class_labels=["A", "B", "C"], bags=bags)
    ok = _compute(s1, per_class_min_samples={"A": 2, "B": 1, "C": 1}, **common)["Bag_1"]
    blocked = _compute(s1, per_class_min_samples={"A": 3, "B": 1, "C": 1}, **common)["Bag_1"]
    assert np.isfinite(ok.loc[0, "Perturbation_A"])
    assert np.isnan(blocked.loc[0, "Perturbation_A"])
    # No sample predicted B or C in the bag -> NaN whatever the guard.
    assert np.isnan(ok.loc[0, "Perturbation_B"]) and np.isnan(ok.loc[0, "Perturbation_C"])
    # The mean uses every row (duplicates included), like the global value.
    cols = s1["zone_cols"][rule]
    D = oracle_abs_delta(s1["est"], s1["data"]["X"], [5, 5, 5, 9], cols)
    assert ok.loc[0, "Perturbation_A"] == pytest.approx(D[:, 0].mean() / len(cols), rel=RTOL)


def test_within_requires_the_class_predicted_column(s1):
    rule = s1["predicates"]["rule"].iloc[0]
    bags = {"Bag_1": {rule: pd.DataFrame({"Zone_Sum": 0.0, "Sample_Index": [1, 2, 3]})}}
    with pytest.raises(ValueError, match="Class_Predicted"):
        _compute(s1, bags=bags, per_class="within", class_labels=["A", "B", "C"],
                 per_class_min_samples={"A": 1, "B": 1, "C": 1})


# ---------------------------------------------------------------------------
# Exclusion rule, label mapping, output shape, normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["within", "all"])
def test_zero_effect_predicates_are_nan_per_class_but_zero_globally(s1, mode):
    """The noise zone Zn has no weight in the stub: its effect is exactly 0 (N9)."""
    kwargs = dict(per_class=mode, class_labels=["A", "B", "C"])
    if mode == "within":
        kwargs["per_class_min_samples"] = {"A": 1, "B": 1, "C": 1}
    res = _compute(s1, **kwargs)
    zn = [row for _, row in _rows(res) if row["Predicate"].startswith("Zn ")]
    assert zn
    for row in zn:
        assert row["Perturbation"] == 0.0
        assert row[["Perturbation_A", "Perturbation_B", "Perturbation_C"]].isna().all()


def test_per_class_columns_follow_labels_not_output_position(s1):
    """Permuting the estimator's output columns (and classes_) permutes nothing per label."""
    est = s1["est"]
    perm = [2, 0, 1]
    permuted = LinearSoftmaxStub(classes=["C", "A", "B"], n_ch=5, n_zones=4)
    permuted.W = est.W[:, perm]
    a = _compute(s1, per_class="all", class_labels=list(est.classes_))
    b = _compute(s1, est=permuted, per_class="all", class_labels=list(permuted.classes_))
    for bag in a:
        if bag.startswith("__"):
            continue
        for k in "ABC":
            np.testing.assert_allclose(b[bag][f"Perturbation_{k}"].values,
                                       a[bag][f"Perturbation_{k}"].values, rtol=1e-12)


def test_one_dimensional_output_cannot_be_split_by_class(s1):
    class OneColumn:
        def predict(self, X):
            return np.asarray(X, dtype=float)[:, :5].sum(axis=1)

    with pytest.raises(ValueError, match="one model output column per class"):
        _compute(s1, est=OneColumn(), metric="mean_abs_diff", per_class="all",
                 class_labels=["A", "B", "C"])


def test_zone_size_normalisation_applies_equally_to_global_and_per_class(s1):
    kwargs = dict(per_class="all", class_labels=["A", "B", "C"])
    raw = _compute(s1, normalize_by_zone_size=False, **kwargs)
    norm = _compute(s1, normalize_by_zone_size=True, zone_size_exponent=0.5, **kwargs)
    n_ch = s1["data"]["n_ch"]
    for bag in raw:
        if bag.startswith("__"):
            continue
        r = raw[bag].set_index("Predicate")
        n = norm[bag].set_index("Predicate").loc[r.index]
        for col in ["Perturbation", "Perturbation_A", "Perturbation_B", "Perturbation_C"]:
            np.testing.assert_allclose(n[col].values, r[col].values / n_ch ** 0.5, rtol=1e-12)


def test_empty_bag_keeps_the_per_class_schema(s1):
    res = _compute(s1, bags={"Bag_1": {}}, per_class="all", class_labels=["A", "B", "C"])
    assert list(res["Bag_1"].columns) == ["Predicate", "Perturbation",
                                          "Perturbation_A", "Perturbation_B", "Perturbation_C"]
    assert res["Bag_1"].empty
