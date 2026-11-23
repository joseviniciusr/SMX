"""Public API contract: removed names/kwargs and constructor validation."""

from __future__ import annotations

import numpy as np
import pytest

import smx
from smx import SMX
from smx.predicates.metrics import PerturbationMetric
from helpers import LinearRegressorStub, LinearSoftmaxStub, make_known_truth


@pytest.fixture(scope="module")
def cuts():
    return make_known_truth(seed=7)["cuts"]


def _smx(cuts, **kwargs):
    params = dict(spectral_cuts=cuts, quantiles=[0.5], estimator=LinearSoftmaxStub())
    params.update(kwargs)
    return SMX(**params)


# ---------------------------------------------------------------------------
# Names and keyword arguments that were removed must stay removed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module, name", [
    ("smx", "CovarianceMetric"),
    ("smx.predicates", "CovarianceMetric"),
    ("smx.predicates.metrics", "CovarianceMetric"),
])
def test_removed_names_are_gone(module, name):
    import importlib
    assert not hasattr(importlib.import_module(module), name)
    assert name not in smx.__all__


@pytest.mark.parametrize("kwarg, value", [
    ("metric", "perturbation"),
    ("class_threshold", 0.5),
    ("covariance_threshold", 0.01),
])
def test_removed_init_kwargs_raise_type_error(cuts, kwarg, value):
    with pytest.raises(TypeError, match=kwarg):
        _smx(cuts, **{kwarg: value})


def test_fit_rejects_removed_y_pred_cal_keyword(cuts):
    data = make_known_truth(seed=7)
    with pytest.raises(TypeError, match="y_pred_cal"):
        _smx(cuts).fit(data["X"], y_pred_cal=np.zeros(len(data["X"])))


def test_new_public_names_are_exported():
    from smx.graph import aggregate_lrc_across_seeds_per_class as a
    assert smx.aggregate_lrc_across_seeds_per_class is a
    assert "aggregate_lrc_across_seeds_per_class" in smx.__all__


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_estimator_is_required(cuts):
    with pytest.raises(ValueError, match="estimator is required"):
        SMX(spectral_cuts=cuts, quantiles=[0.5], estimator=None)


@pytest.mark.parametrize("per_class, expected", [
    (False, None), (None, None), (True, "within"), ("within", "within"), ("all", "all"),
])
def test_per_class_mode_resolution(cuts, per_class, expected):
    assert _smx(cuts, per_class=per_class).per_class_mode_ == expected


@pytest.mark.parametrize("per_class", ["WITHIN", "both", 1, 0.5, "true"])
def test_per_class_invalid_mode(cuts, per_class):
    with pytest.raises(ValueError, match="per_class must be"):
        _smx(cuts, per_class=per_class)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, 0, -3, True, False, "0.2", None])
def test_per_class_min_samples_invalid(cuts, value):
    with pytest.raises(ValueError, match="per_class_min_samples"):
        _smx(cuts, per_class="within", per_class_min_samples=value)


@pytest.mark.parametrize("value", [0.2, 1.0, 1e-6, 1, 25, np.int64(5), np.float64(0.3)])
def test_per_class_min_samples_valid(cuts, value):
    assert _smx(cuts, per_class="within", per_class_min_samples=value).per_class_min_samples == value


def test_per_class_min_samples_is_validated_even_when_per_class_is_off(cuts):
    # An invalid value is a programming error regardless of the mode.
    with pytest.raises(ValueError, match="per_class_min_samples"):
        _smx(cuts, per_class=False, per_class_min_samples=2.0)


@pytest.mark.parametrize("metric", [
    "prediction_change_rate", "accuracy_drop", "f1_drop", "decision_function_shift",
    "mean_diff", "mean_relative_dev",
])
def test_per_class_rejects_non_decomposable_metrics(cuts, metric):
    est = LinearSoftmaxStub()
    est.decision_function = est.predict_proba  # make decision_function_shift constructible
    with pytest.raises(ValueError, match="decomposes by output column"):
        _smx(cuts, estimator=est, perturbation_metric=metric, per_class="within")


@pytest.mark.parametrize("metric", sorted(PerturbationMetric._PER_CLASS_METRICS))
def test_per_class_accepts_decomposable_metrics(cuts, metric):
    est = LinearSoftmaxStub() if metric == "probability_shift" else LinearRegressorStub()
    assert _smx(cuts, estimator=est, perturbation_metric=metric, per_class="all").per_class_mode_ == "all"


@pytest.mark.parametrize("labels", [[], ["A", "A", "B"], ("x", "x")])
def test_per_class_output_labels_must_be_unique_and_non_empty(cuts, labels):
    with pytest.raises(ValueError, match="per_class_output_labels"):
        _smx(cuts, per_class="within", per_class_output_labels=labels)


def test_per_class_attributes_are_none_before_fit(cuts):
    s = _smx(cuts, per_class="within")
    for attr in ("per_class_labels_", "per_class_min_samples_", "lrc_per_class_",
                 "lrc_unique_per_class_", "lrc_natural_per_class_",
                 "graphs_by_class_by_seed_", "valid_seeds_per_class_", "per_class_coverage_"):
        assert getattr(s, attr) is None, attr


# ---------------------------------------------------------------------------
# PerturbationMetric validation (used directly by power users)
# ---------------------------------------------------------------------------

def _metric(**kwargs):
    data = make_known_truth(seed=7)
    params = dict(estimator=LinearSoftmaxStub(), Xcalclass_prep=data["X"],
                  predicates_df=None, spectral_cuts=data["cuts"],
                  metric="probability_shift")
    params.update(kwargs)
    return PerturbationMetric(**params)


@pytest.mark.parametrize("kwargs, match", [
    (dict(per_class="both", class_labels=["A"]), "per_class must be one of"),
    (dict(per_class="all"), "requires class_labels"),
    (dict(per_class="all", class_labels=[]), "requires class_labels"),
    (dict(per_class="all", class_labels=["A", "A"]), "must be unique"),
    (dict(per_class="within", class_labels=["A", "B"]), "requires per_class_min_samples"),
    (dict(per_class="within", class_labels=["A", "B"], per_class_min_samples={"A": 3}),
     r"missing the classes \['B'\]"),
    (dict(per_class="within", class_labels=["A"], per_class_min_samples={"A": 0}), "int >= 1"),
    (dict(per_class="within", class_labels=["A"], per_class_min_samples={"A": 2.5}), "int >= 1"),
    (dict(per_class="all", class_labels=["A"], metric="prediction_change_rate"),
     "decomposes by output column"),
])
def test_perturbation_metric_per_class_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _metric(**kwargs)


def test_perturbation_metric_casts_labels_and_guard_keys_to_str():
    m = _metric(per_class="within", class_labels=[0, 1], per_class_min_samples={0: 2, "1": 3})
    assert m.class_labels == ["0", "1"]
    assert m.per_class_min_samples == {"0": 2, "1": 3}
    assert m.per_class_columns == {"0": "Perturbation_0", "1": "Perturbation_1"}
    assert _metric().per_class_columns == {}
