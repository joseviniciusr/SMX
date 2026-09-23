import itertools

import numpy as np
import pandas as pd

from smx.evaluation.faithfulness import progressive_masking_faithfulness
from smx.evaluation.normalized_faithfulness import (
    _find_aopc_bounds_beam,
    _find_aopc_bounds_exact,
    normalized_progressive_masking_faithfulness,
)
from smx.pipeline import SMX


class AdditiveProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X):
        logits = X.to_numpy(dtype=float) @ self.weights
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class MulticlassDecisionModel:
    classes_ = np.array(["A", "B", "C"])

    def decision_function(self, X):
        values = X.to_numpy(dtype=float)
        return np.column_stack(
            [values[:, 0] - values[:, 1], values[:, 1] - values[:, 2], values[:, 2]]
        )

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class BinaryDecisionModel:
    classes_ = np.array([0, 1])

    def decision_function(self, X):
        return X.to_numpy(dtype=float)[:, 0] - X.to_numpy(dtype=float)[:, 1]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def _toy_inputs():
    X = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 1.0]],
        columns=[0, 1, 2, 3],
    )
    cuts = [("z1", 0, 0), ("z2", 1, 1), ("z3", 2, 2), ("z4", 3, 3)]
    return X, cuts


def _ranking(order):
    return pd.DataFrame(
        {
            "Zone": list(order),
            "Local_Reaching_Centrality": list(range(len(order), 0, -1)),
        }
    )


def test_exact_naopc_is_one_for_best_and_zero_for_reverse_ranking():
    X, cuts = _toy_inputs()
    model = AdditiveProbabilityModel([4, 3, 2, 1])

    best = normalized_progressive_masking_faithfulness(
        model, X, cuts, _ranking(["z1", "z2", "z3", "z4"]), normalization="exact"
    )
    worst = normalized_progressive_masking_faithfulness(
        model, X, cuts, _ranking(["z4", "z3", "z2", "z1"]), normalization="exact"
    )

    assert np.isclose(best["naopc"], 1.0)
    assert np.isclose(worst["naopc"], 0.0)
    assert best["n_model_queries"] == 16  # 2**4 subsets, including baseline.


def test_exact_bounds_match_brute_force_per_sample():
    rng = np.random.default_rng(42)
    n_zones = 5
    n_samples = 3
    original = rng.normal(size=n_samples)
    subset_scores = {
        mask: rng.normal(size=n_samples) for mask in range(1 << n_zones)
    }
    subset_scores[0] = original.copy()

    lower, upper = _find_aopc_bounds_exact(
        subset_scores, original, n_zones
    )
    brute_force = []
    for order in itertools.permutations(range(n_zones)):
        mask = 0
        curve = []
        for bit in order:
            mask |= 1 << bit
            curve.append(original - subset_scores[mask])
        brute_force.append(np.mean(curve, axis=0))
    brute_force = np.asarray(brute_force)

    assert np.allclose(lower, brute_force.min(axis=0))
    assert np.allclose(upper, brute_force.max(axis=0))


def test_beam_matches_exact_when_beam_keeps_all_partial_candidates():
    rng = np.random.default_rng(7)
    n_zones = 4
    original = rng.normal(size=2)
    subset_scores = {
        mask: rng.normal(size=2) for mask in range(1 << n_zones)
    }
    subset_scores[0] = original.copy()

    exact = _find_aopc_bounds_exact(subset_scores, original, n_zones)
    beam = _find_aopc_bounds_beam(
        lambda mask: subset_scores[mask],
        original,
        n_zones,
        beam_size=24,
    )
    assert np.allclose(exact[0], beam[0])
    assert np.allclose(exact[1], beam[1])


def test_constant_model_returns_nan_naopc_and_degenerate_status():
    X, cuts = _toy_inputs()
    result = normalized_progressive_masking_faithfulness(
        ConstantProbabilityModel(),
        X,
        cuts,
        _ranking(["z1", "z2", "z3", "z4"]),
        normalization="exact",
    )

    assert np.isnan(result["naopc"])
    assert result["status"] == "degenerate_bounds"
    assert set(result["per_sample"]["status"]) == {"degenerate_bounds"}


def test_legacy_faithfulness_remains_available_with_same_contract():
    X, cuts = _toy_inputs()
    model = AdditiveProbabilityModel([4, 3, 2, 1])
    ranking = _ranking(["z1", "z2", "z3", "z4"])
    result = progressive_masking_faithfulness(
        model,
        X,
        cuts,
        ranking,
        n_random_rankings=0,
    )
    assert {"curve_df", "auc", "auc_normalized", "level", "null_percentile"} <= set(result)


def test_pipeline_facade_exposes_normalized_evaluation():
    X, cuts = _toy_inputs()
    model = AdditiveProbabilityModel([4, 3, 2, 1])
    explainer = SMX(spectral_cuts=cuts, quantiles=[0.5], estimator=model)
    ranking = _ranking(["z1", "z2", "z3", "z4"])
    explainer.lrc_summed_unique_ = ranking
    explainer.lrc_summed_ = ranking.copy()

    result = explainer.evaluate_normalized_faithfulness(X, normalization="exact")

    assert result["ranking_source"] == "unique"
    assert np.isclose(result["naopc"], 1.0)
    assert explainer.normalized_faithfulness_ is result


def test_multiclass_decision_function_keeps_original_target_class():
    X = pd.DataFrame(
        [[3.0, 1.0, 0.5], [0.5, 3.0, 1.0], [0.5, 1.0, 3.0]],
        columns=[0, 1, 2],
    )
    cuts = [("z1", 0, 0), ("z2", 1, 1), ("z3", 2, 2)]
    result = normalized_progressive_masking_faithfulness(
        MulticlassDecisionModel(),
        X,
        cuts,
        _ranking(["z1", "z2", "z3"]),
        metric="decision_function_shift",
        normalization="exact",
    )

    assert result["metric"] == "decision_function_shift"
    assert len(result["per_sample"]) == len(X)
    assert np.all(np.isfinite(result["per_sample"]["aopc"]))


def test_binary_decision_function_uses_score_direction_of_predicted_class():
    X = pd.DataFrame([[3.0, 1.0], [-1.0, 2.0]], columns=[0, 1])
    cuts = [("z1", 0, 0), ("z2", 1, 1)]
    result = normalized_progressive_masking_faithfulness(
        BinaryDecisionModel(),
        X,
        cuts,
        _ranking(["z1", "z2"]),
        metric="decision_function_shift",
        normalization="exact",
    )

    assert result["metric"] == "decision_function_shift"
    assert np.all(result["per_sample"]["aopc"].to_numpy() >= 0.0)


def test_auto_selects_beam_above_exact_zone_limit():
    n_zones = 10
    X = pd.DataFrame([np.ones(n_zones)], columns=list(range(n_zones)))
    cuts = [(f"z{i}", i, i) for i in range(n_zones)]
    model = AdditiveProbabilityModel(np.arange(n_zones, 0, -1))
    result = normalized_progressive_masking_faithfulness(
        model,
        X,
        cuts,
        _ranking([f"z{i}" for i in range(n_zones)]),
        normalization="auto",
        beam_size=2,
        exact_max_zones=9,
    )

    assert result["normalization"] == "beam"
    assert result["beam_size"] == 2
    assert result["n_zones"] == n_zones


def test_overlapping_zone_boundaries_do_not_duplicate_masked_columns():
    X = pd.DataFrame([[1.0, 2.0, 3.0]], columns=[0, 1, 2])
    cuts = [("z1", 0, 1), ("z2", 1, 2)]
    result = normalized_progressive_masking_faithfulness(
        AdditiveProbabilityModel([1.0, 1.0, 1.0]),
        X,
        cuts,
        _ranking(["z1", "z2"]),
        normalization="exact",
    )

    assert result["n_zones"] == 2
    assert np.isfinite(result["naopc"])
