"""Benchmark exact and beam NAOPC on additive synthetic spectral zones.

Run from the repository root with::

    python experiments/naopc/beam_convergence.py

The generated CSV is intentionally small and is suitable for checking the
beam-size choice before running the larger real-dataset experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smx.evaluation.normalized_faithfulness import (
    normalized_progressive_masking_faithfulness,
)


class AdditiveProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = X.to_numpy(dtype=float) @ self.weights
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class InteractionProbabilityModel(AdditiveProbabilityModel):
    """Synthetic model whose zone interactions make beam search non-trivial."""

    def __init__(self, n_zones: int):
        super().__init__(np.linspace(0.2, 1.5, n_zones))
        rng = np.random.default_rng(7 + n_zones)
        matrix = rng.normal(0.0, 1.0, size=(n_zones, n_zones))
        self.interactions = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(self.interactions, 0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy(dtype=float)
        logits = values @ self.weights
        logits += 0.6 * np.einsum("bi,ij,bj->b", values, self.interactions, values)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])


def make_case(n_zones: int, n_samples: int = 8):
    rng = np.random.default_rng(42 + n_zones)
    weights = np.linspace(2.0, 0.2, n_zones)
    X = pd.DataFrame(
        rng.normal(loc=0.8, scale=0.25, size=(n_samples, n_zones)),
        columns=list(range(n_zones)),
    )
    cuts = [(f"z{i}", i, i) for i in range(n_zones)]
    ranking = pd.DataFrame(
        {
            "Zone": [f"z{i}" for i in range(n_zones)],
            "Local_Reaching_Centrality": weights,
        }
    )
    return X, cuts, ranking, AdditiveProbabilityModel(weights)


def make_interaction_case(n_zones: int, n_samples: int = 8):
    rng = np.random.default_rng(100 + n_zones)
    X = pd.DataFrame(
        np.ones((n_samples, n_zones)), columns=list(range(n_zones))
    )
    cuts = [(f"z{i}", i, i) for i in range(n_zones)]
    order = rng.permutation(n_zones)
    ranking = pd.DataFrame(
        {
            "Zone": [f"z{i}" for i in order],
            "Local_Reaching_Centrality": np.arange(n_zones, 0, -1),
        }
    )
    return X, cuts, ranking, InteractionProbabilityModel(n_zones)


def main() -> None:
    rows = []
    for scenario, builder in (
        ("additive", make_case),
        ("interaction", make_interaction_case),
    ):
        for n_zones in (4, 6, 8, 9):
            X, cuts, ranking, model = builder(n_zones)
            exact = normalized_progressive_masking_faithfulness(
                model, X, cuts, ranking, normalization="exact"
            )
            for beam_size in (1, 2, 5, 10, 20, 50):
                beam = normalized_progressive_masking_faithfulness(
                    model,
                    X,
                    cuts,
                    ranking,
                    normalization="beam",
                    beam_size=beam_size,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "n_zones": n_zones,
                        "beam_size": beam_size,
                        "exact_naopc": exact["naopc"],
                        "beam_naopc": beam["naopc"],
                        "absolute_error": abs(beam["naopc"] - exact["naopc"]),
                        "exact_runtime_seconds": exact["runtime_seconds"],
                        "beam_runtime_seconds": beam["runtime_seconds"],
                        "exact_model_queries": exact["n_model_queries"],
                        "beam_model_queries": beam["n_model_queries"],
                    }
                )

    output_path = Path(__file__).with_name("beam_convergence.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
