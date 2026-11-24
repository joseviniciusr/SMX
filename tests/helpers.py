"""Shared, training-free building blocks for the SMX test-suite.

Everything here is deterministic and independent of scikit-learn model
fitting, so expected values can be derived by hand or by the naive oracles
below.  Pytest fixtures that wrap these helpers live in ``conftest.py``.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from smx import SMX


# ---------------------------------------------------------------------------
# Deterministic estimator stubs
# ---------------------------------------------------------------------------

def _zone_weights(n_ch: int, n_zones: int, n_classes: int, gain: float) -> np.ndarray:
    """Zone ``j`` pushes output ``j`` (``j < n_classes``); later zones are noise."""
    W = np.zeros((n_ch * n_zones, n_classes))
    for j in range(n_classes):
        W[j * n_ch:(j + 1) * n_ch, j] = gain / n_ch
    return W


class LinearSoftmaxStub:
    """Classifier without training: ``p = softmax(X @ W)``.

    Zone ``j`` raises the logit of class ``j``.  With ``expose_classes=False``
    the ``classes_`` attribute is absent (exercises explicit output labels).
    """

    def __init__(self, classes: Sequence[str] = ("A", "B", "C"), n_ch: int = 5,
                 n_zones: int = 4, gain: float = 1.5, expose_classes: bool = True):
        self._classes = np.array(list(classes))
        self.W = _zone_weights(n_ch, n_zones, len(classes), gain)
        if expose_classes:
            self.classes_ = self._classes

    def predict_proba(self, X) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.W
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self._classes[self.predict_proba(X).argmax(axis=1)]


class LinearRegressorStub:
    """PLS-like regressor without training: ``predict`` returns ``X @ W`` (n, K).

    It has no ``classes_`` and no ``predict_proba``; the class of a sample is
    the ``argmax`` of the outputs, mapped to ``labels``.
    """

    def __init__(self, labels: Sequence[str] = ("A", "B", "C"), n_ch: int = 5,
                 n_zones: int = 4, gain: float = 1.5):
        self.labels = np.array(list(labels))
        self.W = _zone_weights(n_ch, n_zones, len(labels), gain)

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.W

    def predict_labels(self, X) -> np.ndarray:
        return self.labels[self.predict(X).argmax(axis=1)]


class ConstantProbaStub:
    """Classifier insensitive to its input (every perturbation shift is 0)."""

    def __init__(self, classes: Sequence[str] = ("A", "B", "C")):
        self.classes_ = np.array(list(classes))

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        p = np.linspace(1.0, 2.0, len(self.classes_))
        return np.tile(p / p.sum(), (n, 1))

    def predict(self, X) -> np.ndarray:
        # Deterministic, input-independent labels that still cover all classes.
        n = len(X)
        return self.classes_[np.arange(n) % len(self.classes_)]


# ---------------------------------------------------------------------------
# Synthetic data with known ground truth
# ---------------------------------------------------------------------------

def make_known_truth(seed: int = 7, n: int = 150, n_ch: int = 5,
                     classes: Sequence[str] = ("A", "B", "C"),
                     probs: Optional[Sequence[float]] = None,
                     shift: float = 2.5) -> Dict:
    """Scenario S1: every class has its own marker zone, plus one noise zone.

    Zones ``Za, Zb, Zc`` (one per class, in order) carry a ``+shift`` offset
    for the samples of that class; the last zone ``Zn`` is pure noise.
    """
    classes = list(classes)
    rng = np.random.default_rng(seed)
    if probs is None:
        probs = [0.45, 0.37, 0.18][: len(classes)] if len(classes) <= 3 else None
        if probs is not None:
            probs = np.asarray(probs) / np.sum(probs)
    y = rng.choice(classes, size=n, p=probs)
    zone_names = [f"Z{c.lower()}" for c in classes] + ["Zn"]
    blocks = []
    for k in classes + [None]:
        s = (y == k) * shift + rng.normal(0, 1, n) if k is not None else rng.normal(0, 1, n)
        blocks.append(s[:, None] * (1 + 0.1 * np.arange(n_ch)) + rng.normal(0, 0.3, (n, n_ch)))
    n_zones = len(zone_names)
    X = pd.DataFrame(np.hstack(blocks), columns=[str(float(i)) for i in range(n_zones * n_ch)])
    cuts = [(z, float(i * n_ch), float(i * n_ch + n_ch - 1)) for i, z in enumerate(zone_names)]
    return {"X": X, "y_true": y, "cuts": cuts, "zones": zone_names,
            "classes": classes, "n_ch": n_ch, "n_zones": n_zones}


def make_exclusion_truth(seed: int = 7, n: int = 150, n_ch: int = 5, shift: float = 2.5) -> Dict:
    """Scenario S2: class B is defined *by exclusion* (neither A nor C markers).

    Zones: ``Za`` (+shift for A), ``Zc`` (+shift for C), ``Zb`` and ``Zn`` noise.
    The stub logit of B is ``-(Za + Zc)``-like, so B's relevant zones are Za/Zc.
    """
    rng = np.random.default_rng(seed)
    y = rng.choice(["A", "B", "C"], size=n, p=[0.35, 0.35, 0.30])
    zone_names = ["Za", "Zb", "Zc", "Zn"]
    blocks = []
    for k in ["A", None, "C", None]:
        s = (y == k) * shift + rng.normal(0, 1, n) if k is not None else rng.normal(0, 1, n)
        blocks.append(s[:, None] * (1 + 0.1 * np.arange(n_ch)) + rng.normal(0, 0.3, (n, n_ch)))
    X = pd.DataFrame(np.hstack(blocks), columns=[str(float(i)) for i in range(4 * n_ch)])
    cuts = [(z, float(i * n_ch), float(i * n_ch + n_ch - 1)) for i, z in enumerate(zone_names)]
    est = LinearSoftmaxStub(n_ch=n_ch, n_zones=4)
    W = np.zeros((4 * n_ch, 3))
    W[0:n_ch, 0] = 1.5 / n_ch                      # Za -> A
    W[2 * n_ch:3 * n_ch, 2] = 1.5 / n_ch           # Zc -> C
    W[0:n_ch, 1] = -1.5 / n_ch                     # B = neither A ...
    W[2 * n_ch:3 * n_ch, 1] = -1.5 / n_ch          # ... nor C
    est.W = W
    return {"X": X, "y_true": y, "cuts": cuts, "zones": zone_names, "est": est,
            "classes": ["A", "B", "C"], "n_ch": n_ch, "n_zones": 4}


# ---------------------------------------------------------------------------
# Naive oracles (written independently of the library code)
# ---------------------------------------------------------------------------

def zone_columns(X: pd.DataFrame, start: float, end: float) -> List[str]:
    energies = pd.to_numeric(pd.Series(X.columns.astype(str)), errors="coerce").values
    lo, hi = min(start, end), max(start, end)
    return [c for c, e in zip(X.columns, energies) if lo <= e <= hi]


def oracle_abs_delta(estimator, X: pd.DataFrame, rows: Sequence[int], cols: Sequence[str],
                     use_proba: bool = True) -> np.ndarray:
    """|output(original) - output(zone set to calibration median)| for ``rows`` (n, K)."""
    X_eval = X.iloc[list(rows)].copy()
    X_pert = X_eval.copy()
    med = X[list(cols)].median(axis=0)
    for c in cols:
        X_pert[c] = med[c]
    f = estimator.predict_proba if use_proba else estimator.predict
    out0 = np.asarray(f(X_eval), dtype=float)
    out1 = np.asarray(f(X_pert), dtype=float)
    return np.abs(out0 - out1)


def oracle_lrc_chain(weights: Sequence[float]) -> List[float]:
    """Weighted LRC of every node of the chain ``n0 -> n1 -> ... -> nm``.

    Edge ``i`` (``n_i -> n_{i+1}``) has weight ``weights[i]``.  Definition
    (Mones et al. 2012, as normalised by networkx):

        LRC(i) = sum_{j > i} mean(w[i:j]) / mean(w) / (N - 1)

    i.e. the average edge weight of each path leaving ``i``, summed over the
    reachable nodes, divided by the mean edge weight of the graph and by
    ``N - 1``.  In a chain every path is unique, so no shortest-path logic
    is needed.
    """
    w = np.asarray(weights, dtype=float)
    mean_w = w.sum() / len(w)
    n_nodes = len(w) + 1
    out = []
    for i in range(n_nodes):
        acc = sum(w[i:j].sum() / (j - i) for j in range(i + 1, n_nodes))
        out.append(acc / mean_w / (n_nodes - 1))
    return out


@contextlib.contextmanager
def quiet():
    """Silence the library's progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# Small, silenced SMX fit used across the suite
# ---------------------------------------------------------------------------

SMALL_SMX = dict(quantiles=[0.25, 0.5, 0.75], n_repetitions=2, n_bags=5)


def fit_smx(data, estimator, *, y=None, natural=True, **kwargs):
    """Fit a small SMX on ``data`` (silenced); ``y`` defaults to ``estimator.predict``."""
    X = data["X"]
    if y is None:
        y = estimator.predict(X)
    params = {**SMALL_SMX, **kwargs}
    with quiet():
        smx = SMX(spectral_cuts=data["cuts"], estimator=estimator, **params)
        smx.fit(X, y, X_cal_natural=(X * 1.7 + 3.0) if natural else None)
    return smx


def add_constant_zone(data, estimator, value: float = 5.0, n_ch: int = 3):
    """Append a zone ``Zk`` that is constant in the whole dataset (B2).

    Returns new ``(data, estimator)``: the estimator gets zero weights on the
    new channels, so its predictions are unchanged.
    """
    import copy

    X = data["X"].copy()
    start = X.shape[1]
    names = [str(float(start + i)) for i in range(n_ch)]
    for c in names:
        X[c] = value
    est = copy.deepcopy(estimator)
    est.W = np.vstack([est.W, np.zeros((n_ch, est.W.shape[1]))])
    out = dict(data, X=X, cuts=data["cuts"] + [("Zk", float(start), float(start + n_ch - 1))],
               zones=data["zones"] + ["Zk"])
    return out, est
