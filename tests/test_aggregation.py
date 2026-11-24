"""ZoneAggregator: the PCA explained variance of a constant zone (B2)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from smx.zones.aggregation import ZoneAggregator


def _zones():
    rng = np.random.default_rng(0)
    varying = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    constant = pd.DataFrame(np.full((30, 3), 5.0), columns=list("efg"))
    return {"Zv": varying, "Zk": constant}


def test_constant_zone_has_zero_variance_and_a_warning(caplog):
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no 0/0 RuntimeWarning leaks out
        with caplog.at_level(logging.WARNING, logger="smx.zones.aggregation"):
            agg = ZoneAggregator(method="pca").fit(_zones())
    assert agg.pca_info_["Zk"]["variance_explained"] == 0.0
    assert 0 < agg.pca_info_["Zv"]["variance_explained"] <= 1
    msgs = [r.getMessage() for r in caplog.records]
    assert any("'Zk' is constant" in m for m in msgs)
    assert not any("'Zv'" in m for m in msgs)


def test_constant_zone_scores_are_finite():
    scores = ZoneAggregator(method="pca").fit_transform(_zones())
    assert np.isfinite(scores.to_numpy(dtype=float)).all()
