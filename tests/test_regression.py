"""Cross-version regression of the *global* SMX output.

``tests/data/golden_global.json`` was generated from the code **before** the
per-class feature existed (SMX 0.1.8).  Every later version must reproduce it:
same row order and floats within ``rtol=1e-12`` (tolerates BLAS differences
between machines, nothing else).  Regenerate only on purpose with::

    pytest tests/test_regression.py --update-golden

Note on ``proba_no_varexp``: the stub's noise zone has an effect of exactly 0,
so the graph has zero-weight edges stored as Python floats and the known Q6
defect of ``compute_lrc`` sets every predicate's LRC to 0. The golden freezes
that output on purpose (it is today's behaviour); fixing Q6 must change this
configuration, and only this one, and then requires ``--update-golden``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers import LinearRegressorStub, LinearSoftmaxStub, fit_smx, make_known_truth

GOLDEN = Path(__file__).parent / "data" / "golden_global.json"
RTOL = 1e-12


# ---------------------------------------------------------------------------
# Configurations frozen in the golden file
# ---------------------------------------------------------------------------

def _s1():
    data = make_known_truth(seed=7)
    return data, LinearSoftmaxStub(n_ch=data["n_ch"], n_zones=data["n_zones"])


def _config(name):
    if name == "proba_varexp":
        data, est = _s1()
        return fit_smx(data, est, perturbation_metric="probability_shift", var_exp=True)
    if name == "proba_no_varexp":
        data, est = _s1()
        return fit_smx(data, est, perturbation_metric="probability_shift", var_exp=False)
    if name == "mean_abs_diff_2d":
        data = make_known_truth(seed=7)
        est = LinearRegressorStub(n_ch=data["n_ch"], n_zones=data["n_zones"])
        return fit_smx(data, est, y=est.predict_labels(data["X"]),
                       perturbation_metric="mean_abs_diff")
    if name == "binary":
        data = make_known_truth(seed=11, classes=("A", "B"))
        est = LinearSoftmaxStub(classes=("A", "B"), n_ch=data["n_ch"], n_zones=data["n_zones"])
        return fit_smx(data, est, perturbation_metric="probability_shift")
    raise KeyError(name)


CONFIGS = ["proba_varexp", "proba_no_varexp", "mean_abs_diff_2d", "binary"]


# ---------------------------------------------------------------------------
# Snapshot (JSON-safe; floats round-trip exactly through repr)
# ---------------------------------------------------------------------------

def _cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None if v is None else float("nan")
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if v is pd.NA:
        return None
    return v


def _table(df):
    return {"columns": [str(c) for c in df.columns],
            "rows": [[_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]}


def snapshot(smx):
    out = {"lrc_": _table(smx.lrc_), "lrc_unique_": _table(smx.lrc_unique_),
           "lrc_natural_": _table(smx.lrc_natural_), "valid_seeds_": list(smx.valid_seeds_),
           "graphs": {}}
    for seed, g in smx.graphs_by_seed_.items():
        out["graphs"][str(seed)] = {
            "nodes": sorted([n, a.get("node_type")] for n, a in g.nodes(data=True)),
            "edges": sorted([u, v, float(d["weight"])] for u, v, d in g.edges(data=True)),
        }
    return out


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)
    return a == b


def assert_snapshot_equal(got, exp, where=""):
    if isinstance(exp, dict):
        assert set(got) == set(exp), f"{where}: keys {sorted(got)} != {sorted(exp)}"
        for k in exp:
            assert_snapshot_equal(got[k], exp[k], f"{where}.{k}")
    elif isinstance(exp, list):
        assert len(got) == len(exp), f"{where}: length {len(got)} != {len(exp)}"
        for i, (g, e) in enumerate(zip(got, exp)):
            assert_snapshot_equal(g, e, f"{where}[{i}]")
    else:
        assert _same(got, exp), f"{where}: {got!r} != {exp!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("other", ["fitted_within", "fitted_all"])
def test_global_outputs_are_bit_identical_with_per_class(request, fitted_global, other):
    """D5: enabling per_class never changes a single bit of the global outputs."""
    s = request.getfixturevalue(other)
    for attr in ("lrc_", "lrc_unique_", "lrc_natural_", "zone_scores_", "predicates_df_"):
        pd.testing.assert_frame_equal(getattr(s, attr), getattr(fitted_global, attr),
                                      check_exact=True, obj=attr)
    assert s.valid_seeds_ == fitted_global.valid_seeds_
    for seed, g in fitted_global.graphs_by_seed_.items():
        h = s.graphs_by_seed_[seed]
        assert dict(g.nodes(data=True)) == dict(h.nodes(data=True))
        assert {(u, v): d["weight"] for u, v, d in g.edges(data=True)} == \
               {(u, v): d["weight"] for u, v, d in h.edges(data=True)}


def test_golden_global_output(request):
    snaps = {name: snapshot(_config(name)) for name in CONFIGS}
    # JSON round-trip so the comparison sees exactly what the file stores.
    snaps = json.loads(json.dumps(snaps))
    if request.config.getoption("--update-golden"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(snaps, indent=1, sort_keys=True), encoding="utf-8")
        pytest.skip(f"golden rewritten: {GOLDEN}")
    assert GOLDEN.exists(), "golden file missing; generate it with --update-golden"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert_snapshot_equal(snaps, expected, "golden")
