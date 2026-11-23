"""Scientific checks on scenarios whose answer is known by construction.

S1 — each class has its own marker zone (Za -> A, Zb -> B, Zc -> C) plus a
     noise zone: the 'within' ranking of every class must put the class's own
     marker first (robust in 8/8 probed datasets).
S2 — class B is defined by exclusion (neither the A nor the C marker): the
     'all' ranking of B must put Za and Zc on top (8/8 probed datasets).
"""

from __future__ import annotations

import pytest

from helpers import LinearSoftmaxStub, fit_smx, make_exclusion_truth, make_known_truth

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_s1_within_puts_each_class_marker_first(seed):
    data = make_known_truth(seed=seed)
    est = LinearSoftmaxStub(n_ch=data["n_ch"], n_zones=data["n_zones"])
    s = fit_smx(data, est, per_class="within", natural=False)
    for k, marker in zip("ABC", ["Za", "Zb", "Zc"]):
        ranking = s.lrc_unique_per_class_[k]
        assert ranking["Zone"].iloc[0] == marker
        # The own marker dominates: well above the runner-up.
        lrc = ranking["Local_Reaching_Centrality"].values
        assert lrc[0] > 3 * lrc[1]


@pytest.mark.parametrize("seed", [0, 3])
def test_s2_all_ranks_the_exclusion_zones_first_for_the_excluded_class(seed):
    data = make_exclusion_truth(seed=seed)
    s = fit_smx(data, data["est"], per_class="all", natural=False)
    assert set(s.lrc_unique_per_class_["B"]["Zone"].iloc[:2]) == {"Za", "Zc"}
