"""PredicateBagger: membership, 20 % guard, determinism, duplicates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smx.predicates.bagging import PredicateBagger
from smx.predicates.generation import PredicateGenerator
from smx.zones.aggregation import ZoneAggregator
from smx.zones.extraction import extract_spectral_zones
from helpers import make_known_truth, quiet


@pytest.fixture(scope="module")
def scores_and_predicates():
    data = make_known_truth(seed=7)
    zones = extract_spectral_zones(data["X"], data["cuts"])
    scores = ZoneAggregator(method="pca").fit_transform(zones)
    gen = PredicateGenerator(quantiles=[0.25, 0.5, 0.75]).fit(scores)
    return scores, gen.predicates_df_


def _run(scores, predicates, **kwargs):
    params = dict(random_seed=3, n_bags=4, n_samples_fraction=0.8)
    params.update(kwargs)
    with quiet():
        return PredicateBagger(**params).run(scores, predicates)


def _rule_mask(scores, row):
    vals = scores[row["zone"]].values
    t = float(row["thresholds"])
    return vals <= t if row["operator"] == "<=" else vals > t


def test_every_bag_sample_satisfies_its_predicate_and_nothing_is_missed(scores_and_predicates):
    scores, predicates = scores_and_predicates
    np.random.seed(3)  # replicate the bagger's draws to know each bag's sample set
    n = len(scores)
    draws = [np.random.choice(range(n), size=int(n * 0.8), replace=False) for _ in range(4)]
    bags = _run(scores, predicates)
    rows = predicates.set_index("rule")
    assert bags
    for bag_name, preds in bags.items():
        drawn = draws[int(bag_name.split("_")[1]) - 1]
        for rule, df in preds.items():
            mask = _rule_mask(scores, rows.loc[rule])
            expected = drawn[mask[drawn]]
            np.testing.assert_array_equal(df["Sample_Index"].values, expected)
            np.testing.assert_array_equal(df["Zone_Sum"].values, scores[rows.loc[rule, "zone"]].values[expected])


def test_guard_discards_predicates_below_20_percent_of_the_dataset(scores_and_predicates):
    scores, predicates = scores_and_predicates
    guard = max(1, int(len(scores) * 0.2))
    bags = _run(scores, predicates, n_bags=6)
    kept = [len(df) for preds in bags.values() for df in preds.values()]
    assert kept and min(kept) >= guard
    # Some predicates (e.g. the 0.25 quantile with '<=') sit near the guard and
    # are discarded in at least one bag: the guard is actually exercised.
    n_rules = len(predicates)
    assert any(len(preds) < n_rules for preds in bags.values())


def test_same_seed_same_bags_different_seed_different_bags(scores_and_predicates):
    scores, predicates = scores_and_predicates
    a, b = _run(scores, predicates), _run(scores, predicates)
    c = _run(scores, predicates, random_seed=4)
    assert a.keys() == b.keys()
    for bag in a:
        assert a[bag].keys() == b[bag].keys()
        for rule in a[bag]:
            pd.testing.assert_frame_equal(a[bag][rule], b[bag][rule])
    first = next(iter(a["Bag_1"]))
    assert not np.array_equal(a["Bag_1"][first]["Sample_Index"].values,
                              c["Bag_1"][first]["Sample_Index"].values)


def test_replace_true_keeps_duplicates_and_guard_counts_them(scores_and_predicates):
    """N7: with replace=True duplicates are kept and counted by the 20 % guard."""
    scores, predicates = scores_and_predicates
    bags = _run(scores, predicates, replace=True, n_bags=6)
    guard = max(1, int(len(scores) * 0.2))
    dfs = [df for preds in bags.values() for df in preds.values()]
    # Duplicated rows are kept (bootstrap) and the guard is applied to rows.
    assert any(df["Sample_Index"].duplicated().any() for df in dfs)
    assert all(len(df) >= guard for df in dfs)


def test_bag_frames_have_exactly_the_documented_columns(scores_and_predicates):
    scores, predicates = scores_and_predicates
    bags = _run(scores, predicates, n_bags=1)
    for df in bags["Bag_1"].values():
        assert list(df.columns) == ["Zone_Sum", "Sample_Index"]
