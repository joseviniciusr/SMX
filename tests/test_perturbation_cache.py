import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC

from smx.datasets.synthetic import generate_synthetic_spectral_data
from smx.pipeline import SMX
from smx.predicates.bagging import PredicateBagger
from smx.predicates.generation import PredicateGenerator
from smx.predicates.metrics import PerturbationMetric
from smx.zones.aggregation import ZoneAggregator
from smx.zones.extraction import extract_spectral_zones

CUTS = [("z1", 0, 200), ("z2", 200, 400), ("z3", 400, 600), ("z4", 600, 800), ("z5", 800, 1000)]


@pytest.fixture(scope="module")
def data():
    df = generate_synthetic_spectral_data(
        [
            {"name": "A", "n_samples": 40, "peaks": [250, 550]},
            {"name": "B", "n_samples": 40, "peaks": [250, 700]},
        ],
        n_points=200,
        seed=0,
    )
    X = df.drop(columns="Class")
    X = X - X.mean()
    y = (df["Class"] == "A").astype(int)
    return X, y


def _metric_inputs(X, y_cont):
    zone_scores = ZoneAggregator(method="pca").fit_transform(extract_spectral_zones(X, CUTS))
    gen = PredicateGenerator(quantiles=[0.2, 0.4, 0.6, 0.8])
    gen.fit(zone_scores)
    bags = PredicateBagger(n_bags=5, n_samples_fraction=0.8, random_seed=0).run(
        zone_scores, pd.Series(y_cont), gen.predicates_df_
    )
    return gen.predicates_df_, bags


@pytest.mark.parametrize("mode,source", [("median", "full"), ("constant", "full"), ("mean", "predicate")])
def test_cached_metric_matches_uncached(data, mode, source):
    X, y = data
    svc = SVC(probability=True, random_state=0).fit(X, y)
    predicates_df, bags = _metric_inputs(X, svc.predict_proba(X)[:, 1])
    kw = dict(
        estimator=svc, Xcalclass_prep=X, predicates_df=predicates_df, spectral_cuts=CUTS,
        perturbation_mode=mode, stats_source=source, metric="probability_shift",
        normalize_by_zone_size=True,
    )
    cached = PerturbationMetric(**kw).compute(bags)
    plain = PerturbationMetric(**kw, cache_predictions=False).compute(bags)
    for bag in bags:
        # SVC scores rows independently, so the cached outputs are bit-identical
        pd.testing.assert_frame_equal(cached[bag], plain[bag], check_exact=True)


def test_cache_evaluates_model_once_per_zone(data):
    X, y = data
    svc = SVC(probability=True, random_state=0).fit(X, y)
    predicates_df, bags = _metric_inputs(X, svc.predict_proba(X)[:, 1])
    calls = []

    class Counting:
        classes_ = svc.classes_

        def predict_proba(self, Z):
            calls.append(len(Z))
            return svc.predict_proba(Z)

    metric = PerturbationMetric(
        estimator=Counting(), Xcalclass_prep=X, predicates_df=predicates_df, spectral_cuts=CUTS,
        perturbation_mode="median", metric="probability_shift",
    )
    metric.compute(bags)
    metric.compute(bags)
    assert len(calls) == predicates_df["zone"].nunique() + 1
    assert set(calls) == {len(X)}


@pytest.mark.parametrize("model", ["svc", "pls"])
def test_smx_n_jobs_does_not_change_result(data, model):
    X, y = data
    if model == "svc":
        est = SVC(probability=True, random_state=0).fit(X, y)
        y_cont, metric = est.predict_proba(X)[:, 0], "probability_shift"
    else:
        est = PLSRegression(n_components=2).fit(X, y)
        y_cont, metric = est.predict(X).ravel(), "mean_abs_diff"
    kw = dict(spectral_cuts=CUTS, quantiles=[0.2, 0.4, 0.6, 0.8], n_repetitions=3, n_bags=4,
              estimator=est, perturbation_metric=metric)
    serial = SMX(**kw).fit(X, y_cont)
    parallel = SMX(**kw, n_jobs=2).fit(X, y_cont)
    pd.testing.assert_frame_equal(serial.lrc_summed_, parallel.lrc_summed_, check_exact=True)
