# Normalized faithfulness (NAOPC)

SMX exposes NAOPC through evaluate_normalized_faithfulness. The existing evaluate_faithfulness method and its output contract remain unchanged.

```python
normalized = smx.evaluate_normalized_faithfulness(
    X_test_prep,
    ranking="unique",
    masking_strategy="zero",
    metric="calibration_invariant",
    normalization="auto",  # "exact", "beam", or "auto"
    beam_size=5,
    exact_max_zones=9,
)

print(normalized["naopc"], normalized["level"], normalized["null_percentile"])
print(normalized["per_sample"].head())
```

The evaluate_normalized_faithfulness method returns the NAOPC score dictionary. The main score fields are:

- aopc: observed area over the perturbation curve.
- lower_bound and upper_bound: attainable AOPC limits for the same model, sample, masking strategy, and zone set.
- naopc: (aopc - lower_bound) / (upper_bound - lower_bound), aggregated over non-degenerate samples.
- null_naopc_distribution and null_percentile: normalized scores for random zone rankings and the percentile of the supplied ranking.
- level: categorical label derived from null_percentile.
- per_sample and curve_df: auditable per-sample and progressive-curve values.


For each evaluation sample, SMX keeps the originally predicted class fixed,
computes the AOPC over `k=1..m` cumulative zone masks, and normalizes it as:

```text
NAOPC = (AOPC - lower_bound) / (upper_bound - lower_bound)
```

For calibration-independent comparisons, use `metric="calibration_invariant"`.
This requires `decision_function()` and evaluates signed target-class margin
shifts. Positive affine rescaling of the margin cancels between the observed
AOPC and its attainable bounds, unlike probability-shift evaluation, which can
change under confidence calibration.

The result also includes `null_percentile` and `level`. These are computed from
random zone rankings using the same NAOPC normalization and use the existing
SMX thresholds: `Low` (<60), `Moderate` (60–79.9), `High` (80–94.9), and
`Very High` (≥95).

`normalization="exact"` evaluates all `2**m` masked subsets and uses dynamic
programming to obtain the exact bounds. `normalization="beam"` uses separate
deterministic upper- and lower-bound beam searches for larger numbers of
zones. `"auto"` selects exact mode through `exact_max_zones`.

Exact mode therefore grows exponentially in the number of zones. Beam mode
reduces the search, but its per-sample searches can still request many unique
masked subsets; `n_model_queries` and `runtime_seconds` are returned so that
benchmark results can report this cost explicitly.

The result includes aggregate values (`aopc`, `naopc`, `lower_bound`, and
`upper_bound`), plus `level`, `null_percentile`, and
`null_naopc_distribution`, an auditable `per_sample` DataFrame, the observed `curve_df`,
the normalization method, model query count, and runtime. If the attainable
bounds are degenerate for a sample, its `naopc` is `NaN` and its status is
`"degenerate_bounds"`.

This implementation follows the MoRF AOPC convention used in Edin et al.,
excluding the unmasked `k=0` baseline from the area average. It is an additive evaluator and does not replace the legacy metric.

## Reference

The normalization follows Edin et al., who introduced NAOPC to make AOPC
comparable across models and inputs by dividing the observed AOPC by the
attainable model- and input-specific lower and upper limits. This implementation
follows the MoRF convention and excludes the unmasked step from the area average.
See [Edin et al., ACL 2025](https://aclanthology.org/2025.acl-long.86/)
(DOI: [10.18653/v1/2025.acl-long.86](https://doi.org/10.18653/v1/2025.acl-long.86)).

    @inproceedings{edin-etal-2025-normalized,
      title = {Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attributions Explainability},
      author = {Edin, Joakim and Motzfeldt, Andreas Geert and Christensen, Casper L. and Ruotsalo, Tuukka and Maaloe, Lars and Maistro, Maria},
      booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      pages = {1715--1730},
      year = {2025},
      publisher = {Association for Computational Linguistics},
      doi = {10.18653/v1/2025.acl-long.86}
    }

## Validation in this fork

The comparison used 39 valid real-dataset/model pairs (PLS, MLP, and SVM),
666 leave-one-zone-out retraining ablations, and 500 random rankings per pair.
The external oracle used the increase in held-out log-loss after retraining as
its primary impact measure, with accuracy-drop results reported as a secondary
check.

| Label agreement with the external oracle | Exact agreement | Mean ordinal error |
|---|---:|---:|
| Original SMX faithfulness | 12/39 (30.8%) | 1.38 levels |
| NAOPC | 16/39 (41.0%) | 1.23 levels |

NAOPC was correct in four cases where the original method was not; there were
no cases where only the original method matched the oracle. This supports NAOPC
as the preferred score for cross-model comparisons, but it does not justify
calling every NAOPC label definitive: both methods still assigned Very High to
more cases than the external oracle. The aggregate oracle results above are retained as the validation summary; the raw experiment runner and generated outputs are intentionally not part of this API merge request.

In the controlled calibration benchmark, the maximum change in NAOPC under
positive output-score rescaling was 2.3e-16. The legacy raw AOPC changed with
the output scale, confirming why raw AOPC should not be used as the primary
metric for comparisons across differently calibrated models.

The original faithfulness method remains available for backwards compatibility.
The NAOPC result should be interpreted together with the masking strategy,
normalization mode, null percentile, and query cost.
