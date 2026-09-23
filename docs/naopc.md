# Normalized faithfulness (NAOPC)

The fork adds `evaluate_normalized_faithfulness` as a complementary metric.
The existing `evaluate_faithfulness` method and its output contract remain
unchanged.

```python
normalized = smx.evaluate_normalized_faithfulness(
    X_test_prep,
    ranking="unique",
    masking_strategy="zero",
    metric="auto",
    normalization="auto",  # "exact", "beam", or "auto"
    beam_size=5,
    exact_max_zones=9,
)

print(normalized["naopc"])
print(normalized["per_sample"].head())
```

For each evaluation sample, SMX keeps the originally predicted class fixed,
computes the AOPC over `k=1..m` cumulative zone masks, and normalizes it as:

```text
NAOPC = (AOPC - lower_bound) / (upper_bound - lower_bound)
```

`normalization="exact"` evaluates all `2**m` masked subsets and uses dynamic
programming to obtain the exact bounds. `normalization="beam"` uses separate
deterministic upper- and lower-bound beam searches for larger numbers of
zones. `"auto"` selects exact mode through `exact_max_zones`.

Exact mode therefore grows exponentially in the number of zones. Beam mode
reduces the search, but its per-sample searches can still request many unique
masked subsets; `n_model_queries` and `runtime_seconds` are returned so that
benchmark results can report this cost explicitly.

The result includes aggregate values (`aopc`, `naopc`, `lower_bound`, and
`upper_bound`), an auditable `per_sample` DataFrame, the observed `curve_df`,
the normalization method, model query count, and runtime. If the attainable
bounds are degenerate for a sample, its `naopc` is `NaN` and its status is
`"degenerate_bounds"`.

This implementation follows the MoRF AOPC convention used in Edin et al.,
excluding the unmasked `k=0` baseline from the area average. It is an
experimental fork feature and does not replace the legacy metric.
