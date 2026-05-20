# Faithfulness evaluation

SMX includes a progressive masking protocol to evaluate faithfulness. Ranked
zones are masked in order, and the model prediction shift is summarized by the
area under the curve (AUC).

## Evaluate with the pipeline

```python
faithfulness = smx.evaluate_faithfulness(
    X_eval=X_test_prep,
    ranking="unique",
    masking_strategy="zero",
    metric="auto",  # automatically selects "probability_shift", "decision_function_shift", or "mean_abs_diff" based on the estimator's available methods
    output_path="faithfulness_curve.html",
)

print(faithfulness["auc"], faithfulness["level"])  # AUC and qualitative level
```

## Standalone function

```python
from smx.evaluation import progressive_masking_faithfulness

result = progressive_masking_faithfulness(
    estimator=model,
    X_eval=X_test_prep,
    spectral_cuts=spectral_cuts,
    ranking_df=smx.lrc_summed_unique_,
    masking_strategy="mean",
)
```

## Plot saved results

```python
smx.plot_faithfulness("faithfulness_curve.html")
```
