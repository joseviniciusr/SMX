# NAOPC validation log

This log records the first validation pass of the `research/naopc` fork.
Results were collected on 2026-09-23 from commit `556a05e` plus the local
NAOPC changes.

## Environment

- Python 3.10.12
- NumPy 1.26.4
- pandas 2.3.3 in the project virtual environment
- scikit-learn 1.7.2
- Plotly 6.6.0

## Baseline quickstart

The existing quickstart completed its numerical pipeline with the plotting
calls disabled. Static PNG export is not available in this environment because
the optional `kaleido` package is not installed.

| Metric | Baseline |
|---|---:|
| SVM test accuracy | 98.90% |
| Legacy AUC | 4.789539 |
| Legacy level | Very High |
| Legacy null percentile | 100.0 |

The same controlled legacy evaluation produced identical output values in the
original repository and in this fork.

## NAOPC quickstart evaluation

Using the same fitted SVM, split, zones, and `X_test_prep`:

| Metric | Result |
|---|---:|
| AOPC | 0.424274 |
| NAOPC | 0.797412 |
| Mean lower bound | 0.025221 |
| Mean upper bound | 0.540052 |
| Zones | 12 |
| Normalization | beam, `beam_size=5` |
| Model queries | 3,187 |
| Runtime | 13.04 s |
| Status | ok |

The automatic mode selected beam normalization because the quickstart has more
than the default nine exact zones.

## Tests and beam convergence

The fork test suite currently passes **9 tests**. It covers exact bounds,
brute-force equivalence, beam equivalence when the beam is complete,
multiclass and binary `decision_function`, degenerate bounds, the pipeline
facade, and legacy API availability.

The detailed synthetic convergence table is available at
[`experiments/naopc/beam_convergence.csv`](../experiments/naopc/beam_convergence.csv).
In the tested interaction cases with up to nine zones, `beam_size=5` had a
maximum absolute NAOPC error of approximately 0.0037; `beam_size=10` matched
the exact result for the nine-zone case. This is evidence for continued
benchmarking, not yet a universal default recommendation.

## Remaining validation

- Run the same comparison on the existing SMX real datasets and model types.
- Repeat with every masking strategy already supported by SMX.
- Measure beam stability for 10, 12, 16, 20, 30, and 50 zones.
- Decide whether aggregate bounds should remain means or be exposed only in
  the per-sample table.

## First real-dataset benchmark

The existing `SMX_research` loader, preprocessing, split, and model wrappers
were reused. The full SMX configuration used four repetitions, ten bags, and
100 legacy null rankings.

| Dataset | Model | Mask | Test accuracy | Legacy AUC | Legacy percentile | NAOPC | NAOPC time |
|---|---|---|---:|---:|---:|---:|---:|
| `bank_notes` | SVM | zero | 1.000 | 4.910883 | 100 | 0.709364 | 55.2 s |
| `milk` | SVM | zero | 0.843 | 3.877492 | 99 | 0.632387 | 21.5 s |
| `milk` | SVM | median | 0.843 | 3.494973 | 98 | 0.636164 | 21.8 s |
| `milk` | PLS | zero | 0.765 | 1.917531 | 90 | 0.526302 | 22.0 s |

The complete summaries and per-sample tables are under
[`experiments/naopc/real_results/`](../experiments/naopc/real_results/). The
initial results suggest that legacy percentile and NAOPC are related but not
interchangeable, and that the median masking conclusion for `milk`/SVM is
close to the zero-masking conclusion. These are exploratory observations,
not yet general claims across the full dataset collection.

The real benchmark also exposed and fixed a boundary-overlap bug: when two
adjacent spectral cuts include the same boundary column, NAOPC now deduplicates
that column before assignment, matching the legacy protocol.
