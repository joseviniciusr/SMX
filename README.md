<p align="center">
  <img src="SMX_logo.png" alt="SMX logo" width="360">
</p>

# SMX

This is the official repository for the `spectral-model-explainer` (SMX) library, an eXplainable AI tool designed to provide explanations for Machine Learning (ML) models trained on spectral data (*e.g.*, XRF, GRS, Raman, and related modalities).

SMX is a post-hoc, global, model-agnostic framework that explains spectral-based ML classifiers directly in terms of expert-informed spectral zones. It aggregates each zone via PCA, formulates quantile-based logical predicates, estimates their relevance through perturbation experiments within stochastic subsamples, and integrates the results into a directed weighted graph whose global structure is summarized by Local Reaching Centrality. A distinctive feature is threshold spectrum reconstruction, which back-projects each predicate's decision boundary into the original spectral domain in natural measurement units, enabling practitioners to visually compare their spectra against the model-related boundaries.

## Method Overview in the Library

The high-level workflow is implemented in the `SMX` pipeline class and can also be executed component-by-component through the public API:

1. spectral zone extraction
2. zone aggregation (typically PCA-based)
3. predicate generation from quantiles
4. bagging-based robustness evaluation
5. predicate relevance scoring
6. directed graph construction
7. centrality-based ranking and optional mapping back to natural scale

This implementation allows both:

- end-to-end execution through a single pipeline object
- advanced control through direct use of dedicated classes/functions

## Spectral Zone Construction

The method starts by partitioning the spectral axis into zones using `extract_spectral_zones`. Input spectra are expected as a DataFrame in which columns represent numeric spectral positions (energies, wavelengths, channels, etc.).

### How zones must be provided

The `cuts` argument accepts multiple valid formats:

- `(start, end)`
- `(name, start, end)`
- `(name, start, end, group)`
- `{name, start, end}`
- `{name, start, end, group}`

Important behavior:

- boundaries are interpreted numerically and inclusively
- if `start > end`, the library automatically reorders them
- grouped cuts (same `group`) are concatenated into one merged zone
- non-grouped cuts are kept as independent zones

This flexibility enables both physically meaningful elemental regions and composite regions such as aggregated background segments.

## Predicate Construction from Zone Scores

After extraction, each zone is transformed into one scalar score per sample (default strategy: PC1 score via `ZoneAggregator(method="pca")`). These zone-level summaries are the basis for predicate generation.

`PredicateGenerator` creates binary threshold predicates from a user-defined set of quantiles. For each zone and each quantile value `q`, two complementary predicates are produced:

- `zone <= threshold(q)`
- `zone > threshold(q)`

Therefore, if `k` quantiles are provided, the initial candidate set is `2k` predicates per zone (before duplicate removal). Duplicate rules are automatically removed when quantiles collapse to identical threshold values.

## Bagging and Robustness Hyperparameters

SMX estimates predicate robustness through repeated bagging cycles. In the high-level pipeline, this is controlled primarily by:

- `n_bags`: number of bags generated per repetition (seed)
- `n_repetitions`: number of independent repetitions (seed loop)
- `n_samples_fraction`: fraction of samples drawn in each bag
- `quantiles`: quantile grid that defines predicate thresholds

Operationally:

- each repetition creates a new random context for bag generation
- each bag evaluates which predicates are sufficiently supported by sampled data
- predicates with very low support in a bag are discarded for that bag
- final rankings are aggregated across valid repetitions to reduce seed sensitivity

This design makes the explanation less dependent on a single random split and more representative of stable decision behavior.

## Predicate Relevance and Graph Construction

Within each bag, predicates are ranked by an importance metric based on perturbation experiments:

- perturbation-based relevance (`PerturbationMetric`), using a fitted estimator

`PredicateGraphBuilder` then constructs a directed graph from ranked predicates:

- consecutive predicates in a ranking induce directed edges
- edge weights are accumulated across bags
- terminal class nodes are linked from last predicates in each path
- bidirectional conflicts are resolved by keeping the stronger direction (ties are randomized)
- edge weighting can incorporate zone-level explained variance from PCA (`var_exp=True`), which constrains the graph structure to reflect both predictive relevance and variance importance of zones

Finally, the graph is summarized through Local Reaching Centrality (LRC), producing a ranked list of influential predicates/zones. Accordngly, the final output is a DataFrame with predicates ranked by their LRC scores, along with their corresponding natural-scale thresholds and zone information. This allows practitioners to identify which spectral zones and thresholds are most influential in the model's decision-making process, providing insights into the underlying spectral features driving predictions. Beyond identifying relevant zones, the predicate's threshold values themselves live in PCA space and are back-projected to the original domain as per-zone multivariate thresholds that can be overlaid on measured spectra, translating an abstract condition into a physically readable boundary. Thus, SMX goes beyond numerical importances by delivering condition-aware, subset-aware explanations that support validation, hypothesis generation, and more actionable domain conclusions.

## Model Compatibility Note

At the current stage, SMX is primarily designed for use with scikit-learn-style estimators. In practical terms, this means that when the perturbation-based relevance strategy is employed, the estimator passed to the pipeline is expected to be already fitted and to expose the standard prediction interface required by the selected perturbation metric.

More specifically, the minimum requirement is a valid `predict` method. In addition, some perturbation metrics require richer interfaces: `probability_shift` requires `predict_proba`, while `decision_function_shift` requires `decision_function`. Consequently, any model class that follows this contract can be integrated in a technically consistent manner, independently of the specific learning algorithm (for example, SVMs, tree ensembles, linear models, and related scikit-learn-compatible estimators).

Ongoing development is focused on extending this compatibility layer beyond the current scikit-learn-centric workflow, with the objective of supporting additional model ecosystems and API styles in Python while preserving methodological consistency and interpretability guarantees.

## Installation and Optional Plotting Dependency

SMX is intentionally distributed with a lightweight core dependency set, where visualization is treated as an optional capability rather than a mandatory runtime requirement. This design ensures that users interested exclusively in methodological analysis (zone extraction, predicate construction, bagging, graph construction, and centrality-based ranking) can install and execute the framework without incurring additional graphical dependencies.

Base installation:

```bash
pip install spectral-model-explainer
```

Installation with plotting support:

```bash
pip install "spectral-model-explainer[plotting]"
```

In practical terms, the plotting extra enables functions that generate interactive visual outputs (for example, threshold-spectrum overlays used to inspect reconstructed multivariate decision boundaries in the natural spectral domain). The analytical SMX pipeline remains fully functional without this extra.

If plotting routines are invoked in an environment where the plotting extra has not been installed, SMX raises an explicit import-related error with installation guidance. This behavior is intentional: it preserves minimal installation overhead for non-visual workflows while providing clear and immediate feedback when visualization features are requested.

## Plotting Helpers

SMX includes Plotly-based visualization helpers for common explanation views.

### Zone Ranking Over Spectrum

After fitting an `SMX` explainer, you can export the ranked spectral zones over
the reference spectrum. The output format is inferred from the file extension —
use `.html` for an interactive figure or `.png` / `.svg` / `.pdf` for a static
image (requires `pip install kaleido`):

```python
from smx import SMX

explainer = SMX(
    spectral_cuts=spectral_cuts,
    quantiles=[0.2, 0.4, 0.6, 0.8],
    seeds=[0, 1, 2, 3],
    metric="perturbation",
    estimator=model,
)
explainer.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal_raw)

# Interactive HTML
explainer.plot_zone_ranking_over_spectrum("zone_ranking.html", ranking="unique")

# Static PNG (requires kaleido)
explainer.plot_zone_ranking_over_spectrum(
    "zone_ranking.png",
    ranking="unique",
    X_natural=X_cal_raw,
    y_labels=y_cal,
    width=1400,
    height=520,
)
```

You can also call the standalone plotting function:

```python
from smx import plot_zone_ranking_over_spectrum

plot_zone_ranking_over_spectrum(
    zone_ranking_df=explainer.lrc_summed_unique_,
    spectral_cuts=spectral_cuts,
    reference_spectrum=explainer.zones_natural_,
    output_path="zone_ranking.png",   # or .html
    class_spectra={"A": X_cal[y_cal == "A"], "B": X_cal[y_cal == "B"]},
)
```

![Zone ranking over spectrum](assets/zone_ranking_over_spectrum.png)

## Easy Usage

```python
import pandas as pd
from sklearn.svm import SVC
from smx import SMX

# X_cal_prep: preprocessed calibration spectra (DataFrame)
# X_cal_natural: original calibration spectra before preprocessing (DataFrame)
# y_cal_labels: class labels for calibration samples (Series)

spectral_cuts = [
("F1", 1.0, 100.0),
("background", 100.0, 200.0, "background_group"),
("F2", 200.0, 300.0),
]

model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_cal_prep, y_cal_labels)

# Example: probability of the first class as continuous output
y_pred_cal = 

smx = SMX(
spectral_cuts=spectral_cuts,
quantiles=[0.2, 0.4, 0.6, 0.8],
n_repetitions=4,
n_bags=10,
n_samples_fraction=0.8,
estimator=model,
perturbation_metric="probability_shift",
)

smx.fit(X_cal_prep, y_pred_cal, X_cal_natural=X_cal_natural)

# Main result (ranked predicates with natural-scale thresholds)
results = smx.lrc_natural_
print(results.head())
```

For a complete, executable walkthrough with synthetic data and visualization outputs, see the quickstart notebook:

[examples/quickstart.ipynb](examples/quickstart.ipynb)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
