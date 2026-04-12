# SMX

![SMX Logo](SMX_logo.png)

SMX (Spectral Model eXplainer) is a Python library for generating interpretable, zone-level explanations of machine learning models trained on spectral data (*e.g.*, XRF, GRS, Raman, and related modalities). The method is designed to bridge predictive performance and explainability by converting high-dimensional spectral variables into hierarchically ordered predicates, then organizing them into a directed weighted graph.

Rather than treating model explanability as a post-hoc global ranking of individual variables, SMX explicitly models the spectral axis through user-defined zones, derives threshold-based predicates from zone representations, quantifies their predictive relevance under repeated resampling, and synthesizes the final explanatory structure through graph centrality.

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
- `replace`: whether sampling is with replacement
- `quantiles`: quantile grid that defines predicate thresholds

Operationally:

- each repetition creates a new random context for bag generation
- each bag evaluates which predicates are sufficiently supported by sampled data
- predicates with very low support in a bag are discarded for that bag
- final rankings are aggregated across valid repetitions to reduce seed sensitivity

This design makes the explanation less dependent on a single random split and more representative of stable decision behavior.

## Predicate Relevance and Graph Construction

Within each bag, predicates are ranked by an importance metric. The library supports at least two major strategies:

- covariance-based relevance (`CovarianceMetric`)
- perturbation-based relevance (`PerturbationMetric`), using a fitted estimator

`PredicateGraphBuilder` then constructs a directed graph from ranked predicates:

- consecutive predicates in a ranking induce directed edges
- edge weights are accumulated across bags
- terminal class nodes are linked from last predicates in each path
- bidirectional conflicts are resolved by keeping the stronger direction (ties are randomized)

Optionally, edge weighting can incorporate zone-level explained variance from PCA (`var_exp=True`).

Finally, the graph is summarized through Local Reaching Centrality (LRC), producing a ranked list of influential predicates/zones. The pipeline can also map thresholds from preprocessed score space back to natural spectral scale for physically meaningful interpretation.

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
y_pred_cal = model.predict_proba(X_cal_prep)[:, 0]

smx = SMX(
	spectral_cuts=spectral_cuts,
	quantiles=[0.25, 0.50, 0.75],
	n_repetitions=4,
	n_bags=10,
	n_samples_fraction=0.8,
	replace=False,
	metric="perturbation",
	estimator=model,
	perturbation_mode="median",
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
