# SMX

```{raw} html
<div class="smx-hero">
  <div class="smx-hero-text">
    <p class="smx-kicker">Spectral Model eXplainer</p>
    <h1>Explain spectral ML models with zone-level predicates</h1>
    <p>
      SMX is a global, model-agnostic explainer for spectral classifiers.
      It turns spectra into expert zones, generates logical predicates, and
      ranks them with a graph-based centrality score.
    </p>
    <div class="smx-hero-actions">
      <a class="smx-button" href="quickstart.html">Get started</a>
      <a class="smx-button smx-button-outline" href="plotting.html">Plotting gallery</a>
    </div>
  </div>
  <div class="smx-hero-logo">
    <img src="_static/SMX_logo.png" alt="SMX logo" />
  </div>
</div>
```

## What SMX provides

- End-to-end pipeline via the `SMX` class
- Zone extraction and PCA aggregation tailored to spectral inputs
- Quantile predicates with bagging and perturbation-based ranking
- Directed predicate graph with Local Reaching Centrality (LRC)
- Natural-scale threshold reconstruction and Plotly visuals
- Faithfulness evaluation with progressive masking

::::{grid} 3
:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Run SMX end to end on a synthetic dataset in minutes.
:::
:::{grid-item-card} Pipeline
:link: pipeline
:link-type: doc

Understand how zones, predicates, and graphs fit together.
:::
:::{grid-item-card} Plotting Gallery
:link: plotting
:link-type: doc

Explore the full suite of SMX visualization helpers.
:::
:::{grid-item-card} API Reference
:link: api_reference
:link-type: doc

Auto-generated reference for every public module.
:::
:::{grid-item-card} Examples
:link: examples
:link-type: doc

Scripts and notebooks shipped with the repository.
:::
:::{grid-item-card} Repository Map
:link: repository
:link-type: doc

A guided tour of the SMX source tree.
:::
::::

## Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
pipeline
zones
predicates
graph
faithfulness
plotting
datasets
examples
repository
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api_reference
```
