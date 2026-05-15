# Repository map

SMX is intentionally small and readable. The main directories are:

- `smx/` - library source code
- `examples/` - runnable scripts and notebooks
- `assets/` - images used in the gallery and README
- `.github/workflows/` - CI workflows (release and docs)
- `pyproject.toml` - project metadata and dependencies

## Package layout

`smx/` is organized by responsibility:

- `pipeline.py` - high-level `SMX` class
- `zones/` - zone extraction and aggregation
- `predicates/` - predicate generation, bagging, and metrics
- `graph/` - graph construction and LRC ranking
- `evaluation/` - faithfulness evaluation
- `plotting/` - Plotly visualization helpers
- `datasets/` - synthetic data generation

If you want to contribute a new module, the best starting point is the
`SMX` pipeline in `smx/pipeline.py` and the public API exports in
`smx/__init__.py`.
