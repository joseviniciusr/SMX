# SMX Documentation

This directory contains the Sphinx documentation for SMX.

## Build and serve locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install ".[docs]"
```

This installs:
- `sphinx` - documentation generator
- `pydata-sphinx-theme` - theme
- `sphinx-autoapi` - API docs from docstrings
- `myst-parser` - Markdown support
- `sphinx-copybutton` - copy button for code blocks
- `sphinx-design` - grid cards, tabs, badges

### Build the documentation

```bash
sphinx-build -b html docs/ docs/_build/html
```

### Serve locally

```bash
cd docs/_build/html
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Documentation structure

- `index.md` - landing page
- `quickstart.md` - minimal usage walkthrough
- `pipeline.md` - pipeline overview
- `plotting.md` - visualization guide
- `api_reference.md` - API docs entry point
- `conf.py` - Sphinx configuration
- `_static/` - custom CSS and images
- `_build/` - generated output (git ignored)

## Publishing

Documentation builds on every push to `main` via `.github/workflows/release.yml`.
If the project is configured on Read the Docs, the build is driven by
`.readthedocs.yaml` and the same `docs/` source tree.
