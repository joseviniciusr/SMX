# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

# If the package is not installed, point Sphinx at the source tree so autoapi
# can discover the modules without needing an editable install.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "SMX"
author = "Jose Vinicius Ribeiro, Rafael Figueira Goncalves, Sylvio Barbon Junior"
copyright = "2026, Jose Vinicius Ribeiro, Rafael Figueira Goncalves, Sylvio Barbon Junior"

try:
    release = package_version("spectral-model-explainer")
except PackageNotFoundError:
    release = "0.0.0"

version = release.split("+", 1)[0]
if "." in version:
    version = ".".join(version.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    # Auto-generate API reference pages from docstrings.
    "autoapi.extension",
    # Google-style docstring support.
    "sphinx.ext.napoleon",
    # Cross-link to NumPy, pandas, scikit-learn, etc.
    "sphinx.ext.intersphinx",
    # "View Source" links on generated pages.
    "sphinx.ext.viewcode",
    # Render Markdown files via MyST.
    "myst_parser",
    # Copy button on code blocks.
    "sphinx_copybutton",
    # Grid cards, tabs, badges.
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress autoapi warnings that can arise from static analysis.
suppress_warnings = ["autoapi"]

# ---------------------------------------------------------------------------
# sphinx-autoapi
# ---------------------------------------------------------------------------
autoapi_dirs = ["../smx"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = False
# Put the auto-generated pages under /api/
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = False

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx - cross-link to external project docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "plotly": ("https://plotly.com/python/", None),
}

# ---------------------------------------------------------------------------
# MyST parser options
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# ---------------------------------------------------------------------------
# HTML output - pydata-sphinx-theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/joseviniciusr/SMX",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_nav_level": 3,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/spectral-model-explainer/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "joseviniciusr",
    "github_repo": "SMX",
    "github_version": "main",
    "doc_path": "docs",
}

html_logo = "_static/SMX_logo.png"
html_title = "SMX"
html_short_title = "SMX"
html_css_files = ["custom.css"]

# Remove the left sidebar from the landing page.
html_sidebars = {
    "index": [],
}
