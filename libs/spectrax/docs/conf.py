"""Sphinx configuration for the SpectraX documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SpectraX"
copyright = "2026, Erfan Zare Chavoshi - SpectraX"
author = "Erfan Zare Chavoshi"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../CHANGELOG.md"]

intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]

source_suffix = [".rst", ".md", ".ipynb"]
autosummary_generate = True
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
