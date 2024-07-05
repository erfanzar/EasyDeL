import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "EasyDeL"
copyright = "2023, Erfan Zare Chavoshi - EasyDeL"
author = "Erfan Zare Chavoshi"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
}

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "style.css",
]

source_suffix = [".rst", ".md", ".ipynb"]
