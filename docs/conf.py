# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'EasyDel'
copyright = '2023, The EasyDel. NumPy, JAX and SciPy documentation are copyright the respective authors.'
author = 'Erfan Zare Chavoshi'

version = ''
release = ''
needs_sphinx = '2.1'

sys.path.append(os.path.abspath('sphinxext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
    'myst_nb',
    "sphinx_remove_toctrees",
    'sphinx_copybutton',
    'jax_extensions',
    'sphinx_design'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

suppress_warnings = [
    'ref.citation',
    'ref.footnote',
    'myst.header',
    'misc.highlighting_failure'
]

templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

main_doc = 'index'
language = 'en'

exclude_patterns = [
    '*.md'
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

autosummary_generate = True
napolean_use_rtype = False

html_theme = 'sphinx_book_theme'

html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/erfanzar/EasyDel',
    'use_repository_button': True,
    'navigation_with_keys': False,
}

html_logo = 'light-logo.png'
html_static_path = ['_static']

html_css_files = [
    'style.css',
]

myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ['dollarmath']
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True

nb_execution_timeout = 100

htmlhelp_basename = 'FJFormerdoc'

latex_elements = {
}

man_pages = [
    (main_doc, 'EasyDel', 'EasyDel Documentation',
     [author], 1)
]

texinfo_documents = [
    (main_doc, 'EasyDel', 'EasyDel Documentation',
     author, 'EasyDel', 'One line description of project.',
     'Miscellaneous'),
]
epub_title = project
epub_exclude_files = ['search.html']

always_document_param_types = True

remove_from_toctrees = ["_autosummary/*"]
