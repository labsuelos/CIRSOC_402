# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.join(os.path.abspath('..'),'cirsoc_402')) 
sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'CIRSOC 402'
copyright = '2021, A. Sfriso, P. Fernandez, I. Cueto, M. Biedma, J. Manduca, P. Barbieri'
author = 'A. Sfriso, P. Fernandez, I. Cueto, M. Biedma, J. Manduca, P. Barbieri'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', # sphinx loads docstrings
              'sphinx.ext.napoleon', # shpinx understand numpy docstring
              'nbsphinx', # use jupyter notebooks for documentation
              "sphinx.ext.intersphinx", # package requested by the theme
              "sphinx.ext.mathjax", # package requested by the theme to parse mathematical formulas
              "sphinx.ext.viewcode", # package requested by the theme
              "sphinx_copybutton", # copy to cliploard button
              "sphinxcontrib.bibtex" #bibtex bibliography
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = 'CIRSOC 402'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# enable nbsphinx to  import equation numbering
mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# bibliographycal reference file
bibtex_bibfiles = ['refs.bib']