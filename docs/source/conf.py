# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))
load_dotenv()

project = 'open-mimic-iii'
copyright = '2024, Amadou Wolfgang Cisse'
author = 'Amadou Wolfgang Cisse'
release = '-'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []
default_role = "autolink"
exclude_patterns = ["../src/handlers.py"]

html_theme_options = {
    'navigation_depth': -1,
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autosummary_generate = True
napoleon_numpy_docstring = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
