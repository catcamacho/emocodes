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
sys.path.insert(0, os.path.abspath('../emocodes'))


# -- Project information -----------------------------------------------------

project = 'emocodes'
copyright = '2021, M. Catalina Camacho'
author = 'M. Catalina Camacho; Elizabeth M. Williams'

# The full version, including alpha/beta/rc tags
release = '1.0.14'
autoclass_content = 'class'
autosummary_generate = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones. , 'sphinx_toolbox.confval'
extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'autoapi.extension',
              'nbsphinx']

autoapi_type = 'python'
autoapi_dirs = [os.path.abspath('../emocodes')]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'testing*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# Set link name generated in the top bar.
html_title = 'EmoCodes'
github_url = 'https://github.com/catcamacho/emocodes/'
html_static_path = ['.']
html_logo = os.path.abspath('../logos/circle_color.png')

# Material theme options (see theme.conf for more information)
html_theme_options = {
    'style_nav_header_background': '#343131',
    'navigation_depth': 4,
    'collapse_navigation': False,
    'logo_only': True
}
