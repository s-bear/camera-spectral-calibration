# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os,sys
sys.path.insert(0, os.path.abspath('../scripts'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Camera Spectral Calibration'
copyright = '2022, Samuel Powell'
author = 'Samuel Powell'
release = '2022-12-05'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', #for markdown parsing
              'sphinx_copybutton', #add "copy" buttons to code listings
              'sphinx.ext.githubpages', #make output compatible with github pages
              'sphinx.ext.mathjax', #math support
              'sphinx.ext.autodoc', #retrieve docstrings from python source
              'sphinx.ext.linkcode', #link to github for code
              'sphinx.ext.napoleon', #support for numpy/google docstring format
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

numfig = True #number figures

default_role = 'any'

autodoc_member_order = 'bysource'
autodoc_type_aliases = {'NDArray': 'numpy.ndarray', 'ArrayLike': 'array_like'}
autodoc_mock_imports = ['numpy','numba','scipy','matplotlib','h5py','sklearn','imageio','piexif','xlsxwriter']

source_repository = 'https://github.com/s-bear/camera-spectral-calibration'
source_directory = 'scripts'

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if 'module' not in info:
        return None
    fname = info['module'].replace('.','/')
    return f'{source_repository}/blob/main/{source_directory}/{fname}.py'

# for a fancy linkcode_resolve see https://github.com/aaugustin/websockets/blob/778a1ca6936ac67e7a3fe1bbe585db2eafeaa515/docs/conf.py#L100-L134



# -- Myst options ------------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = ['colon_fence','deflist']
myst_title_to_header=True
myst_update_mathjax=False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Camera Spectral Calibration'

html_theme_options = {
    'source_repository': source_repository,
    'source_branch': 'main',
    'source_directory': 'docs/',
}

mathjax3_config = {
    'options': {'processHtmlClass': 'tex2jax_process|mathjax_process|math|output_area'},
    'tex': {
        'inlineMath': [['$','$'],['\\(','\\)']],
        'macros': {
            'RR': '\\mathbb{R}', #all real number symbol
            'mat': ['\\mathbf{#1}',1], #matrix formatting (bold, upright)
            'trans': '\\intercal', #transpose T
            'units': ['\\ \\mathrm{\\left[#1\\right]}',1], #units formatting (space, upright text in brackets)
        },
    }
}