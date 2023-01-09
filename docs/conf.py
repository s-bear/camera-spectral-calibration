# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os,sys, importlib, inspect, warnings
import subprocess

from sphinx.ext.autodoc.mock import mock

sys.path.insert(0, os.path.abspath('../scripts'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Camera Spectral Calibration'
copyright = '2022, Samuel Powell'
author = 'Samuel Powell'
release = '2022-12-05'

source_repository = 'https://github.com/s-bear/camera-spectral-calibration'

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

def get_commit():
    # use git to get the commit hash
    cmd = 'git describe --dirty --tags --long --always'.split(' ')
    commit = subprocess.run(cmd,capture_output=True,check=True,text=True,timeout=1).stdout.strip()
    if 'dirty' in commit:
        warnings.warn('Generating docs on dirty repository -- source links will be broken!')
    return commit

commit = get_commit()

source_url = f'{source_repository}/blob/{commit}'

def linkcode_resolve(domain, info):
    #modified from https://github.com/aaugustin/websockets/blob/main/docs/conf.py

    assert domain == "py", "expected only Python objects"

    with mock(autodoc_mock_imports):
        mod = importlib.import_module(info["module"])

    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath("."))
    if not file.startswith("scripts"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{source_url}/{file}#L{start}-L{end}"



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