# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "swctools"
copyright = "2025, Jordan M. R. Fox"
author = "Jordan M. R. Fox"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/jmrfox/swctools",
    "show_prev_next": False,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "show_toc_level": 2,
}

# -- Napoleon settings (NumPy-style docstrings) -----------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Autosummary -------------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "plotly": ("https://plotly.com/python-api-reference", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}


# -- Custom build hook for notebook conversion --------------------------------
def setup(app):
    """Convert notebook to HTML before building."""
    import subprocess
    import sys

    # Convert notebook to HTML
    try:
        script_path = os.path.join(os.path.dirname(__file__), "convert_notebook.py")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode != 0:
            print(f"Warning: Notebook conversion failed: {result.stderr}")
        else:
            print(result.stdout)
    except Exception as e:
        print(f"Warning: Could not convert notebook: {e}")
