#!/usr/bin/env python3
"""
Script to convert demo and frusta_animation notebooks to HTML for documentation.
Called from Sphinx build process.
"""

import os
import sys
from pathlib import Path


def convert_notebook_to_html(notebook_name):
    """Convert a specific notebook to HTML using nbconvert."""
    try:
        import nbformat
        from nbconvert import HTMLExporter

        # Paths
        docs_dir = Path(__file__).parent
        notebook_path = docs_dir.parent / "notebooks" / f"{notebook_name}.ipynb"
        html_path = docs_dir / "_static" / f"{notebook_name}.html"

        # Create _static directory if it doesn't exist
        html_path.parent.mkdir(exist_ok=True)

        # Convert notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = "classic"

        # Read notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = f.read()

        # Convert to HTML
        (body, resources) = html_exporter.from_notebook_node(
            nbformat.reads(notebook_content, as_version=4)
        )

        # Write HTML file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(body)

        print(f"Successfully converted {notebook_path} to {html_path}")
        return True

    except ImportError:
        print(
            "nbconvert or nbformat not available. Install with: pip install nbconvert nbformat"
        )
        return False
    except Exception as e:
        print(f"Error converting notebook {notebook_name}: {e}")
        return False


def convert_all_notebooks():
    """Convert all notebooks to HTML."""
    notebooks = ["demo", "frusta_animation"]
    success = True

    for notebook in notebooks:
        if not convert_notebook_to_html(notebook):
            success = False

    return success


if __name__ == "__main__":
    success = convert_all_notebooks()
    sys.exit(0 if success else 1)
