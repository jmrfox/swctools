@echo off
echo Building documentation with notebook conversion...
uv run sphinx-build docs docs/_build %*
echo Documentation built successfully!
echo Open docs/_build/index.html to view the documentation.
