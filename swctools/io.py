"""SWC file parsing utilities.

This module provides functions to parse SWC morphology files and extract:
- records for each SWC node (n, T, x, y, z, r, parent)
- header annotations for cycle break reconnections

It performs basic validations (unique ids, parent references) and can
optionally validate that requested reconnection node pairs share identical
(x, y, z, r) values.

Notes
-----
- The SWC format is documented by the NeuronLand spec and INCF.
- Header reconnection annotations follow lines like:
  "# CYCLE_BREAK reconnect i j"
  Parsing is case-insensitive for the tokens "CYCLE_BREAK" and "reconnect".
- Geometry/graph construction are handled elsewhere (e.g., SWCModel / GeneralModel).

Example
-------
>>> from swctools.io import parse_swc
>>> result = parse_swc("data/example.swc")
>>> len(result.records) > 0
True
>>> result.reconnections  # list of (i, j) tuples (may be empty)
[(16, 8)]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
import io
import os
import re


# Public data structures -------------------------------------------------------------------------


@dataclass(frozen=True)
class SWCRecord:
    """One SWC row.

    Attributes
    ----------
    n: int
        Node id (unique within file)
    t: int
        Structure type code
    x, y, z: float
        Coordinates (usually micrometers)
    r: float
        Radius
    parent: int
        Parent id; -1 indicates root
    line: int
        1-based line number in the source file/string
    """

    n: int
    t: int
    x: float
    y: float
    z: float
    r: float
    parent: int
    line: int


@dataclass(frozen=True)
class SWCParseResult:
    """Parsed SWC content."""

    records: Dict[int, SWCRecord]
    reconnections: List[Tuple[int, int]]
    comments: List[str]

    def __str__(self) -> str:
        return f"SWCParseResult(records={len(self.records)}, reconnections={len(self.reconnections)}, comments={len(self.comments)})"

    def __repr__(self) -> str:
        return str(self)


# Regex to capture reconnection directives in header comment lines
_RECONNECT_RE = re.compile(
    r"^\s*#\s*CYCLE_BREAK\s+reconnect\s+(?P<i>\d+)\s+(?P<j>\d+)\b",
    re.IGNORECASE,
)


# Public API --------------------------------------------------------------------------------------


def parse_swc(
    source: Union[str, os.PathLike, Iterable[str], io.TextIOBase],
    *,
    strict: bool = True,
    validate_reconnections: bool = True,
    float_tol: float = 1e-9,
) -> SWCParseResult:
    """Parse an SWC file or text stream.

    Parameters
    ----------
    source
        Path to an SWC file, a file-like object, an iterable of lines, or a string
        containing SWC content.
    strict
        If True, enforce 7-column rows and validate parent references exist.
    validate_reconnections
        If True, ensure reconnection node pairs share identical (x, y, z, r).
    float_tol
        Tolerance used when comparing floating-point coordinates/radii.

    Returns
    -------
    SWCParseResult
        Parsed records, reconnection pairs, and collected comments.

    Raises
    ------
    ValueError
        If parsing or validation fails.
    FileNotFoundError
        If a string path is provided that does not exist.
    """
    records: Dict[int, SWCRecord] = {}
    comments: List[str] = []
    reconnections: List[Tuple[int, int]] = []

    for lineno, raw in _iter_lines(source):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(raw.rstrip("\n"))
            m = _RECONNECT_RE.match(raw)
            if m:
                i = int(m.group("i"))
                j = int(m.group("j"))
                # Normalize order for stable results
                a, b = sorted((i, j))
                reconnections.append((a, b))
            continue

        parts = line.split()
        if len(parts) < 7:
            raise ValueError(
                f"Line {lineno}: expected 7 columns 'n T x y z r parent', got {len(parts)}"
            )
        if strict and len(parts) > 7:
            raise ValueError(
                f"Line {lineno}: expected exactly 7 columns, got {len(parts)}"
            )

        try:
            n = int(_coerce_int(parts[0]))
            t = int(_coerce_int(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            r = float(parts[5])
            parent = int(_coerce_int(parts[6]))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Line {lineno}: failed to parse values -> {e}") from e

        if n in records:
            prev = records[n]
            raise ValueError(
                f"Line {lineno}: duplicate node id {n} (previously defined at line {prev.line})"
            )

        records[n] = SWCRecord(n=n, t=t, x=x, y=y, z=z, r=r, parent=parent, line=lineno)

    # Validation: parent references
    if strict:
        for rec in records.values():
            if rec.parent == -1:
                continue
            if rec.parent not in records:
                raise ValueError(
                    f"Line {rec.line}: parent id {rec.parent} does not exist for node {rec.n}"
                )

    # Validation: reconnections require identical xyzr
    if validate_reconnections and reconnections:
        for a, b in reconnections:
            if a not in records or b not in records:
                raise ValueError(
                    f"Reconnection pair ({a}, {b}) refers to undefined node id(s)"
                )
            ra, rb = records[a], records[b]
            if not (
                _close(ra.x, rb.x, float_tol)
                and _close(ra.y, rb.y, float_tol)
                and _close(ra.z, rb.z, float_tol)
                and _close(ra.r, rb.r, float_tol)
            ):
                raise ValueError(
                    "Reconnection requires identical (x, y, z, r) but got:\n"
                    f"  {a}: (x={ra.x}, y={ra.y}, z={ra.z}, r={ra.r})\n"
                    f"  {b}: (x={rb.x}, y={rb.y}, z={rb.z}, r={rb.r})"
                )

    return SWCParseResult(
        records=records, reconnections=reconnections, comments=comments
    )


# Helpers -----------------------------------------------------------------------------------------


def _iter_lines(
    source: Union[str, os.PathLike, Iterable[str], io.TextIOBase],
) -> Iterator[Tuple[int, str]]:
    """Yield (1-based line number, line) from various sources.

    - Path-like or existing string path -> open and read
    - File-like object -> iterate its lines
    - Iterable of strings -> iterate
    - Other strings -> treat as content string
    """
    # Path-like or existing path string
    if isinstance(source, (str, os.PathLike)):
        path_str = os.fspath(source)
        if os.path.exists(path_str):
            with open(path_str, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    yield i, line
            return
        # If it's a string but not an existing path, treat it as content
        if isinstance(source, str):
            for i, line in enumerate(io.StringIO(source), start=1):
                yield i, line
            return
        raise FileNotFoundError(f"Path does not exist: {path_str}")

    # File-like
    if hasattr(source, "read"):
        for i, line in enumerate(source, start=1):
            yield i, line
        return

    # Iterable of strings
    try:
        iterator = iter(source)  # type: ignore[arg-type]
    except TypeError as e:  # noqa: BLE001
        raise TypeError(
            "Unsupported source type for parse_swc(); expected path, file-like, "
            "iterable of lines, or string content"
        ) from e
    else:
        for i, line in enumerate(iterator, start=1):
            yield i, line


def _close(a: float, b: float, tol: float) -> bool:
    return isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _coerce_int(value: str) -> int:
    """Coerce an integer possibly represented as float text like '3.0'."""
    # Some SWC files may include integer fields with trailing .0
    if value.strip().endswith(".0"):
        try:
            as_float = float(value)
            as_int = int(as_float)
            if float(as_int) == as_float:
                return as_int
        except Exception:  # noqa: BLE001
            pass
    return int(value)


__all__ = [
    "SWCRecord",
    "SWCParseResult",
    "parse_swc",
]
