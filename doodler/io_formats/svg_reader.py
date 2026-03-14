#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import re
import xml.etree.ElementTree as ET

import numpy as np

from ..common import Real
from ..errors import NotYetImplemented, Unrecoverable
from ..geometry.wire_segments import WireSegment2D, as_xyz
from .vtk_writer import export_polylines
from ..r3 import R3Axes, R3Vector, r3vector_copy, axes3d_copy


def _local_tag(element: ET.Element) -> str:
    """Return the local tag name, stripping any XML namespace prefix."""
    tag = element.tag
    if tag.startswith('{'):
        return tag[tag.index('}') + 1:]
    return tag


_PATH_TOKEN_RE = re.compile(
    r'([MmLlHhVvZzAaQqTtCcSs])'
    r'|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
)
_CURVE_COMMAND_RE = re.compile(r'[AaQqTtCcSs]')


def _parse_path_d(d_str: str, element_id: str) -> list[tuple[Real, Real]]:
    """Parse an SVG path d attribute into a list of (x, y) pairs.

    Raises NotYetImplemented if the path contains arc (A/a), quadratic
    bezier (Q/q, T/t), or cubic bezier (C/c, S/s) commands.
    Supports M, m, L, l, H, h, V, v, Z, z.
    """
    if _CURVE_COMMAND_RE.search(d_str):
        raise NotYetImplemented(
            f'Path id={element_id!r} contains unsupported curve commands'
        )

    tokens = []
    for m in _PATH_TOKEN_RE.finditer(d_str):
        if m.group(1):
            tokens.append(m.group(1))
        else:
            tokens.append(m.group(2))

    points: list[tuple[Real, Real]] = []
    current_x = Real(0)
    current_y = Real(0)
    start_x = Real(0)
    start_y = Real(0)
    i = 0
    current_cmd: str | None = None

    while i < len(tokens):
        token = tokens[i]
        if isinstance(token, str) and token.isalpha():
            current_cmd = token
            i += 1
            if current_cmd in ('Z', 'z'):
                current_x = start_x
                current_y = start_y
                points.append((current_x, current_y))
                continue

        if current_cmd is None:
            raise Unrecoverable(
                f'Path id={element_id!r} has coordinate data before any command'
            )

        if current_cmd in ('M', 'm'):
            try:
                x = Real(tokens[i])
                y = Real(tokens[i + 1])
            except (IndexError, ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            i += 2
            if current_cmd == 'm':
                current_x += x
                current_y += y
            else:
                current_x = x
                current_y = y
            start_x, start_y = current_x, current_y
            points.append((current_x, current_y))
            # Subsequent coordinate pairs after M/m are treated as implicit L/l
            current_cmd = 'L' if current_cmd == 'M' else 'l'
        elif current_cmd in ('L', 'l'):
            try:
                x = Real(tokens[i])
                y = Real(tokens[i + 1])
            except (IndexError, ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            i += 2
            if current_cmd == 'l':
                current_x += x
                current_y += y
            else:
                current_x = x
                current_y = y
            points.append((current_x, current_y))
        elif current_cmd in ('H', 'h'):
            try:
                x = Real(tokens[i])
            except (IndexError, ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            i += 1
            if current_cmd == 'h':
                current_x += x
            else:
                current_x = x
            points.append((current_x, current_y))
        elif current_cmd in ('V', 'v'):
            try:
                y = Real(tokens[i])
            except (IndexError, ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            i += 1
            if current_cmd == 'v':
                current_y += y
            else:
                current_y = y
            points.append((current_x, current_y))
        else:
            raise Unrecoverable(
                f'Path id={element_id!r} contains unknown command {current_cmd!r}'
            )

    if not points:
        raise Unrecoverable(f'Path id={element_id!r} has no drawable points')
    return points


def _parse_polyline_points(points_str: str, element_id: str) -> list[tuple[Real, Real]]:
    """Parse a SVG points attribute string into a list of (x, y) pairs."""
    tokens = points_str.replace(',', ' ').split()
    if len(tokens) % 2 != 0:
        raise Unrecoverable(
            f'Polyline id={element_id!r} points attribute has odd token count: {points_str!r}'
        )
    try:
        return [(Real(tokens[i]), Real(tokens[i + 1])) for i in range(0, len(tokens), 2)]
    except (ValueError, TypeError) as exc:
        raise Unrecoverable(exc)


def read_svg(path: str) -> dict[str, WireSegment2D]:
    """Read an SVG file and return wire segments keyed by element id.

    Every <line>, <polyline>, and <path> element must carry an id attribute
    and a <desc> child element containing a non-empty description string;
    all produce :class:`WireSegment2D` instances.

    Parameters
    ----------
    path:
        Filesystem path to the SVG file.

    Returns
    -------
    dict mapping each element id to a :class:`WireSegment2D` instance.

    Raises
    ------
    Unrecoverable:
        If the file cannot be read or parsed, any element lacks an id
        attribute, a line is missing a coordinate attribute, ids are
        duplicated, coordinate data is malformed, or any element is missing
        its required non-empty <desc> child.
    NotYetImplemented:
        If a path contains arc, quadratic bezier, or cubic bezier commands.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise Unrecoverable(exc)
    except OSError as exc:
        raise Unrecoverable(exc)

    result: dict[str, WireSegment2D] = {}

    for element in tree.iter():
        local = _local_tag(element)
        if local not in ('line', 'polyline', 'path'):
            continue

        element_id = element.get('id')
        if element_id is None:
            raise Unrecoverable(
                f'SVG <{local}> element is missing required id attribute'
            )
        if element_id in result:
            raise Unrecoverable(
                f'Duplicate id {element_id!r} found in SVG file {path!r}'
            )

        if local == 'line':
            for attr in ('x1', 'y1', 'x2', 'y2'):
                if element.get(attr) is None:
                    raise Unrecoverable(
                        f'Line id={element_id!r} is missing required coordinate attribute {attr!r}'
                    )
            try:
                x1 = Real(element.get('x1'))
                y1 = Real(element.get('y1'))
                x2 = Real(element.get('x2'))
                y2 = Real(element.get('y2'))
            except (ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            desc_el = next((c for c in element if _local_tag(c) == 'desc'), None)
            if desc_el is None:
                raise Unrecoverable(
                    f'Line id={element_id!r} is missing required <desc> child element'
                )
            desc_text = (desc_el.text or '').strip()
            if not desc_text:
                raise Unrecoverable(
                    f'Line id={element_id!r} <desc> element must contain non-empty text'
                )
            result[element_id] = WireSegment2D([(x1, y1), (x2, y2)], desc_text)
        elif local == 'polyline':
            points_attr = element.get('points', '').strip()
            if not points_attr:
                raise Unrecoverable(
                    f'Polyline id={element_id!r} has empty points attribute'
                )
            points = _parse_polyline_points(points_attr, element_id)
            desc_el = next((c for c in element if _local_tag(c) == 'desc'), None)
            if desc_el is None:
                raise Unrecoverable(
                    f'Polyline id={element_id!r} is missing required <desc> child element'
                )
            desc_text = (desc_el.text or '').strip()
            if not desc_text:
                raise Unrecoverable(
                    f'Polyline id={element_id!r} <desc> element must contain non-empty text'
                )
            result[element_id] = WireSegment2D(points, desc_text)
        else:  # path
            d_attr = element.get('d', '').strip()
            if not d_attr:
                raise Unrecoverable(
                    f'Path id={element_id!r} has empty d attribute'
                )
            desc_el = next((c for c in element if _local_tag(c) == 'desc'), None)
            if desc_el is None:
                raise Unrecoverable(
                    f'Path id={element_id!r} is missing required <desc> child element'
                )
            desc_text = (desc_el.text or '').strip()
            if not desc_text:
                raise Unrecoverable(
                    f'Path id={element_id!r} <desc> element must contain non-empty text'
                )
            points = _parse_path_d(d_attr, element_id)
            result[element_id] = WireSegment2D(points, desc_text)

    return result


# `as_xyz` is implemented in `doodler.geometry.wire_segments`.


def export_polylines(segments: dict[str, list[R3Vector]], filename: str) -> None:
    """Write 3-D segment data to a legacy VTK ASCII file as LINES.

    Parameters
    ----------
    segments:
        Output of :func:`as_xyz` — a dict mapping names to lists of
        length-3 numpy arrays in global x, y, z coordinates.
    filename:
        Destination file path.  Overwrites existing files.

    Raises
    ------
    Unrecoverable:
        If the output file cannot be written.
    """
    # Delegated to doodler.io_formats.vtk_writer.export_polylines
    return export_polylines(segments, filename)
