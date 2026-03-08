#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import re
import xml.etree.ElementTree as ET

import numpy as np

from .common import Real
from .errors import NotYetImplemented, Unrecoverable
from .r3 import R3Axes, R3Vector, r3vector_copy, axes3d_copy


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


def read_svg(path: str) -> dict[str, list[tuple[Real, Real]]]:
    """Read an SVG file and return wire segments keyed by element id.

    Every <line> and <polyline> element must carry an id attribute.
    Lines produce two-point lists; polylines produce one point per vertex.

    Parameters
    ----------
    path:
        Filesystem path to the SVG file.

    Returns
    -------
    dict mapping each element id to a list of (x, y) coordinate pairs.

    Raises
    ------
    Unrecoverable:
        If the file cannot be read or parsed, any line/polyline lacks an id
        attribute, any line is missing a coordinate attribute, ids are
        duplicated, or coordinate data is malformed.
    """
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise Unrecoverable(exc)
    except OSError as exc:
        raise Unrecoverable(exc)

    result: dict[str, list[tuple[Real, Real]]] = {}

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
            result[element_id] = [(x1, y1), (x2, y2)]
        elif local == 'polyline':
            points_attr = element.get('points', '').strip()
            if not points_attr:
                raise Unrecoverable(
                    f'Polyline id={element_id!r} has empty points attribute'
                )
            result[element_id] = _parse_polyline_points(points_attr, element_id)
        else:  # path
            d_attr = element.get('d', '').strip()
            if not d_attr:
                raise Unrecoverable(
                    f'Path id={element_id!r} has empty d attribute'
                )
            result[element_id] = _parse_path_d(d_attr, element_id)

    return result


def as_xyz(
    segments: dict[str, list[tuple[Real, Real]]],
    uvw: R3Axes,
    xyz_offset: R3Vector,
) -> dict[str, list[R3Vector]]:
    """Convert 2-D SVG segments into 3-D global coordinates.

    Each (u, v) point in *segments* is treated as a position in the plane
    w=0 of the orthonormal frame *uvw*, then shifted by *xyz_offset*.

    Parameters
    ----------
    segments:
        Output of :func:`read_svg` — a dict mapping names to lists of
        (u, v) coordinate pairs.
    uvw:
        A (3, 3) array whose rows are the orthonormal basis vectors
        u = uvw[0], v = uvw[1], w = uvw[2].  Validated via
        :func:`axes3d_copy`.
    xyz_offset:
        A length-3 global offset added to every converted point.
        Validated via :func:`r3vector_copy`.

    Returns
    -------
    dict mapping each segment name to a list of length-3 ``numpy`` arrays
    (dtype :data:`Real`) in global x, y, z coordinates.
    """
    frame = axes3d_copy(uvw)
    offset = r3vector_copy(xyz_offset)
    u_hat = frame[0]
    v_hat = frame[1]

    result: dict[str, list[R3Vector]] = {}
    for name, points in segments.items():
        xyz_points: list[R3Vector] = []
        for u, v in points:
            point = np.array(u * u_hat + v * v_hat, dtype=Real) + offset
            xyz_points.append(point)
        result[name] = xyz_points
    return result


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
    # Flatten all points and record per-segment index ranges.
    all_points: list[R3Vector] = []
    line_index_lists: list[list[int]] = []
    for pts in segments.values():
        start = len(all_points)
        all_points.extend(pts)
        line_index_lists.append(list(range(start, start + len(pts))))

    n_points = len(all_points)
    n_lines = len(line_index_lists)
    # VTK legacy: total_size = sum of (npts_per_line + 1) for each line
    total_size = sum(len(idx) + 1 for idx in line_index_lists)

    try:
        with open(filename, 'w', encoding='ascii') as f:
            f.write('# vtk DataFile Version 2.0\n')
            f.write('wire-doodler polylines\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')
            f.write(f'POINTS {n_points} float\n')
            for pt in all_points:
                f.write(f'{float(pt[0])} {float(pt[1])} {float(pt[2])}\n')
            f.write(f'LINES {n_lines} {total_size}\n')
            for indices in line_index_lists:
                f.write(' '.join([str(len(indices))] + [str(i) for i in indices]) + '\n')
    except OSError as exc:
        raise Unrecoverable(exc)
