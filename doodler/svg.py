#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import xml.etree.ElementTree as ET

from .common import Real
from .errors import Unrecoverable


def _local_tag(element: ET.Element) -> str:
    """Return the local tag name, stripping any XML namespace prefix."""
    tag = element.tag
    if tag.startswith('{'):
        return tag[tag.index('}') + 1:]
    return tag


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
        attribute, ids are duplicated, or coordinate data is malformed.
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
        if local not in ('line', 'polyline'):
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
            try:
                x1 = Real(element.get('x1', '0'))
                y1 = Real(element.get('y1', '0'))
                x2 = Real(element.get('x2', '0'))
                y2 = Real(element.get('y2', '0'))
            except (ValueError, TypeError) as exc:
                raise Unrecoverable(exc)
            result[element_id] = [(x1, y1), (x2, y2)]
        else:  # polyline
            points_attr = element.get('points', '').strip()
            if not points_attr:
                raise Unrecoverable(
                    f'Polyline id={element_id!r} has empty points attribute'
                )
            result[element_id] = _parse_polyline_points(points_attr, element_id)

    return result
