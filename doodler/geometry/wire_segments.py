#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from ..common import Real
from ..errors import NeverImplement
from ..errors import Unrecoverable
from ..r3 import R3Axes, R3Vector, r3vector_copy, axes3d_copy
import numpy as np


class WireSegment2D:
    """A single 2-D wire segment parsed from an SVG element and a description.

    This is a lightweight container used by :func:`read_svg` to represent
    individual wire-like segments in the SVG, regardless of which specific
    element type produced them (e.g., ``<path>``, ``<line>``, ``<polyline>``).

    Attributes
    ----------
    points:
        Ordered list of (x, y) coordinate pairs in the source SVG coordinate
        system describing the 2-D wire geometry. These are typically extracted
        from the element's geometry attributes (such as a path ``d`` attribute,
        ``x1``/``y1``/``x2``/``y2`` on a line, or a polyline ``points`` list).
    description:
        Non-empty string taken from the required ``<desc>`` child element of
        the source SVG element.
    """

    def __init__(self, points: list[tuple[Real, Real]], description: str) -> None:
        # Store an internal immutable copy of the points to prevent external mutation.
        self._points: tuple[tuple[Real, Real], ...] = tuple(points)
        self._description = description

    @property
    def points(self) -> list[tuple[Real, Real]]:
        '''Ordered list of (x, y) coordinate pairs describing the 2-D wire segment.'''
        # Return a defensive copy so callers cannot mutate internal state.
        return list(self._points)

    @points.setter
    def points(self, value) -> None:
        raise NeverImplement('WireSegment2D points are immutable')

    @property
    def description(self) -> str:
        '''Non-empty string taken from the required <desc> child element.'''
        return self._description

    @description.setter
    def description(self, value) -> None:
        raise NeverImplement('WireSegment2D description is immutable')


def as_xyz(
    segments: dict[str, "WireSegment2D"],
    uvw: R3Axes,
    xyz_offset: R3Vector,
) -> dict[str, list[R3Vector]]:
    """Convert 2-D SVG segments into 3-D global coordinates.

    Each (u, v) point in *segments* is treated as a position in the plane
    w=0 of the orthonormal frame *uvw*, then shifted by *xyz_offset*.

    Parameters
    ----------
    segments:
        A dict mapping names to :class:`WireSegment2D` instances.
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
    for name, segment in segments.items():
        xyz_points: list[R3Vector] = []
        for u, v in segment.points:
            point = np.array(u * u_hat + v * v_hat, dtype=Real) + offset
            xyz_points.append(point)
        result[name] = xyz_points
    return result
