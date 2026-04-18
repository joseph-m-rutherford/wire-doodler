#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from ..common import Index, Integer, Real
from ..errors import NeverImplement
from ..errors import NotYetImplemented
from ..errors import Unrecoverable
from ..r3 import R3Axes, R3Vector, r3vector_copy, axes3d_copy, r3vector_equality, TOLERANCE
import numpy as np


def _segment_segment_closest_points(p0, p1, q0, q1):
    """Return the closest point pair (c1, c2) between 3-D segments p0-p1 and q0-q1."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    _tol2 = TOLERANCE * TOLERANCE
    if a <= _tol2 and e <= _tol2:
        return (p0.copy(), q0.copy())
    if a <= _tol2:
        s = Real(0)
        t = Real(np.clip(f / e, 0.0, 1.0))
    elif e <= _tol2:
        t = Real(0)
        c = np.dot(d1, r)
        s = Real(np.clip(-c / a, 0.0, 1.0))
    else:
        b = np.dot(d1, d2)
        c = np.dot(d1, r)
        denom = a * e - b * b
        if abs(denom) > _tol2:
            s = Real(np.clip((b * f - c * e) / denom, 0.0, 1.0))
        else:
            s = Real(0)
        t = (b * s + f) / e
        if t < 0.0:
            t = Real(0)
            s = Real(np.clip(-c / a, 0.0, 1.0))
        elif t > 1.0:
            t = Real(1)
            s = Real(np.clip((b - c) / a, 0.0, 1.0))
    closest1 = p0 + s * d1
    closest2 = q0 + t * d2
    return (closest1, closest2)


def _is_shared_endpoint_intersection(
    p0: R3Vector,
    p1: R3Vector,
    q0: R3Vector,
    q1: R3Vector,
    c1: R3Vector,
    c2: R3Vector,
    reltol: Real,
) -> bool:
    """Return True when segment intersection is exactly at one endpoint of each segment."""
    if not r3vector_equality(c1, c2, reltol):
        return False

    p_at_endpoint = r3vector_equality(c1, p0, reltol) or r3vector_equality(c1, p1, reltol)
    q_at_endpoint = r3vector_equality(c2, q0, reltol) or r3vector_equality(c2, q1, reltol)
    return p_at_endpoint and q_at_endpoint


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


class MeshFunctions:
    """Immutable mapping from function index to subsegment pairs.

    Each function is represented by a unique unordered pair of subsegments
    that share exactly one vertex.
    """

    def __init__(self, mesh: "WireMesh3D") -> None:
        reltol = mesh.reltol
        n_subsegments = len(mesh.subsegment_index)

        endpoints: list[tuple[R3Vector, R3Vector]] = [
            mesh.subsegment_endpoints(Index(i)) for i in range(n_subsegments)
        ]

        pairs: list[tuple[Index, Index]] = []
        for i in range(n_subsegments):
            a0, a1 = endpoints[i]
            for j in range(i + 1, n_subsegments):
                b0, b1 = endpoints[j]
                shared_count = 0
                if r3vector_equality(a0, b0, reltol):
                    shared_count += 1
                if r3vector_equality(a0, b1, reltol):
                    shared_count += 1
                if r3vector_equality(a1, b0, reltol):
                    shared_count += 1
                if r3vector_equality(a1, b1, reltol):
                    shared_count += 1

                if shared_count == 1:
                    pairs.append((Index(i), Index(j)))

        self._function_subsegment_pairs: tuple[tuple[Index, Index], ...] = tuple(pairs)

    @property
    def function_subsegment_pairs(self) -> list[tuple[Index, Index]]:
        '''List indexed by function index: (subsegment_i, subsegment_j).'''
        return list(self._function_subsegment_pairs)

    @function_subsegment_pairs.setter
    def function_subsegment_pairs(self, value) -> None:
        raise NeverImplement('MeshFunctions function_subsegment_pairs are immutable')

    @property
    def function_map(self) -> dict[Index, tuple[Index, Index]]:
        '''Dict mapping function index -> (subsegment_i, subsegment_j).'''
        return {
            Index(i): pair
            for i, pair in enumerate(self._function_subsegment_pairs)
        }

    @function_map.setter
    def function_map(self, value) -> None:
        raise NeverImplement('MeshFunctions function_map is immutable')

    def pair(self, function_index: Index) -> tuple[Index, Index]:
        idx = int(function_index)
        if idx < 0 or idx >= len(self._function_subsegment_pairs):
            raise Unrecoverable(
                ''.join([
                    'MeshFunctions: function index ', str(idx),
                    ' is out of range for ', str(len(self._function_subsegment_pairs)),
                    ' functions',
                ])
            )
        return self._function_subsegment_pairs[idx]


class WireMesh3D:
    """A 3-D wire mesh assembled from named polylines sampled at a target density.

    Parameters
    ----------
    named_polylines:
        Mapping of segment names to ordered lists of 3-D points (as returned
        by :func:`as_xyz`).
    h:
        Target mesh density — the approximate arc-length spacing between
        generated mesh nodes along each wire segment.  Must be positive.
    reltol:
        Relative tolerance used to detect collisions between points, measured
        relative to each point's distance from the origin (via
        :func:`~doodler.r3.r3vector_equality`).  Must be positive.
    """

    def __init__(
        self,
        named_polylines: dict[str, list[R3Vector]],
        h: Real,
        reltol: Real,
    ) -> None:
        h = Real(h)
        reltol = Real(reltol)
        if h <= Real(0):
            raise Unrecoverable('WireMesh3D: mesh density h must be positive')
        if reltol <= Real(0):
            raise Unrecoverable('WireMesh3D: tolerance reltol must be positive')

        # Validate and deep-copy polylines; detect intra-polyline collisions.
        copied: dict[str, list[R3Vector]] = {}
        for name, points in named_polylines.items():
            pts = [r3vector_copy(p) for p in points]
            if len(pts) < 2:
                raise Unrecoverable(
                    ''.join(['WireMesh3D: polyline "', name, '" must have at least 2 points'])
                )
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if r3vector_equality(pts[i], pts[j], reltol):
                        raise Unrecoverable(
                            ''.join([
                                'WireMesh3D: polyline "', name, '" has colliding points at indices ',
                                str(i), ' and ', str(j),
                            ])
                        )
            copied[name] = pts

        # Inter-polyline collision checks.
        poly_items = list(copied.items())
        for i in range(len(poly_items)):
            name_a, pts_a = poly_items[i]
            for j in range(i + 1, len(poly_items)):
                name_b, pts_b = poly_items[j]
                # Segment-segment intersections are only allowed at shared endpoints.
                for ia in range(len(pts_a) - 1):
                    for ib in range(len(pts_b) - 1):
                        p0 = pts_a[ia]
                        p1 = pts_a[ia + 1]
                        q0 = pts_b[ib]
                        q1 = pts_b[ib + 1]
                        c1, c2 = _segment_segment_closest_points(
                            p0,
                            p1,
                            q0,
                            q1,
                        )
                        if r3vector_equality(c1, c2, reltol) and not _is_shared_endpoint_intersection(
                            p0,
                            p1,
                            q0,
                            q1,
                            c1,
                            c2,
                            reltol,
                        ):
                            raise NotYetImplemented(
                                ''.join([
                                    'WireMesh3D: intersecting segments are not yet supported ',
                                    '(polylines "', name_a, '" segment ', str(ia),
                                    ' and "', name_b, '" segment ', str(ib), ')',
                                ])
                            )

        self._named_polylines = copied
        self._h = h
        self._reltol = reltol

        # Compute the number of even subsegments for each polyline segment.
        # Every segment must have at least 1 subsegment.
        named_subsegment_counts: dict[str, list[Integer]] = {}
        for name, pts in self._named_polylines.items():
            counts: list[Integer] = []
            for k in range(len(pts) - 1):
                length = Real(np.linalg.norm(pts[k + 1] - pts[k]))
                counts.append(Integer(max(1, int(np.ceil(length / h)))))
            named_subsegment_counts[name] = counts
        self._named_subsegment_counts = named_subsegment_counts

        # Build flat subsegment index: sorted by name, then segment, then subsegment.
        # Each entry maps mesh_index -> (wire_name, segment_index, subsegment_index).
        subsegment_index: list[tuple[str, Index, Index]] = []
        for name in sorted(self._named_subsegment_counts.keys()):
            for seg_idx, count in enumerate(self._named_subsegment_counts[name]):
                for sub_idx in range(count):
                    subsegment_index.append((name, Index(seg_idx), Index(sub_idx)))
        self._subsegment_index = subsegment_index
        self._mesh_functions = MeshFunctions(self)

    @property
    def named_polylines(self) -> dict[str, list[R3Vector]]:
        '''Named polylines in global x, y, z coordinates.'''
        return {name: list(pts) for name, pts in self._named_polylines.items()}

    @named_polylines.setter
    def named_polylines(self, value) -> None:
        raise NeverImplement('WireMesh3D named_polylines are immutable')

    @property
    def h(self) -> Real:
        '''Target mesh density.'''
        return self._h

    @h.setter
    def h(self, value) -> None:
        raise NeverImplement('WireMesh3D h is immutable')

    @property
    def reltol(self) -> Real:
        '''Relative tolerance for collision detection.'''
        return self._reltol

    @reltol.setter
    def reltol(self, value) -> None:
        raise NeverImplement('WireMesh3D reltol is immutable')

    @property
    def named_subsegment_counts(self) -> dict[str, list[Integer]]:
        '''Subsegment counts per polyline segment (number of even subdivisions of each segment).'''
        return {name: list(counts) for name, counts in self._named_subsegment_counts.items()}

    @named_subsegment_counts.setter
    def named_subsegment_counts(self, value) -> None:
        raise NeverImplement('WireMesh3D named_subsegment_counts are immutable')

    @property
    def subsegment_index(self) -> list[tuple[str, Index, Index]]:
        '''Flat list mapping each mesh index to (wire_name, segment_index, subsegment_index).

        Names are visited in lexicographical order; within each named polyline the
        segments are visited in order and each segment's subsegments are visited
        in order, so the list position is the global mesh index.
        '''
        return list(self._subsegment_index)

    @subsegment_index.setter
    def subsegment_index(self, value) -> None:
        raise NeverImplement('WireMesh3D subsegment_index is immutable')

    def subsegment_endpoints(self, mesh_index: Index) -> tuple[R3Vector, R3Vector]:
        '''Endpoints of a flat-indexed subsegment.'''
        idx = int(mesh_index)
        if idx < 0 or idx >= len(self._subsegment_index):
            raise Unrecoverable(
                ''.join([
                    'WireMesh3D: mesh index ', str(idx),
                    ' is out of range for ',
                    str(len(self._subsegment_index)), ' subsegments',
                ])
            )

        name, seg_idx, sub_idx = self._subsegment_index[idx]
        seg_i = int(seg_idx)
        sub_i = int(sub_idx)

        p0 = self._named_polylines[name][seg_i]
        p1 = self._named_polylines[name][seg_i + 1]
        count = int(self._named_subsegment_counts[name][seg_i])

        alpha0 = Real(sub_i) / Real(count)
        alpha1 = Real(sub_i + 1) / Real(count)

        start = np.array(p0 + alpha0 * (p1 - p0), dtype=Real)
        end = np.array(p0 + alpha1 * (p1 - p0), dtype=Real)
        return (start, end)

    @property
    def mesh_functions(self) -> MeshFunctions:
        '''Function-index mapping for this mesh.'''
        return self._mesh_functions

    @mesh_functions.setter
    def mesh_functions(self, value) -> None:
        raise NeverImplement('WireMesh3D mesh_functions are immutable')


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
