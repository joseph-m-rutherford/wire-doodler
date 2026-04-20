#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np

from ..common import Index, Integer, Real
from ..errors import NeverImplement
from ..errors import NotYetImplemented
from ..errors import Unrecoverable
from ..operators.mesh_functions import MeshFunctions
from ..r3 import R3Vector, r3vector_copy, r3vector_equality, TOLERANCE


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

        # Compute the number of uniform subsegments for each polyline segment.
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
        return {
            name: [r3vector_copy(pt) for pt in pts]
            for name, pts in self._named_polylines.items()
        }

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
        '''Subsegment counts per polyline segment (number of uniform subdivisions of each segment).'''
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
