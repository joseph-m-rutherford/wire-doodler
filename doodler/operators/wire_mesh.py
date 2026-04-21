#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np

from ..common import Index, Integer, Real
from ..errors import NeverImplement
from ..errors import NotYetImplemented
from ..errors import Unrecoverable
from ..operators.mesh_functions import MeshFunctions
from ..r3 import R3Vector, r3vector_copy, r3vector_equality, Octree, TOLERANCE


class PointRegistry:
    """Fast-lookup registry mapping 3-D points to unique integer indices.

    Uses an :class:`~doodler.r3.Octree` to partition points into cells so
    that ``get_or_insert`` only checks candidates in the same octree cell
    via :func:`~doodler.r3.r3vector_equality`.
    """

    def __init__(self, min_xyz: R3Vector, max_xyz: R3Vector, reltol: Real) -> None:
        reltol = Real(reltol)
        if reltol <= Real(0):
            raise Unrecoverable('PointRegistry: reltol must be positive')
        self._reltol = reltol
        self._points: list[R3Vector] = []

        min_xyz = r3vector_copy(min_xyz)
        max_xyz = r3vector_copy(max_xyz)

        # Derive an absolute tolerance for the octree from a bound on the
        # largest point norm in the bounding box. This keeps the octree cell
        # width consistent with r3vector_equality(), which scales tolerance
        # by the Euclidean norm of the point rather than by its largest
        # coordinate component.
        max_point_norm = Real(max(
            float(np.linalg.norm(min_xyz)),
            float(np.linalg.norm(max_xyz)),
            1.0,
        ))
        abstol = Real(reltol * max_point_norm)

        self._octree = Octree(min_xyz, max_xyz, abstol)
        # Map Morton key -> list of point indices sharing that cell.
        self._cell_indices: dict[int, list[int]] = {}

    @property
    def reltol(self) -> Real:
        '''Relative tolerance for point equality.'''
        return self._reltol

    @reltol.setter
    def reltol(self, value) -> None:
        raise NeverImplement('PointRegistry reltol is immutable')

    @property
    def min_xyz(self) -> R3Vector:
        '''Minimum corner of the octree bounding box.'''
        return self._octree.min_xyz

    @min_xyz.setter
    def min_xyz(self, value) -> None:
        raise NeverImplement('PointRegistry min_xyz is immutable')

    @property
    def max_xyz(self) -> R3Vector:
        '''Maximum corner of the octree bounding box.'''
        return self._octree.max_xyz

    @max_xyz.setter
    def max_xyz(self, value) -> None:
        raise NeverImplement('PointRegistry max_xyz is immutable')

    @property
    def count(self) -> int:
        '''Number of unique points in the registry.'''
        return len(self._points)

    @property
    def points(self) -> list[R3Vector]:
        '''Copy of all registered points.'''
        return [r3vector_copy(p) for p in self._points]

    @points.setter
    def points(self, value) -> None:
        raise NeverImplement('PointRegistry points are immutable')

    def point(self, index: Index) -> R3Vector:
        '''Return a copy of the point at *index*.'''
        idx = int(index)
        if idx < 0 or idx >= len(self._points):
            raise Unrecoverable(
                ''.join([
                    'PointRegistry: index ', str(idx),
                    ' is out of range for ', str(len(self._points)), ' points',
                ])
            )
        return r3vector_copy(self._points[idx])

    def get_or_insert(self, point: R3Vector) -> Index:
        '''Return the index of *point*, inserting it first if no match exists.'''
        point = r3vector_copy(point)
        key = self._octree.morton_key(point)
        candidates = self._cell_indices.get(key, [])
        for idx in candidates:
            if r3vector_equality(self._points[idx], point, self._reltol):
                return Index(idx)
        new_idx = len(self._points)
        self._points.append(point)
        if key not in self._cell_indices:
            self._cell_indices[key] = []
        self._cell_indices[key].append(new_idx)
        return Index(new_idx)


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

        # Build point registry and subsegment point pairs.
        # Compute bounding box from polyline vertices (subsegment endpoints
        # are interpolated within the convex hull of these vertices).
        all_pts_flat = np.array([pt for pts in copied.values() for pt in pts])
        bbox_min = r3vector_copy(np.min(all_pts_flat, axis=0))
        bbox_max = r3vector_copy(np.max(all_pts_flat, axis=0))
        # Pad to guarantee strict min < max and avoid boundary issues.
        max_coord = Real(max(
            float(np.max(np.abs(bbox_max))),
            float(np.max(np.abs(bbox_min))),
            1.0,
        ))
        pad = reltol * max_coord
        bbox_min -= pad
        bbox_max += pad
        self._point_registry = PointRegistry(bbox_min, bbox_max, reltol)
        subsegment_point_pairs: list[tuple[Index, Index]] = []
        for mesh_idx in range(len(self._subsegment_index)):
            start, end = self.subsegment_endpoints(Index(mesh_idx))
            start_idx = self._point_registry.get_or_insert(start)
            end_idx = self._point_registry.get_or_insert(end)
            subsegment_point_pairs.append((start_idx, end_idx))
        self._subsegment_point_pairs = subsegment_point_pairs

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
    def vertex_count(self) -> int:
        '''Number of unique vertices in the mesh.'''
        return self._point_registry.count

    @vertex_count.setter
    def vertex_count(self, value) -> None:
        raise NeverImplement('WireMesh3D vertex_count is immutable')

    def vertex_xyz(self, index: Index) -> R3Vector:
        '''Return a copy of the 3-D position of vertex *index*.'''
        return self._point_registry.point(index)

    @property
    def point_registry(self) -> PointRegistry:
        '''Point registry containing unique 3-D points of this mesh.'''
        return self._point_registry

    @point_registry.setter
    def point_registry(self, value) -> None:
        raise NeverImplement('WireMesh3D point_registry is immutable')

    @property
    def subsegment_point_pairs(self) -> list[tuple[Index, Index]]:
        '''Subsegment connectivity as pairs of point-registry indices.'''
        return list(self._subsegment_point_pairs)

    @subsegment_point_pairs.setter
    def subsegment_point_pairs(self, value) -> None:
        raise NeverImplement('WireMesh3D subsegment_point_pairs are immutable')

    @property
    def mesh_functions(self) -> MeshFunctions:
        '''Function-index mapping for this mesh.'''
        return self._mesh_functions

    @mesh_functions.setter
    def mesh_functions(self, value) -> None:
        raise NeverImplement('WireMesh3D mesh_functions are immutable')

    @classmethod
    def _with_shared_registry(
        cls,
        source: "WireMesh3D",
        registry: PointRegistry,
        remap: list[Index],
    ) -> "WireMesh3D":
        '''Construct a new WireMesh3D that shares *registry* with remapped point indices.'''
        instance = object.__new__(cls)
        instance._named_polylines = {
            name: [r3vector_copy(pt) for pt in pts]
            for name, pts in source._named_polylines.items()
        }
        instance._h = source._h
        instance._reltol = source._reltol
        instance._named_subsegment_counts = {
            name: list(counts) for name, counts in source._named_subsegment_counts.items()
        }
        instance._subsegment_index = list(source._subsegment_index)
        instance._point_registry = registry
        instance._subsegment_point_pairs = [
            (remap[int(a)], remap[int(b)]) for a, b in source._subsegment_point_pairs
        ]
        instance._mesh_functions = MeshFunctions(instance)
        return instance


def unify_meshes(
    a: WireMesh3D,
    b: WireMesh3D,
) -> tuple[WireMesh3D, WireMesh3D]:
    """Unify two meshes so they share the same point registry.

    Points that are equal (within the larger of the two meshes' tolerances)
    receive the same index in the shared registry.

    Returns
    -------
    unified_a : WireMesh3D
        Copy of *a* whose point registry and subsegment point pairs reference
        the shared registry.
    unified_b : WireMesh3D
        Copy of *b* likewise remapped into the same shared registry.
    """
    reltol = Real(max(float(a.reltol), float(b.reltol)))
    shared_min = r3vector_copy(np.minimum(a.point_registry.min_xyz, b.point_registry.min_xyz))
    shared_max = r3vector_copy(np.maximum(a.point_registry.max_xyz, b.point_registry.max_xyz))
    shared = PointRegistry(shared_min, shared_max, reltol)

    remap_a: list[Index] = []
    for i in range(a.point_registry.count):
        remap_a.append(shared.get_or_insert(a.point_registry.point(Index(i))))

    remap_b: list[Index] = []
    for i in range(b.point_registry.count):
        remap_b.append(shared.get_or_insert(b.point_registry.point(Index(i))))

    unified_a = WireMesh3D._with_shared_registry(a, shared, remap_a)
    unified_b = WireMesh3D._with_shared_registry(b, shared, remap_b)
    return unified_a, unified_b
