#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np

from ..common import Index, Real
from ..errors import NeverImplement, Unrecoverable
from . import R3Vector, vector_copy, vector_equality
from .octree import Octree


class PointRegistry:
    """Fast-lookup registry mapping 3-D points to unique integer indices.

    Uses an :class:`~doodler.r3.Octree` to partition points into cells so
    that ``get_or_insert`` only checks candidates in the same octree cell
    via :func:`~doodler.r3.vector_equality`.
    """

    def __init__(self, min_xyz: R3Vector, max_xyz: R3Vector, reltol: Real) -> None:
        reltol = Real(reltol)
        if reltol <= Real(0):
            raise Unrecoverable('PointRegistry: reltol must be positive')
        self._reltol = reltol
        self._points: list[R3Vector] = []

        min_xyz = vector_copy(min_xyz)
        max_xyz = vector_copy(max_xyz)

        # Derive an absolute tolerance for the octree from a bound on the
        # largest point norm in the bounding box. This keeps the octree cell
        # width consistent with vector_equality(), which scales tolerance
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
    def abstol(self) -> Real:
        '''Absolute tolerance used by the underlying octree.'''
        return self._octree.tolerance

    @abstol.setter
    def abstol(self, value) -> None:
        raise NeverImplement('PointRegistry abstol is immutable')

    @property
    def points(self) -> list[R3Vector]:
        '''Copy of all registered points.'''
        return [vector_copy(p) for p in self._points]

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
        return vector_copy(self._points[idx])

    def morton_keys_for_aabb(
        self,
        min_xyz: R3Vector,
        max_xyz: R3Vector,
        max_cells: int | None = None,
    ) -> list[int] | None:
        '''Return all Morton keys overlapped by an axis-aligned bounding box.

        Returns ``None`` when the overlapped cell count exceeds *max_cells*.
        '''
        min_xyz = vector_copy(min_xyz)
        max_xyz = vector_copy(max_xyz)

        for i in range(3):
            if min_xyz[i] > max_xyz[i]:
                raise Unrecoverable('PointRegistry: invalid AABB with min > max')

        oct_min = self._octree.min_xyz
        oct_max = self._octree.max_xyz

        # Disjoint from registry bounds.
        for i in range(3):
            if max_xyz[i] < oct_min[i] or min_xyz[i] > oct_max[i]:
                return []

        clamped_min = np.maximum(min_xyz, oct_min)
        clamped_max = np.minimum(max_xyz, oct_max)

        lo = self._octree._point_to_cell(clamped_min)
        hi = self._octree._point_to_cell(clamped_max)

        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        total_cells = nx * ny * nz

        if max_cells is not None and total_cells > max_cells:
            return None

        keys: list[int] = []
        depth = self._octree.depth
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    keys.append(Octree._interleave_bits(ix, iy, iz, depth))
        return keys

    def get_or_insert(self, point: R3Vector) -> Index:
        '''Return the index of *point*, inserting it first if no match exists.'''
        point = vector_copy(point)
        key = self._octree.morton_key(point)
        candidates = self._cell_indices.get(key, [])
        for idx in candidates:
            if vector_equality(self._points[idx], point, self._reltol):
                return Index(idx)
        new_idx = len(self._points)
        self._points.append(point)
        if key not in self._cell_indices:
            self._cell_indices[key] = []
        self._cell_indices[key].append(new_idx)
        return Index(new_idx)
