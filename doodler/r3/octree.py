#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np

from ..errors import NeverImplement, Unrecoverable
from . import Real, R3Vector, vector_copy


class Octree:
    """Axis-aligned octree for fast 3-D point lookup via Morton keys.

    The bounding box is subdivided until the cell size in each dimension is
    at most *tolerance* (the minimum resolvable distance).  Points that land
    in the same cell share a Morton key.

    Parameters
    ----------
    min_xyz : R3Vector
        Minimum corner of the axis-aligned bounding box.
    max_xyz : R3Vector
        Maximum corner of the axis-aligned bounding box.  Every component
        must be strictly greater than the corresponding *min_xyz* component.
    tolerance : Real
        Absolute tolerance — the minimum resolvable distance.  Determines
        the finest subdivision level of the octree.  Must be positive.
    """

    def __init__(
        self,
        min_xyz: R3Vector,
        max_xyz: R3Vector,
        tolerance: Real,
    ) -> None:
        min_xyz = vector_copy(min_xyz)
        max_xyz = vector_copy(max_xyz)
        tolerance = Real(tolerance)

        if tolerance <= Real(0):
            raise Unrecoverable('Octree: tolerance must be positive')

        for i in range(3):
            if min_xyz[i] >= max_xyz[i]:
                raise Unrecoverable(
                    'Octree: min_xyz must be strictly less than max_xyz in all dimensions'
                )

        self._min_xyz = min_xyz
        self._max_xyz = max_xyz
        self._tolerance = tolerance

        extent = max_xyz - min_xyz
        max_extent = Real(np.max(extent))

        # Depth such that max_extent / 2^depth <= tolerance.
        self._depth = max(0, int(np.ceil(np.log2(float(max_extent / tolerance)))))
        self._n_cells = 2 ** self._depth

        # Associative container: Morton key -> stored 3-D coordinates.
        self._points: dict[int, R3Vector] = {}

    # -- immutable properties ------------------------------------------------

    @property
    def min_xyz(self) -> R3Vector:
        '''Minimum corner of the bounding box.'''
        return vector_copy(self._min_xyz)

    @min_xyz.setter
    def min_xyz(self, value) -> None:
        raise NeverImplement('Octree min_xyz is immutable')

    @property
    def max_xyz(self) -> R3Vector:
        '''Maximum corner of the bounding box.'''
        return vector_copy(self._max_xyz)

    @max_xyz.setter
    def max_xyz(self, value) -> None:
        raise NeverImplement('Octree max_xyz is immutable')

    @property
    def tolerance(self) -> Real:
        '''Absolute tolerance (minimum resolvable distance).'''
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value) -> None:
        raise NeverImplement('Octree tolerance is immutable')

    @property
    def depth(self) -> int:
        '''Number of octree subdivision levels.'''
        return self._depth

    @depth.setter
    def depth(self, value) -> None:
        raise NeverImplement('Octree depth is immutable')

    @property
    def count(self) -> int:
        '''Number of stored points.'''
        return len(self._points)

    # -- internal helpers ----------------------------------------------------

    def _point_to_cell(self, point: R3Vector) -> tuple[int, int, int]:
        '''Map a 3-D point to integer cell coordinates at the finest level.'''
        extent = self._max_xyz - self._min_xyz
        normalized = (point - self._min_xyz) / extent
        ix = int(np.clip(int(np.floor(float(normalized[0]) * self._n_cells)), 0, self._n_cells - 1))
        iy = int(np.clip(int(np.floor(float(normalized[1]) * self._n_cells)), 0, self._n_cells - 1))
        iz = int(np.clip(int(np.floor(float(normalized[2]) * self._n_cells)), 0, self._n_cells - 1))
        return (ix, iy, iz)

    @staticmethod
    def _interleave_bits(x: int, y: int, z: int, depth: int) -> int:
        '''Compute a Morton key by interleaving bits of *x*, *y*, *z*.'''
        key = 0
        for bit in range(depth):
            key |= ((x >> bit) & 1) << (3 * bit)
            key |= ((y >> bit) & 1) << (3 * bit + 1)
            key |= ((z >> bit) & 1) << (3 * bit + 2)
        return key

    # -- public API ----------------------------------------------------------

    def morton_key(self, point: R3Vector) -> int:
        '''Compute the Morton key for *point* without inserting it.'''
        point = vector_copy(point)
        for i in range(3):
            if point[i] < self._min_xyz[i] or point[i] > self._max_xyz[i]:
                raise Unrecoverable(
                    'Octree: point is outside the bounding box'
                )
        ix, iy, iz = self._point_to_cell(point)
        return self._interleave_bits(ix, iy, iz, self._depth)

    def insert(self, point: R3Vector) -> int:
        '''Insert *point* and return its Morton key.

        If a point with the same Morton key is already present, the stored
        coordinates are kept unchanged and the existing key is returned.
        '''
        key = self.morton_key(point)
        if key not in self._points:
            self._points[key] = vector_copy(point)
        return key

    def point(self, key: int) -> R3Vector:
        '''Return a copy of the point stored under Morton *key*.'''
        if key not in self._points:
            raise Unrecoverable(
                ''.join(['Octree: Morton key ', str(key), ' not found'])
            )
        return vector_copy(self._points[key])
