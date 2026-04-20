#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from typing import TYPE_CHECKING

from ..common import Index
from ..errors import NeverImplement
from ..errors import Unrecoverable
from ..r3 import R3Vector, r3vector_equality

if TYPE_CHECKING:
    from .wire_mesh import WireMesh3D


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
