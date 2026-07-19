#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

from typing import TYPE_CHECKING

from ..common import Index
from ..errors import NeverImplement
from ..errors import Unrecoverable
from .partitioner import PartitionMethod, Partitioner

if TYPE_CHECKING:
    from .wire_mesh import WireMesh3D


class MeshFunctions:
    """Immutable mapping from function index to subsegment pairs.

    Each function is represented by a unique unordered pair of subsegments
    that share exactly one vertex.
    """

    def __init__(
        self,
        mesh: "WireMesh3D",
        method: PartitionMethod = PartitionMethod.OCTREE,
        max_n_parts: int = 1,
    ) -> None:
        point_pairs = mesh.subsegment_point_pairs

        vertex_subsegments: dict[Index, list[Index]] = {}
        for i, (a0, a1) in enumerate(point_pairs):
            subsegment_index = Index(i)
            vertex_subsegments.setdefault(a0, []).append(subsegment_index)
            if a1 != a0:
                vertex_subsegments.setdefault(a1, []).append(subsegment_index)

        pair_set: set[tuple[Index, Index]] = set()
        for incident_subsegments in vertex_subsegments.values():
            n_incident = len(incident_subsegments)
            for i in range(n_incident):
                subsegment_i = incident_subsegments[i]
                a0, a1 = point_pairs[int(subsegment_i)]
                for j in range(i + 1, n_incident):
                    subsegment_j = incident_subsegments[j]
                    b0, b1 = point_pairs[int(subsegment_j)]

                    shared_count = 0
                    if a0 == b0:
                        shared_count += 1
                    if a0 == b1:
                        shared_count += 1
                    if a1 == b0:
                        shared_count += 1
                    if a1 == b1:
                        shared_count += 1

                    if shared_count == 1:
                        pair_set.add((subsegment_i, subsegment_j))

        pairs = sorted(pair_set, key=lambda pair: (int(pair[0]), int(pair[1])))
        self._function_subsegment_pairs: tuple[tuple[Index, Index], ...] = tuple(pairs)

        self._partitioner = Partitioner(mesh, self, method, max_n_parts)

    @property
    def partitioner(self) -> Partitioner:
        '''Partitioner for this function set.'''
        return self._partitioner

    @partitioner.setter
    def partitioner(self, value) -> None:
        raise NeverImplement('MeshFunctions partitioner is immutable')

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
