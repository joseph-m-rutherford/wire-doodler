#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..common import Index, Real
from ..errors import NeverImplement, Recoverable, Unrecoverable
from ..r3 import Octree, R3Vector, r3vector_copy

if TYPE_CHECKING:
    from .mesh_functions import MeshFunctions
    from .wire_mesh import WireMesh3D


class PartitionMethod(str, Enum):
    """Strategy used by :class:`Partitioner` to assign functions to partitions."""
    OCTREE = 'octree'
    PYMETIS = 'pymetis'
    SCOTCHPY64 = 'scotchpy64'


def _build_adjacency_list(
    mesh_functions: "MeshFunctions",
    subseg_point_pairs: list[tuple[Index, Index]],
) -> list[list[int]]:
    """Build a vertex-sharing adjacency list over functions.

    Two functions are adjacent iff any vertex index (start or end of either
    subsegment) appears in both functions' combined vertex sets.

    Returns a list of length ``n_functions`` where entry *i* is the sorted
    list of function indices adjacent to function *i* (excluding *i* itself).
    """
    pairs = mesh_functions.function_subsegment_pairs
    n_functions = len(pairs)

    # Map vertex index -> list of function indices incident to that vertex.
    vertex_to_functions: dict[int, list[int]] = {}
    for fn_idx, (sub_i, sub_j) in enumerate(pairs):
        a0, a1 = subseg_point_pairs[int(sub_i)]
        b0, b1 = subseg_point_pairs[int(sub_j)]
        for v in (int(a0), int(a1), int(b0), int(b1)):
            vertex_to_functions.setdefault(v, []).append(fn_idx)

    adj: list[set[int]] = [set() for _ in range(n_functions)]
    for fn_list in vertex_to_functions.values():
        for i in range(len(fn_list)):
            for j in range(i + 1, len(fn_list)):
                fi, fj = fn_list[i], fn_list[j]
                adj[fi].add(fj)
                adj[fj].add(fi)

    return [sorted(neighbors) for neighbors in adj]


class Partitioner:
    """Partition the function index set of a :class:`~doodler.operators.MeshFunctions`
    into *n_parts* non-overlapping, collectively exhaustive groups.

    The partition can be built using one of three strategies:

    * :attr:`PartitionMethod.OCTREE` — purely spatial.  Each function is
      represented by its shared vertex (the single vertex where the two
      subsegments of the function pair meet).  The finest octree depth that
      produces exactly *n_parts* occupied cells is selected; if no such depth
      exists, :class:`~doodler.errors.Unrecoverable` is raised.

    * :attr:`PartitionMethod.PYMETIS` — graph-based via ``pymetis``.  Raises
      :class:`~doodler.errors.Recoverable` if ``pymetis`` is not installed.
      When *n_parts* exceeds the number of connected components the result
      may contain fewer than *n_parts* non-empty partitions (documented
      behaviour, not an error).

    * :attr:`PartitionMethod.SCOTCHPY64` — graph-based via ``scotchpy64``.
      Raises :class:`~doodler.errors.Recoverable` if ``scotchpy64`` is not
      installed.  Same caveat as pymetis for disconnected graphs.

    Parameters
    ----------
    mesh:
        The :class:`~doodler.operators.WireMesh3D` the functions live on.
    mesh_functions:
        The :class:`~doodler.operators.MeshFunctions` to partition.
    method:
        One of the :class:`PartitionMethod` strategies.
    n_parts:
        Desired number of partitions.  Must be >= 1.
    """

    def __init__(
        self,
        mesh: "WireMesh3D",
        mesh_functions: "MeshFunctions",
        method: PartitionMethod,
        n_parts: int,
    ) -> None:
        n_parts = int(n_parts)
        if n_parts < 1:
            raise Unrecoverable('Partitioner: n_parts must be >= 1')

        self._method = PartitionMethod(method)
        self._n_parts = n_parts

        subseg_pairs = mesh.subsegment_point_pairs
        pairs = mesh_functions.function_subsegment_pairs
        n_functions = len(pairs)

        if method == PartitionMethod.OCTREE:
            assignment = self._build_octree(mesh, pairs, subseg_pairs, n_parts)
        elif method == PartitionMethod.PYMETIS:
            assignment = self._build_pymetis(pairs, subseg_pairs, n_parts, n_functions)
        elif method == PartitionMethod.SCOTCHPY64:
            assignment = self._build_scotchpy64(pairs, subseg_pairs, n_parts, n_functions)
        else:
            raise Unrecoverable('Partitioner: unknown method')

        self._partition_assignment: tuple[Index, ...] = tuple(
            Index(p) for p in assignment
        )

        # Build reverse mapping: partition_id -> sorted tuple of function indices.
        buckets: dict[int, list[int]] = {}
        for fn_idx, part_id in enumerate(assignment):
            buckets.setdefault(int(part_id), []).append(fn_idx)
        self._partition_to_functions: tuple[tuple[Index, ...], ...] = tuple(
            tuple(Index(fi) for fi in sorted(buckets.get(pid, [])))
            for pid in range(n_parts)
        )

    # ------------------------------------------------------------------
    # Build strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_vertex_index(
        sub_i: Index,
        sub_j: Index,
        subseg_pairs: list[tuple[Index, Index]],
    ) -> Index:
        """Return the vertex index shared by two subsegments."""
        a0, a1 = subseg_pairs[int(sub_i)]
        b0, b1 = subseg_pairs[int(sub_j)]
        if a0 == b0 or a0 == b1:
            return a0
        return a1

    @staticmethod
    def _build_octree(
        mesh: "WireMesh3D",
        pairs: list[tuple[Index, Index]],
        subseg_pairs: list[tuple[Index, Index]],
        n_parts: int,
    ) -> list[int]:
        """Partition functions spatially by their shared-vertex Morton key."""
        n_functions = len(pairs)
        if n_functions == 0:
            return []

        # Collect shared-vertex coordinates.
        shared_coords: list[R3Vector] = []
        for sub_i, sub_j in pairs:
            sv_idx = Partitioner._shared_vertex_index(sub_i, sub_j, subseg_pairs)
            shared_coords.append(mesh.vertex_xyz(sv_idx))

        coords_arr = np.array(shared_coords)  # shape (n_functions, 3)
        bbox_min = r3vector_copy(np.min(coords_arr, axis=0))
        bbox_max = r3vector_copy(np.max(coords_arr, axis=0))

        # Pad to guarantee strict min < max and numerical headroom.
        reltol = mesh.reltol
        max_norm = Real(max(
            float(np.linalg.norm(bbox_min)),
            float(np.linalg.norm(bbox_max)),
            1.0,
        ))
        pad = Real(reltol * max_norm)
        bbox_min = r3vector_copy(bbox_min - pad)
        bbox_max = r3vector_copy(bbox_max + pad)
        abstol = Real(reltol * max_norm)

        # Build finest-resolution octree to get Morton keys for all shared vertices.
        octree = Octree(bbox_min, bbox_max, abstol)
        max_depth = octree.depth

        fine_keys = [octree.morton_key(c) for c in shared_coords]

        # Find the shallowest depth where the number of occupied cells == n_parts.
        # Coarsen by right-shifting fine_key by 3*(max_depth - d) bits.
        chosen_shift = None
        for d in range(max_depth + 1):
            shift = 3 * (max_depth - d)
            coarse_keys = {k >> shift for k in fine_keys}
            if len(coarse_keys) == n_parts:
                chosen_shift = shift
                break

        if chosen_shift is None:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: octree partitioning cannot produce exactly ',
                    str(n_parts),
                    ' partitions for this geometry (achievable non-empty cell counts: ',
                    str(sorted({
                        len({k >> (3 * (max_depth - d)) for k in fine_keys})
                        for d in range(max_depth + 1)
                    })),
                    ')',
                ])
            )

        # Assign deterministic partition IDs by sorting occupied coarse keys.
        coarse_keys_used = sorted({k >> chosen_shift for k in fine_keys})
        key_to_part: dict[int, int] = {key: pid for pid, key in enumerate(coarse_keys_used)}

        return [key_to_part[k >> chosen_shift] for k in fine_keys]

    @staticmethod
    def _build_pymetis(
        pairs: list[tuple[Index, Index]],
        subseg_pairs: list[tuple[Index, Index]],
        n_parts: int,
        n_functions: int,
    ) -> list[int]:
        """Partition functions using pymetis graph partitioning."""
        try:
            import pymetis  # type: ignore[import]
        except ImportError as exc:
            raise Recoverable(
                'Partitioner: pymetis is not installed; install it with "pip install pymetis"'
            ) from exc

        # Build a temporary MeshFunctions-like object to reuse _build_adjacency_list.
        class _Proxy:
            def __init__(self, p):
                self.function_subsegment_pairs = p

        adj = _build_adjacency_list(_Proxy(pairs), subseg_pairs)  # type: ignore[arg-type]

        if n_functions == 0:
            return []

        _, partition = pymetis.part_graph(n_parts, adjacency=adj)
        return list(partition)

    @staticmethod
    def _build_scotchpy64(
        pairs: list[tuple[Index, Index]],
        subseg_pairs: list[tuple[Index, Index]],
        n_parts: int,
        n_functions: int,
    ) -> list[int]:
        """Partition functions using scotchpy64 graph partitioning."""
        try:
            import scotchpy64  # type: ignore[import]
        except ImportError as exc:
            raise Recoverable(
                'Partitioner: scotchpy64 is not installed; install it with "pip install scotchpy64"'
            ) from exc

        class _Proxy:
            def __init__(self, p):
                self.function_subsegment_pairs = p

        adj = _build_adjacency_list(_Proxy(pairs), subseg_pairs)  # type: ignore[arg-type]

        if n_functions == 0:
            return []

        # Build CSR adjacency for scotchpy64.
        # scotchpy64.Graph expects (xadj, adjncy) in CSR format.
        xadj = [0]
        adjncy = []
        for neighbors in adj:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))

        graph = scotchpy64.Graph()
        graph.build(0, n_functions, xadj, adjncy)
        strat = scotchpy64.Strat()
        strat.graphMapBuild(scotchpy64.STRATDEFAULT, n_parts, n_parts, 0.01)
        arch = scotchpy64.Arch()
        arch.archCmplt(n_parts)
        mapping = scotchpy64.Mapping(graph, arch)
        graph.mapCompute(mapping, strat)
        partition = mapping.toTab()
        return [int(p) for p in partition]

    # ------------------------------------------------------------------
    # Immutable properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> PartitionMethod:
        """Partitioning strategy used."""
        return self._method

    @method.setter
    def method(self, value) -> None:
        raise NeverImplement('Partitioner method is immutable')

    @property
    def n_parts(self) -> int:
        """Number of partitions requested."""
        return self._n_parts

    @n_parts.setter
    def n_parts(self, value) -> None:
        raise NeverImplement('Partitioner n_parts is immutable')

    @property
    def partition_count(self) -> int:
        """Number of partitions (equal to *n_parts*)."""
        return self._n_parts

    @partition_count.setter
    def partition_count(self, value) -> None:
        raise NeverImplement('Partitioner partition_count is immutable')

    @property
    def partition_assignment(self) -> list[Index]:
        """Copy of the per-function partition assignment array."""
        return list(self._partition_assignment)

    @partition_assignment.setter
    def partition_assignment(self, value) -> None:
        raise NeverImplement('Partitioner partition_assignment is immutable')

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def functions_in_partition(self, partition_id: Index) -> list[Index]:
        """Return a sorted list of function indices belonging to *partition_id*.

        Parameters
        ----------
        partition_id:
            Zero-based partition identifier in ``[0, partition_count)``.

        Raises
        ------
        Unrecoverable
            If *partition_id* is out of range.
        """
        pid = int(partition_id)
        if pid < 0 or pid >= self._n_parts:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: partition_id ', str(pid),
                    ' is out of range for ', str(self._n_parts), ' partitions',
                ])
            )
        return list(self._partition_to_functions[pid])

    def partition_of_function(self, function_index: Index) -> Index:
        """Return the partition id that contains *function_index*.

        Parameters
        ----------
        function_index:
            Zero-based function index in ``[0, len(function_subsegment_pairs))``.

        Raises
        ------
        Unrecoverable
            If *function_index* is out of range.
        """
        fi = int(function_index)
        if fi < 0 or fi >= len(self._partition_assignment):
            raise Unrecoverable(
                ''.join([
                    'Partitioner: function_index ', str(fi),
                    ' is out of range for ', str(len(self._partition_assignment)),
                    ' functions',
                ])
            )
        return self._partition_assignment[fi]
