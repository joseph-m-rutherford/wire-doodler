#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..common import Index, Real
from ..errors import NeverImplement, Recoverable, Unrecoverable
from ..r3 import Octree, R3Vector, vector_copy

if TYPE_CHECKING:
    from .function_supports import FunctionSupports
    from .wire_mesh import WireMesh3D


class PartitionMethod(str, Enum):
    """Strategy used by :class:`Partitioner` to assign supports to partitions."""
    OCTREE = 'octree'
    KAHIP = 'kahip'


def _build_adjacency_list(
    function_supports: "FunctionSupports",
    subseg_point_pairs: list[tuple[Index, Index]],
) -> list[list[int]]:
    """Build a vertex-sharing adjacency list over supports.

    Two supports are adjacent iff any vertex index (start or end of either
    subsegment) appears in both supports' combined vertex sets.

    Returns a list of length ``n_supports`` where entry *i* is the sorted
    list of support indices adjacent to support *i* (excluding *i* itself).
    """
    pairs = function_supports.support_subsegment_pairs
    n_supports = len(pairs)

    # Map vertex index -> list of support indices incident to that vertex.
    vertex_to_supports: dict[int, list[int]] = {}
    for support_idx, (sub_i, sub_j) in enumerate(pairs):
        a0, a1 = subseg_point_pairs[int(sub_i)]
        b0, b1 = subseg_point_pairs[int(sub_j)]
        for v in (int(a0), int(a1), int(b0), int(b1)):
            vertex_to_supports.setdefault(v, []).append(support_idx)

    adj: list[set[int]] = [set() for _ in range(n_supports)]
    for support_list in vertex_to_supports.values():
        for i in range(len(support_list)):
            for j in range(i + 1, len(support_list)):
                si, sj = support_list[i], support_list[j]
                if si != sj:  # skip self-loops from double-registered shared vertices
                    adj[si].add(sj)
                    adj[sj].add(si)

    return [sorted(neighbors) for neighbors in adj]


class Partitioner:
    """Partition the support index set of a :class:`~doodler.discretization.FunctionSupports`
    into at most *max_n_parts* non-overlapping, collectively exhaustive groups.

    The partition can be built using one of two strategies:

    * :attr:`PartitionMethod.OCTREE` — purely spatial.  Each support is
      represented by its shared vertex (the single vertex where the two
      subsegments of the support pair meet).  The finest (deepest) octree
      depth that produces at most *max_n_parts* occupied cells is selected.
      The actual partition count (≤ *max_n_parts*) depends on the geometry.

    * :attr:`PartitionMethod.KAHIP` — graph-based via ``kahip`` (KaFFPa).
      Raises :class:`~doodler.errors.Recoverable` if ``kahip`` is not
      installed (``pip install kahip``).  The actual partition count may be
      less than *max_n_parts* if the graph has fewer connected components.

    Parameters
    ----------
    mesh:
        The :class:`~doodler.discretization.WireMesh3D` the supports live on.
    function_supports:
        The :class:`~doodler.discretization.FunctionSupports` to partition.
    method:
        One of the :class:`PartitionMethod` strategies.
    max_n_parts:
        Upper bound on the number of partitions.  Must be >= 1.  The
        actual partition count returned by :attr:`partition_count` may
        be less than *max_n_parts* depending on the geometry.
    support_indices:
        Tuple of global support indices that this partition node covers.  When
        ``None`` (the default for the root node) all supports in
        *function_supports* are included.
    """

    def __init__(
        self,
        mesh: "WireMesh3D",
        function_supports: "FunctionSupports",
        method: PartitionMethod,
        max_n_parts: int,
        support_indices: tuple[Index, ...] | None = None,
    ) -> None:
        max_n_parts = int(max_n_parts)
        if max_n_parts < 1:
            raise Unrecoverable('Partitioner: max_n_parts must be >= 1')

        self._max_n_parts = max_n_parts
        try:
            self._method = PartitionMethod(method)
        except ValueError as e:
            raise Unrecoverable(f'Partitioner: unknown method {method!r}') from e
        # Keep references for child construction via refine().
        self._mesh = mesh
        self._function_supports = function_supports

        subseg_pairs = mesh.subsegment_point_pairs
        all_pairs = function_supports.support_subsegment_pairs

        if support_indices is None:
            support_indices = tuple(Index(i) for i in range(len(all_pairs)))

        self._support_indices: tuple[Index, ...] = support_indices
        n_local = len(support_indices)

        # Fast lookup: global support index -> position in _support_indices.
        self._local_index_map: dict[int, int] = {
            int(si): pos for pos, si in enumerate(support_indices)
        }

        # Restrict the subsegment-pair list to local supports only.
        local_pairs = [all_pairs[int(si)] for si in support_indices]

        if method == PartitionMethod.OCTREE:
            assignment = self._build_octree(mesh, local_pairs, subseg_pairs, max_n_parts)
        elif method == PartitionMethod.KAHIP:
            assignment = self._build_kahip(local_pairs, subseg_pairs, max_n_parts, n_local)
        else:
            raise Unrecoverable('Partitioner: unknown method')

        # Compact partition IDs to consecutive range 0..actual_count-1.
        # OCTREE already produces consecutive IDs; KAHIP may have gaps if some
        # requested partitions end up empty.
        used_ids = sorted(set(assignment))
        if used_ids and used_ids != list(range(len(used_ids))):
            id_map = {old: new for new, old in enumerate(used_ids)}
            assignment = [id_map[a] for a in assignment]
        # Always maintain at least one partition slot (for the empty-support case).
        actual_count = max(1, len(used_ids))

        self._partition_assignment: tuple[Index, ...] = tuple(
            Index(p) for p in assignment
        )

        # Build reverse mapping: partition_id -> sorted tuple of GLOBAL support indices.
        buckets: dict[int, list[int]] = {}
        for local_pos, part_id in enumerate(assignment):
            global_si = int(support_indices[local_pos])
            buckets.setdefault(int(part_id), []).append(global_si)
        self._partition_to_supports: tuple[tuple[Index, ...], ...] = tuple(
            tuple(Index(si) for si in sorted(buckets.get(pid, [])))
            for pid in range(actual_count)
        )

        # _n_parts is the ACTUAL partition count (≤ max_n_parts).
        self._n_parts = actual_count

        # Child partitioners — one slot per partition, filled lazily by refine().
        self._children: list[Partitioner | None] = [None] * actual_count

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
        max_n_parts: int,
    ) -> list[int]:
        """Partition supports spatially by their shared-vertex Morton key.

        Finds the finest (deepest) octree depth where the number of occupied
        cells does not exceed *max_n_parts*.  The actual partition count
        (≤ *max_n_parts*) is determined by the geometry.
        """
        n_supports = len(pairs)
        if n_supports == 0:
            return []

        # Collect shared-vertex coordinates.
        shared_coords: list[R3Vector] = []
        for sub_i, sub_j in pairs:
            sv_idx = Partitioner._shared_vertex_index(sub_i, sub_j, subseg_pairs)
            shared_coords.append(mesh.vertex_xyz(sv_idx))

        coords_arr = np.array(shared_coords)  # shape (n_supports, 3)
        bbox_min = vector_copy(np.min(coords_arr, axis=0))
        bbox_max = vector_copy(np.max(coords_arr, axis=0))

        # Pad to guarantee strict min < max and numerical headroom.
        reltol = mesh.reltol
        max_norm = Real(max(
            float(np.linalg.norm(bbox_min)),
            float(np.linalg.norm(bbox_max)),
            1.0,
        ))
        pad = Real(reltol * max_norm)
        bbox_min = vector_copy(bbox_min - pad)
        bbox_max = vector_copy(bbox_max + pad)
        abstol = Real(reltol * max_norm)

        # Build finest-resolution octree to get Morton keys for all shared vertices.
        octree = Octree(bbox_min, bbox_max, abstol)
        max_depth = octree.depth

        fine_keys = [octree.morton_key(c) for c in shared_coords]

        # Find the finest (deepest) depth where occupied cells <= max_n_parts.
        # Iterate from finest (d=max_depth, shift=0) to coarsest (d=0, shift=3*max_depth).
        # d=0 always yields 1 occupied cell, so the loop always terminates.
        chosen_shift: int = 3 * max_depth  # fallback: depth 0, 1 cell
        occupied_keys: set[int] = {k >> (3 * max_depth) for k in fine_keys}
        for d in range(max_depth, -1, -1):
            shift = 3 * (max_depth - d)
            coarse_keys = {k >> shift for k in fine_keys}
            if len(coarse_keys) <= max_n_parts:
                chosen_shift = shift
                occupied_keys = coarse_keys
                break

        # Assign deterministic partition IDs by sorting occupied coarse keys.
        coarse_keys_used = sorted(occupied_keys)
        key_to_part: dict[int, int] = {key: pid for pid, key in enumerate(coarse_keys_used)}

        return [key_to_part[k >> chosen_shift] for k in fine_keys]

    @staticmethod
    def _build_kahip(
        pairs: list[tuple[Index, Index]],
        subseg_pairs: list[tuple[Index, Index]],
        max_n_parts: int,
        n_supports: int,
    ) -> list[int]:
        """Partition supports using KaHIP (KaFFPa) graph partitioning."""
        try:
            import kahip  # type: ignore[import]
        except ImportError as exc:
            raise Recoverable(
                'Partitioner: kahip is not installed; install it with "pip install kahip"'
            ) from exc

        if n_supports == 0:
            return []

        # kaffpa requires n_parts >= 2; the trivial single-partition case is
        # handled here to avoid passing max_n_parts=1 into the C extension.
        if max_n_parts == 1:
            return [0] * n_supports

        class _Proxy:
            def __init__(self, p):
                self.support_subsegment_pairs = p

        adj = _build_adjacency_list(_Proxy(pairs), subseg_pairs)  # type: ignore[arg-type]

        # Build CSR adjacency arrays.
        xadj: list[int] = [0]
        adjncy: list[int] = []
        for neighbors in adj:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))

        vwgt = [1] * n_supports
        adjcwgt = [1] * len(adjncy)

        _edgecut, blocks = kahip.kaffpa(
            vwgt, xadj, adjcwgt, adjncy,
            max_n_parts,
            0.03,   # imbalance
            1,      # suppress_output
            0,      # seed
            0,      # mode = FAST
        )
        return [int(b) for b in blocks]

    # ------------------------------------------------------------------
    # Immutable properties
    # ------------------------------------------------------------------

    @property
    def support_indices(self) -> list[Index]:
        """Copy of the global support indices covered by this partition node."""
        return list(self._support_indices)

    @support_indices.setter
    def support_indices(self, value) -> None:
        raise NeverImplement('Partitioner support_indices is immutable')

    @property
    def children(self) -> tuple[Partitioner | None, ...]:
        """Tuple of child partitioners, one per partition (None until refined)."""
        return tuple(self._children)

    @children.setter
    def children(self, value) -> None:
        raise NeverImplement('Partitioner children is immutable')

    @property
    def method(self) -> PartitionMethod:
        """Partitioning strategy used."""
        return self._method

    @method.setter
    def method(self, value) -> None:
        raise NeverImplement('Partitioner method is immutable')

    @property
    def max_n_parts(self) -> int:
        """Upper bound on the number of partitions requested."""
        return self._max_n_parts

    @max_n_parts.setter
    def max_n_parts(self, value) -> None:
        raise NeverImplement('Partitioner max_n_parts is immutable')

    @property
    def partition_count(self) -> int:
        """Actual number of non-empty partitions produced (at most *max_n_parts*)."""
        return self._n_parts

    @partition_count.setter
    def partition_count(self, value) -> None:
        raise NeverImplement('Partitioner partition_count is immutable')

    @property
    def partition_assignment(self) -> list[Index]:
        """Copy of the per-support partition assignment array."""
        return list(self._partition_assignment)

    @partition_assignment.setter
    def partition_assignment(self, value) -> None:
        raise NeverImplement('Partitioner partition_assignment is immutable')

    # ------------------------------------------------------------------
    # Tree API
    # ------------------------------------------------------------------

    def child(self, partition_id: Index) -> Partitioner | None:
        """Return the child :class:`Partitioner` for *partition_id*, or ``None``
        if that partition has not yet been refined.

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
        return self._children[pid]

    def refine(
        self,
        partition_id: Index,
        method: PartitionMethod,
        max_n_parts: int,
    ) -> Partitioner:
        """Create and attach a child :class:`Partitioner` for *partition_id*.

        The child covers exactly the global support indices currently assigned
        to *partition_id* at this node and partitions them into at most
        *max_n_parts* sub-groups using *method*.  Any previously attached child
        for *partition_id* is replaced.

        Parameters
        ----------
        partition_id:
            Zero-based partition identifier in ``[0, partition_count)``.
        method:
            Partitioning strategy for the child node.
        max_n_parts:
            Upper bound on sub-partitions in the child node.  Must be >= 1.

        Returns
        -------
        Partitioner
            The newly created child partitioner.

        Raises
        ------
        Unrecoverable
            If *partition_id* is out of range or *max_n_parts* < 1.
        """
        pid = int(partition_id)
        if pid < 0 or pid >= self._n_parts:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: partition_id ', str(pid),
                    ' is out of range for ', str(self._n_parts), ' partitions',
                ])
            )
        child_indices = self._partition_to_supports[pid]
        child = Partitioner(
            self._mesh,
            self._function_supports,
            method,
            max_n_parts,
            child_indices,
        )
        self._children[pid] = child
        return child

    def node_at_path(self, path: list[int]) -> Partitioner:
        """Follow *path* from this node and return the reached :class:`Partitioner`.

        *path* is a sequence of partition IDs, one per level:  ``path[0]``
        selects a child of this node, ``path[1]`` selects a grandchild, and so
        on.  An empty *path* returns ``self``.

        Raises
        ------
        Unrecoverable
            If any step in *path* leads to an unrefined (``None``) child or an
            out-of-range partition ID.
        """
        node: Partitioner = self
        for depth, step in enumerate(path):
            c = node.child(Index(step))
            if c is None:
                raise Unrecoverable(
                    ''.join([
                        'Partitioner: path has no child at depth ', str(depth),
                        ', partition_id ', str(step),
                    ])
                )
            node = c
        return node

    def supports_at_path(self, path: list[int]) -> list[Index]:
        """Return the global support indices held at the node reached by *path*.

        Equivalent to ``node_at_path(path).support_indices`` but more
        convenient for bulk lookup across levels.

        An empty *path* returns this node's :attr:`support_indices`.

        Raises
        ------
        Unrecoverable
            If the path is invalid (see :meth:`node_at_path`).
        """
        return self.node_at_path(path).support_indices

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def supports_in_partition(self, partition_id: Index) -> list[Index]:
        """Return a sorted list of support indices belonging to *partition_id*.

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
        return list(self._partition_to_supports[pid])

    def partition_of_support(self, support_index: Index) -> Index:
        """Return the partition id that contains *support_index*.

        Parameters
        ----------
        support_index:
            Zero-based support index in ``[0, len(support_subsegment_pairs))``.

        Raises
        ------
        Unrecoverable
            If *support_index* is out of range.
        """
        si = int(support_index)
        if si not in self._local_index_map:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: support_index ', str(si),
                    ' is not in this partition node (',
                    str(len(self._support_indices)), ' local supports',
                    ')',
                ])
            )
        return self._partition_assignment[self._local_index_map[si]]
