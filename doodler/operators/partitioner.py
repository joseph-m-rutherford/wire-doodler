#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from __future__ import annotations

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
    KAHIP = 'kahip'


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
                if fi != fj:  # skip self-loops from double-registered shared vertices
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

    * :attr:`PartitionMethod.KAHIP` — graph-based via ``kahip`` (KaFFPa).
      Raises :class:`~doodler.errors.Recoverable` if ``kahip`` is not
      installed (``pip install kahip``).  When *n_parts* exceeds the number
      of connected components the result may contain fewer than *n_parts*
      non-empty partitions (documented behaviour, not an error).

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
    local_function_indices:
        Tuple of global function indices that this partition node covers.  When
        ``None`` (the default for the root node) all functions in
        *mesh_functions* are included.
    """

    def __init__(
        self,
        mesh: "WireMesh3D",
        mesh_functions: "MeshFunctions",
        method: PartitionMethod,
        n_parts: int,
        local_function_indices: tuple[Index, ...] | None = None,
    ) -> None:
        n_parts = int(n_parts)
        if n_parts < 1:
            raise Unrecoverable('Partitioner: n_parts must be >= 1')

        self._method = PartitionMethod(method)
        self._n_parts = n_parts
        # Keep references for child construction via refine().
        self._mesh = mesh
        self._mesh_functions = mesh_functions

        subseg_pairs = mesh.subsegment_point_pairs
        all_pairs = mesh_functions.function_subsegment_pairs

        if local_function_indices is None:
            local_function_indices = tuple(Index(i) for i in range(len(all_pairs)))

        self._local_function_indices: tuple[Index, ...] = local_function_indices
        n_local = len(local_function_indices)

        # Fast lookup: global function index -> position in _local_function_indices.
        self._local_index_map: dict[int, int] = {
            int(fi): pos for pos, fi in enumerate(local_function_indices)
        }

        # Restrict the subsegment-pair list to local functions only.
        local_pairs = [all_pairs[int(fi)] for fi in local_function_indices]

        if method == PartitionMethod.OCTREE:
            assignment = self._build_octree(mesh, local_pairs, subseg_pairs, n_parts)
        elif method == PartitionMethod.KAHIP:
            assignment = self._build_kahip(local_pairs, subseg_pairs, n_parts, n_local)
        else:
            raise Unrecoverable('Partitioner: unknown method')

        self._partition_assignment: tuple[Index, ...] = tuple(
            Index(p) for p in assignment
        )

        # Build reverse mapping: partition_id -> sorted tuple of GLOBAL function indices.
        buckets: dict[int, list[int]] = {}
        for local_pos, part_id in enumerate(assignment):
            global_fi = int(local_function_indices[local_pos])
            buckets.setdefault(int(part_id), []).append(global_fi)
        self._partition_to_functions: tuple[tuple[Index, ...], ...] = tuple(
            tuple(Index(fi) for fi in sorted(buckets.get(pid, [])))
            for pid in range(n_parts)
        )

        # Child partitioners — one slot per partition, filled lazily by refine().
        self._children: list[Partitioner | None] = [None] * n_parts

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
    def _build_kahip(
        pairs: list[tuple[Index, Index]],
        subseg_pairs: list[tuple[Index, Index]],
        n_parts: int,
        n_functions: int,
    ) -> list[int]:
        """Partition functions using KaHIP (KaFFPa) graph partitioning."""
        try:
            import kahip  # type: ignore[import]
        except ImportError as exc:
            raise Recoverable(
                'Partitioner: kahip is not installed; install it with "pip install kahip"'
            ) from exc

        if n_functions == 0:
            return []

        # kaffpa requires n_parts >= 2; the trivial single-partition case is
        # handled here to avoid passing n_parts=1 into the C extension.
        if n_parts == 1:
            return [0] * n_functions

        class _Proxy:
            def __init__(self, p):
                self.function_subsegment_pairs = p

        adj = _build_adjacency_list(_Proxy(pairs), subseg_pairs)  # type: ignore[arg-type]

        # Build CSR adjacency arrays.
        xadj: list[int] = [0]
        adjncy: list[int] = []
        for neighbors in adj:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))

        vwgt = [1] * n_functions
        adjcwgt = [1] * len(adjncy)

        _edgecut, blocks = kahip.kaffpa(
            vwgt, xadj, adjcwgt, adjncy,
            n_parts,
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
    def local_function_indices(self) -> list[Index]:
        """Copy of the global function indices covered by this partition node."""
        return list(self._local_function_indices)

    @local_function_indices.setter
    def local_function_indices(self, value) -> None:
        raise NeverImplement('Partitioner local_function_indices is immutable')

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
        n_parts: int,
    ) -> Partitioner:
        """Create and attach a child :class:`Partitioner` for *partition_id*.

        The child covers exactly the global function indices currently assigned
        to *partition_id* at this node and partitions them into *n_parts*
        sub-groups using *method*.  Any previously attached child for
        *partition_id* is replaced.

        Parameters
        ----------
        partition_id:
            Zero-based partition identifier in ``[0, partition_count)``.
        method:
            Partitioning strategy for the child node.
        n_parts:
            Number of sub-partitions in the child node.  Must be >= 1.

        Returns
        -------
        Partitioner
            The newly created child partitioner.

        Raises
        ------
        Unrecoverable
            If *partition_id* is out of range or *n_parts* < 1.
        """
        pid = int(partition_id)
        if pid < 0 or pid >= self._n_parts:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: partition_id ', str(pid),
                    ' is out of range for ', str(self._n_parts), ' partitions',
                ])
            )
        child_indices = self._partition_to_functions[pid]
        child = Partitioner(
            self._mesh,
            self._mesh_functions,
            method,
            n_parts,
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

    def functions_at_path(self, path: list[int]) -> list[Index]:
        """Return the global function indices held at the node reached by *path*.

        Equivalent to ``node_at_path(path).local_function_indices`` but more
        convenient for bulk lookup across levels.

        An empty *path* returns this node's :attr:`local_function_indices`.

        Raises
        ------
        Unrecoverable
            If the path is invalid (see :meth:`node_at_path`).
        """
        return self.node_at_path(path).local_function_indices

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
        if fi not in self._local_index_map:
            raise Unrecoverable(
                ''.join([
                    'Partitioner: function_index ', str(fi),
                    ' is not in this partition node (',
                    str(len(self._local_function_indices)), ' local functions)',
                ])
            )
        return self._partition_assignment[self._local_index_map[fi]]
