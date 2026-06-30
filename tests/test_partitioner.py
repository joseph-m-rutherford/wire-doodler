#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import builtins
import unittest.mock as mock

import numpy as np
import pytest

from doodler import Index, Real, WireMesh3D
from doodler.errors import NeverImplement, Recoverable, Unrecoverable
from doodler.operators.partitioner import PartitionMethod, Partitioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt(x, y, z):
    return np.array([Real(x), Real(y), Real(z)])


_TOL = Real(0.01)


def _simple_mesh(h=Real(1.0)):
    """A single two-segment polyline: (0,0,0)->(1,0,0)->(2,0,0).
    With h=1.0 each polyline segment -> 1 subsegment = 2 subsegments total.
    The interior vertex (1,0,0) is shared: 1 function pair.
    """
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    return WireMesh3D({'p': poly}, h, _TOL)


def _two_segment_mesh():
    """Two separated polylines — no shared vertices, no function pairs."""
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(5, 0, 0), _pt(6, 0, 0)]
    return WireMesh3D({'a': poly_a, 'b': poly_b}, Real(1.0), _TOL)


# ---------------------------------------------------------------------------
# Octree: n_parts = 1 (trivial)
# ---------------------------------------------------------------------------

def test_partitioner_octree_n_parts_1_succeeds():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1


def test_partitioner_octree_n_parts_1_all_functions_in_partition_0():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    fns = p.functions_in_partition(Index(0))
    expected = list(range(len(mesh.mesh_functions.function_subsegment_pairs)))
    assert sorted(int(f) for f in fns) == expected


# ---------------------------------------------------------------------------
# Octree: coverage + invertibility
# ---------------------------------------------------------------------------

def test_partitioner_octree_coverage():
    """Union of all partitions covers every function index exactly once."""
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))


def test_partitioner_octree_invertibility():
    """partition_of_function(i) == j  <=>  i in functions_in_partition(j)."""
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


def test_partitioner_partition_of_function_value_in_range():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert 0 <= pid < p.partition_count


# ---------------------------------------------------------------------------
# Octree: achievable n_parts > 1
# ---------------------------------------------------------------------------

def test_partitioner_octree_two_parts_spatially_separated():
    """Two function pairs with shared vertices far apart -> 2 occupied cells achievable."""
    # polyline A: (0,0,0)->(1,0,0)->(2,0,0)    shared vertex at (1,0,0)
    # polyline B: (100,0,0)->(110,0,0)->(120,0,0)  shared vertex at (110,0,0)
    # Step of 10 at distance ~110 gives relative separation 10/110 ≈ 9%, well above reltol=0.01.
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    poly_b = [_pt(100, 0, 0), _pt(110, 0, 0), _pt(120, 0, 0)]
    # h=15.0: each length-1 segment -> 1 subsegment; each length-10 segment -> 1
    # subsegment.  Both polylines contribute exactly 1 function pair.
    mesh = WireMesh3D({'a': poly_a, 'b': poly_b}, Real(15.0), _TOL, n_parts=2)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 2
    # Each partition should contain exactly 1 function.
    counts = [len(p.functions_in_partition(Index(pid))) for pid in range(2)]
    assert sorted(counts) == [1, 1]


# ---------------------------------------------------------------------------
# Octree: unachievable n_parts
# ---------------------------------------------------------------------------

def test_partitioner_octree_unachievable_n_parts_raises():
    """Requesting more partitions than spatially distinct functions raises Unrecoverable."""
    # Only 1 function pair exists; asking for 2 partitions is impossible.
    mesh_no_part = WireMesh3D(
        {'p': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]},
        Real(1.0),
        _TOL,
    )
    with pytest.raises(Unrecoverable):
        Partitioner(
            mesh_no_part,
            mesh_no_part.mesh_functions,
            PartitionMethod.OCTREE,
            n_parts=3,
        )


# ---------------------------------------------------------------------------
# Octree: empty function set
# ---------------------------------------------------------------------------

def test_partitioner_octree_empty_functions_n_parts_1_succeeds():
    """No function pairs -> n_parts=1 returns one empty partition (valid, no error)."""
    mesh = _two_segment_mesh()
    assert len(mesh.mesh_functions.function_subsegment_pairs) == 0
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert p.functions_in_partition(Index(0)) == []


# ---------------------------------------------------------------------------
# Invalid n_parts
# ---------------------------------------------------------------------------

def test_partitioner_n_parts_zero_raises():
    mesh = _simple_mesh()
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=0)


def test_partitioner_n_parts_negative_raises():
    mesh = _simple_mesh()
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=-1)


# ---------------------------------------------------------------------------
# Out-of-range query errors
# ---------------------------------------------------------------------------

def test_partitioner_functions_in_partition_out_of_range_raises():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    with pytest.raises(Unrecoverable):
        p.functions_in_partition(Index(p.partition_count))


def test_partitioner_partition_of_function_out_of_range_raises():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    with pytest.raises(Unrecoverable):
        p.partition_of_function(Index(n_fns))


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_partitioner_method_immutable():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.method = PartitionMethod.OCTREE


def test_partitioner_n_parts_immutable():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.n_parts = 2


def test_partitioner_partition_count_immutable():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.partition_count = 2


def test_partitioner_partition_assignment_immutable():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.partition_assignment = []


def test_mesh_functions_partitioner_immutable():
    mesh = _simple_mesh()
    mf = mesh.mesh_functions
    with pytest.raises(NeverImplement):
        mf.partitioner = None


# ---------------------------------------------------------------------------
# functions_in_partition returns a copy
# ---------------------------------------------------------------------------

def test_partitioner_functions_in_partition_returns_copy():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    fns = p.functions_in_partition(Index(0))
    original_len = len(fns)
    fns.clear()
    assert len(p.functions_in_partition(Index(0))) == original_len


def test_partitioner_partition_assignment_returns_copy():
    mesh = _simple_mesh()
    p = mesh.mesh_functions.partitioner
    assignment = p.partition_assignment
    original_len = len(assignment)
    assignment.clear()
    assert len(p.partition_assignment) == original_len


# ---------------------------------------------------------------------------
# Accessibility via WireMesh3D
# ---------------------------------------------------------------------------

def test_partitioner_accessible_via_mesh():
    mesh = _simple_mesh()
    partitioner = mesh.mesh_functions.partitioner
    assert isinstance(partitioner, Partitioner)


def test_wire_mesh_method_property():
    mesh = _simple_mesh()
    assert mesh.method == PartitionMethod.OCTREE


def test_wire_mesh_n_parts_property():
    mesh = _simple_mesh()
    assert mesh.n_parts == 1


def test_wire_mesh_method_immutable():
    mesh = _simple_mesh()
    with pytest.raises(NeverImplement):
        mesh.method = PartitionMethod.OCTREE


def test_wire_mesh_n_parts_immutable():
    mesh = _simple_mesh()
    with pytest.raises(NeverImplement):
        mesh.n_parts = 2


# ---------------------------------------------------------------------------
# WireMesh3D n_parts kwarg validation
# ---------------------------------------------------------------------------

def test_wire_mesh_n_parts_zero_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D(
            {'p': [_pt(0, 0, 0), _pt(1, 0, 0)]},
            Real(1.0),
            _TOL,
            n_parts=0,
        )


# ---------------------------------------------------------------------------
# kahip: Recoverable when not installed
# ---------------------------------------------------------------------------

def test_partitioner_kahip_not_installed_raises_recoverable():
    mesh = _simple_mesh()
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == 'kahip':
            raise ImportError('kahip not installed')
        return real_import(name, *args, **kwargs)

    with mock.patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(Recoverable):
            Partitioner(
                mesh,
                mesh.mesh_functions,
                PartitionMethod.KAHIP,
                n_parts=1,
            )
