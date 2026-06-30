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


def _simple_mesh(h):
    """A single two-segment polyline: (0,0,0)->(1,0,0)->(2,0,0).
    The interior vertex (1,0,0) is shared; the number of function pairs
    depends on *h*.
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
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1


def test_partitioner_octree_n_parts_1_all_functions_in_partition_0():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    fns = p.functions_in_partition(Index(0))
    expected = list(range(len(mesh.mesh_functions.function_subsegment_pairs)))
    assert sorted(int(f) for f in fns) == expected


# ---------------------------------------------------------------------------
# Octree: coverage + invertibility
# ---------------------------------------------------------------------------

def test_partitioner_octree_coverage():
    """Union of all partitions covers every function index exactly once."""
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))


def test_partitioner_octree_invertibility():
    """partition_of_function(i) == j  <=>  i in functions_in_partition(j)."""
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


def test_partitioner_partition_of_function_value_in_range():
    mesh = _simple_mesh(Real(1.0))
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
    mesh = _simple_mesh(Real(1.0))
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=0)


def test_partitioner_n_parts_negative_raises():
    mesh = _simple_mesh(Real(1.0))
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=-1)


# ---------------------------------------------------------------------------
# Out-of-range query errors
# ---------------------------------------------------------------------------

def test_partitioner_functions_in_partition_out_of_range_raises():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    with pytest.raises(Unrecoverable):
        p.functions_in_partition(Index(p.partition_count))


def test_partitioner_partition_of_function_out_of_range_raises():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    with pytest.raises(Unrecoverable):
        p.partition_of_function(Index(n_fns))


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_partitioner_method_immutable():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.method = PartitionMethod.OCTREE


def test_partitioner_n_parts_immutable():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.n_parts = 2


def test_partitioner_partition_count_immutable():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.partition_count = 2


def test_partitioner_partition_assignment_immutable():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    with pytest.raises(NeverImplement):
        p.partition_assignment = []


def test_mesh_functions_partitioner_immutable():
    mesh = _simple_mesh(Real(1.0))
    mf = mesh.mesh_functions
    with pytest.raises(NeverImplement):
        mf.partitioner = None


# ---------------------------------------------------------------------------
# functions_in_partition returns a copy
# ---------------------------------------------------------------------------

def test_partitioner_functions_in_partition_returns_copy():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    fns = p.functions_in_partition(Index(0))
    original_len = len(fns)
    fns.clear()
    assert len(p.functions_in_partition(Index(0))) == original_len


def test_partitioner_partition_assignment_returns_copy():
    mesh = _simple_mesh(Real(1.0))
    p = mesh.mesh_functions.partitioner
    assignment = p.partition_assignment
    original_len = len(assignment)
    assignment.clear()
    assert len(p.partition_assignment) == original_len


# ---------------------------------------------------------------------------
# Accessibility via WireMesh3D
# ---------------------------------------------------------------------------

def test_partitioner_accessible_via_mesh():
    mesh = _simple_mesh(Real(1.0))
    partitioner = mesh.mesh_functions.partitioner
    assert isinstance(partitioner, Partitioner)


def test_wire_mesh_method_property():
    mesh = _simple_mesh(Real(1.0))
    assert mesh.method == PartitionMethod.OCTREE


def test_wire_mesh_n_parts_property():
    mesh = _simple_mesh(Real(1.0))
    assert mesh.n_parts == 1


def test_wire_mesh_method_immutable():
    mesh = _simple_mesh(Real(1.0))
    with pytest.raises(NeverImplement):
        mesh.method = PartitionMethod.OCTREE


def test_wire_mesh_n_parts_immutable():
    mesh = _simple_mesh(Real(1.0))
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
    mesh = _simple_mesh(Real(1.0))
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


# ===========================================================================
# Shape-based geometry fixtures
# ===========================================================================

def _two_parallel_lines_mesh(h, n_parts, method):
    """Two parallel 3-point polylines in the XY plane:
      Line A (y=0): (0,0,0)->(1,0,0)->(2,0,0)
      Line B (y=1): (0,1,0)->(1,1,0)->(2,1,0)

    The two lines are separated in Y by 1 unit (>> reltol=0.01), so all
    functions from line A and line B always land in different octree cells.
    The number of function pairs per line is ceil(2/h) - 1.
    """
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    poly_b = [_pt(0, 1, 0), _pt(1, 1, 0), _pt(2, 1, 0)]
    return WireMesh3D({'a': poly_a, 'b': poly_b}, h, _TOL,
                      method=method, n_parts=n_parts)


def _rectangle_mesh(h, n_parts, method):
    """Unit square in the XY plane built from 4 single-segment polylines:
      bottom: (0,0,0)->(1,0,0)
      right:  (1,0,0)->(1,1,0)
      top:    (1,1,0)->(0,1,0)
      left:   (0,1,0)->(0,0,0)

    With h >= 1.0, only corner shared vertices are produced (4 functions).
    With h = 0.5, edge midpoints are also produced (8 functions total).
    """
    return WireMesh3D({
        'bottom': [_pt(0, 0, 0), _pt(1, 0, 0)],
        'right':  [_pt(1, 0, 0), _pt(1, 1, 0)],
        'top':    [_pt(1, 1, 0), _pt(0, 1, 0)],
        'left':   [_pt(0, 1, 0), _pt(0, 0, 0)],
    }, h, _TOL, method=method, n_parts=n_parts)


def _shared_vertex_xyz(mesh, fn_idx):
    """Return the 3-D position of the shared vertex for *fn_idx*."""
    pairs = mesh.mesh_functions.function_subsegment_pairs
    subseg_pairs = mesh.subsegment_point_pairs
    sub_i, sub_j = pairs[int(fn_idx)]
    a0, a1 = subseg_pairs[int(sub_i)]
    b0, b1 = subseg_pairs[int(sub_j)]
    sv_idx = a0 if (a0 == b0 or a0 == b1) else a1
    return mesh.vertex_xyz(sv_idx)


def _partition_endpoints_bbox(mesh, partitioner, pid):
    """Return (min_xyz, max_xyz) of all subsegment endpoints in partition *pid*."""
    fns = partitioner.functions_in_partition(Index(pid))
    pairs = mesh.mesh_functions.function_subsegment_pairs
    coords = []
    for fn_idx in fns:
        sub_i, sub_j = pairs[int(fn_idx)]
        for s in (sub_i, sub_j):
            start, end = mesh.subsegment_endpoints(s)
            coords.extend([start, end])
    arr = np.array(coords)
    return arr.min(axis=0), arr.max(axis=0)


# ===========================================================================
# Two parallel line segments — h=1.0 unit — octree
# ===========================================================================

def test_octree_parallel_lines_unit_n_parts_1_function_count():
    """n_parts=1: both function pairs collected in a single partition."""
    mesh = _two_parallel_lines_mesh(Real(1.0), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 2


def test_octree_parallel_lines_unit_n_parts_2_function_count():
    """n_parts=2: one function pair per partition (spatially separated in Y)."""
    mesh = _two_parallel_lines_mesh(Real(1.0), 2, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 2
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(2))
    assert counts == [1, 1]


def test_octree_parallel_lines_unit_n_parts_2_shared_vertices_on_different_lines():
    """Each partition's shared vertex lies on a different y-line (y≈0 vs y≈1)."""
    mesh = _two_parallel_lines_mesh(Real(1.0), 2, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    y_coords = []
    for pid in range(2):
        fns = p.functions_in_partition(Index(pid))
        assert len(fns) == 1
        sv = _shared_vertex_xyz(mesh, fns[0])
        y_coords.append(float(sv[1]))
    # Shared vertices at (1,0,0) and (1,1,0): |Δy| should be ≈1.
    assert abs(y_coords[0] - y_coords[1]) == pytest.approx(1.0, abs=float(_TOL))


def test_octree_parallel_lines_unit_n_parts_2_bboxes_disjoint_in_y():
    """Endpoint bounding boxes of the two partitions are disjoint in Y:
    one partition's endpoints all have y≈0, the other's all have y≈1.
    """
    mesh = _two_parallel_lines_mesh(Real(1.0), 2, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    bboxes = [_partition_endpoints_bbox(mesh, p, pid) for pid in range(2)]
    # Sort by minimum y so index 0 = lower line.
    bboxes.sort(key=lambda bb: float(bb[0][1]))
    low_min, low_max = bboxes[0]
    high_min, high_max = bboxes[1]
    # Lower partition: all endpoints at y=0.
    assert float(low_max[1]) == pytest.approx(0.0, abs=float(_TOL))
    # Upper partition: all endpoints at y=1.
    assert float(high_min[1]) == pytest.approx(1.0, abs=float(_TOL))


# ===========================================================================
# Rectangle (4 connected sides) — h=1.0 unit — octree
# ===========================================================================

def test_octree_rectangle_unit_n_parts_1_function_count():
    """n_parts=1: all 4 corner functions in a single partition."""
    mesh = _rectangle_mesh(Real(1.0), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 4


def test_octree_rectangle_unit_n_parts_1_bbox_spans_unit_square():
    """With all 4 functions in one partition, endpoints span the full unit square."""
    mesh = _rectangle_mesh(Real(1.0), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    bb_min, bb_max = _partition_endpoints_bbox(mesh, p, 0)
    assert float(bb_min[0]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_min[1]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_max[0]) == pytest.approx(1.0, abs=float(_TOL))
    assert float(bb_max[1]) == pytest.approx(1.0, abs=float(_TOL))


def test_octree_rectangle_unit_n_parts_4_function_count():
    """n_parts=4: each corner function in its own partition."""
    mesh = _rectangle_mesh(Real(1.0), 4, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(4))
    assert counts == [1, 1, 1, 1]


def test_octree_rectangle_unit_n_parts_4_shared_vertices_at_corners():
    """Each partition's shared vertex is close to one of the 4 unit-square corners,
    and all 4 corners are represented exactly once.
    """
    mesh = _rectangle_mesh(Real(1.0), 4, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    expected_corners = {
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    }
    found = set()
    tol = float(_TOL) * 2
    for pid in range(4):
        fns = p.functions_in_partition(Index(pid))
        assert len(fns) == 1
        sv = _shared_vertex_xyz(mesh, fns[0])
        matched = next(
            (c for c in expected_corners if np.allclose(sv, c, atol=tol)),
            None,
        )
        assert matched is not None, f'Shared vertex {sv} not near any expected corner'
        found.add(matched)
    assert found == expected_corners


def test_octree_rectangle_unit_n_parts_2_unachievable_raises():
    """n_parts=2 is unachievable: the 4 corners always occupy 4 distinct octree
    cells (one per XY quadrant), so no intermediate depth yields exactly 2.
    """
    mesh = _rectangle_mesh(Real(1.0), 1, PartitionMethod.OCTREE)
    with pytest.raises(Unrecoverable):
        Partitioner(
            mesh,
            mesh.mesh_functions,
            PartitionMethod.OCTREE,
            n_parts=2,
        )


# ===========================================================================
# Two parallel line segments — h=1.0 unit — kahip
# ===========================================================================

def test_kahip_parallel_lines_unit_n_parts_2_coverage_and_invertibility():
    """KaHIP assigns all functions to valid partitions; assignment is consistent.
    Exact counts are not asserted because KaHIP does not guarantee balanced
    splits for very small graphs.
    """
    pytest.importorskip('kahip')
    mesh = _two_parallel_lines_mesh(Real(1.0), 2, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 2
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


# ===========================================================================
# Rectangle (4 connected sides) — h=1.0 unit — kahip
# ===========================================================================

def test_kahip_rectangle_unit_n_parts_2_coverage_and_invertibility():
    """KaHIP 2-way partition of 4-function rectangle: coverage and invertibility.
    Exact counts are not asserted because KaHIP does not guarantee balanced
    splits for very small graphs.
    """
    pytest.importorskip('kahip')
    mesh = _rectangle_mesh(Real(1.0), 2, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 2
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


def test_kahip_rectangle_unit_n_parts_4_coverage_and_invertibility():
    """KaHIP 4-way partition of 4-function rectangle: coverage and invertibility."""
    pytest.importorskip('kahip')
    mesh = _rectangle_mesh(Real(1.0), 4, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


# ===========================================================================
# Two parallel line segments — h=0.5 half (3 function pairs per line = 6 total)
# Shared vertices of line A: (0.5,0,0), (1,0,0), (1.5,0,0)
# Shared vertices of line B: (0.5,1,0), (1,1,0), (1.5,1,0)
# Achievable n_parts: {1, 4, 6} (x and y both split simultaneously at depth 1).
# ===========================================================================

def test_octree_parallel_lines_half_n_parts_1_function_count():
    """h=0.5: each line produces 3 function pairs -> 6 total in 1 partition."""
    mesh = _two_parallel_lines_mesh(Real(0.5), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 6


def test_octree_parallel_lines_half_n_parts_6_function_count():
    """h=0.5, n_parts=6: each of the 6 shared vertices in its own partition.
    Achievable counts for this geometry are {1, 4, 6}; n_parts=2 is not achievable
    because both x and y dimensions split simultaneously at the first octree depth.
    """
    mesh = _two_parallel_lines_mesh(Real(0.5), 6, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 6
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(6))
    assert counts == [1] * 6


def test_octree_parallel_lines_half_n_parts_6_shared_vertices_cover_all_positions():
    """h=0.5, n_parts=6: shared vertices cover all 6 expected positions."""
    mesh = _two_parallel_lines_mesh(Real(0.5), 6, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    expected = {
        (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
        (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (1.5, 1.0, 0.0),
    }
    found = set()
    tol = float(_TOL) * 2
    for pid in range(6):
        fns = p.functions_in_partition(Index(pid))
        assert len(fns) == 1
        sv = _shared_vertex_xyz(mesh, fns[0])
        matched = next((e for e in expected if np.allclose(sv, e, atol=tol)), None)
        assert matched is not None, f'Shared vertex {sv} not near any expected position'
        found.add(matched)
    assert found == expected


def test_octree_parallel_lines_half_n_parts_6_bbox_contains_shared_vertex():
    """h=0.5, n_parts=6: each partition's endpoint bbox contains its shared vertex."""
    mesh = _two_parallel_lines_mesh(Real(0.5), 6, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    for pid in range(6):
        fns = p.functions_in_partition(Index(pid))
        sv = _shared_vertex_xyz(mesh, fns[0])
        bb_min, bb_max = _partition_endpoints_bbox(mesh, p, pid)
        for i in range(3):
            assert float(bb_min[i]) <= float(sv[i]) + float(_TOL)
            assert float(bb_max[i]) >= float(sv[i]) - float(_TOL)


def test_octree_parallel_lines_half_n_parts_2_unachievable_raises():
    """h=0.5, n_parts=2: achievable cell counts are {1,4,6}; 2 raises Unrecoverable."""
    mesh = _two_parallel_lines_mesh(Real(0.5), 1, PartitionMethod.OCTREE)
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=2)


def test_octree_parallel_lines_half_n_parts_4_function_count():
    """h=0.5, n_parts=4: octree splits on both x and y yielding [1,1,2,2] counts.
    x-dimension: 0.5 -> cell 0, 1.0 and 1.5 -> cell 1.
    y-dimension: y=0 -> cell 0, y=1 -> cell 1.
    4 cells: (0,0) 1 fn, (1,0) 2 fns, (0,1) 1 fn, (1,1) 2 fns.
    """
    mesh = _two_parallel_lines_mesh(Real(0.5), 4, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(4))
    assert counts == [1, 1, 2, 2]


# ===========================================================================
# Two parallel line segments — h=0.5 half — kahip
# ===========================================================================

def test_kahip_parallel_lines_half_n_parts_2_coverage_and_invertibility():
    """h=0.5, KaHIP n_parts=2: coverage and invertibility over 6 functions."""
    pytest.importorskip('kahip')
    mesh = _two_parallel_lines_mesh(Real(0.5), 2, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 2
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    assert n_fns == 6
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


# ===========================================================================
# Rectangle (4 connected sides) — h=0.5 half (8 shared vertices = 8 functions)
# 4 corner vertices: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
# 4 edge midpoints:  (0.5,0,0), (1,0.5,0), (0.5,1,0), (0,0.5,0)
# Octree cell counts across depths: 1 -> 4 -> 8.
# Achievable n_parts: {1, 4, 8}.  n_parts=2 is unachievable.
# At n_parts=4 (effective octree depth 1) the sorted partition sizes are [1,2,2,3]:
#   one cell holds only (0,0,0); two cells each hold 2 vertices; one holds 3.
# ===========================================================================

def test_octree_rectangle_half_n_parts_1_function_count():
    """h=0.5: corners + edge midpoints -> 8 shared vertices -> 8 functions."""
    mesh = _rectangle_mesh(Real(0.5), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 8


def test_octree_rectangle_half_n_parts_1_bbox_spans_unit_square():
    """h=0.5, n_parts=1: single partition endpoints span the full unit square."""
    mesh = _rectangle_mesh(Real(0.5), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    bb_min, bb_max = _partition_endpoints_bbox(mesh, p, 0)
    assert float(bb_min[0]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_min[1]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_max[0]) == pytest.approx(1.0, abs=float(_TOL))
    assert float(bb_max[1]) == pytest.approx(1.0, abs=float(_TOL))


def test_octree_rectangle_half_n_parts_4_function_count():
    """h=0.5, n_parts=4: octree at coarse depth groups 8 vertices into 4 cells
    with sorted partition sizes [1, 2, 2, 3].
    """
    mesh = _rectangle_mesh(Real(0.5), 4, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(4))
    assert counts == [1, 2, 2, 3]


def test_octree_rectangle_half_n_parts_8_function_count():
    """h=0.5, n_parts=8: each of the 8 shared vertices in its own partition."""
    mesh = _rectangle_mesh(Real(0.5), 8, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 8
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(8))
    assert counts == [1] * 8


def test_octree_rectangle_half_n_parts_8_shared_vertices_cover_all_positions():
    """h=0.5, n_parts=8: the 8 shared vertices across all partitions cover all
    4 corners and 4 edge midpoints of the unit square exactly once.
    """
    mesh = _rectangle_mesh(Real(0.5), 8, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    expected = {
        (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 0.5, 0.0),
    }
    found = set()
    tol = float(_TOL) * 2
    for pid in range(8):
        fns = p.functions_in_partition(Index(pid))
        assert len(fns) == 1
        sv = _shared_vertex_xyz(mesh, fns[0])
        matched = next(
            (e for e in expected if np.allclose(sv, e, atol=tol)),
            None,
        )
        assert matched is not None, f'Shared vertex {sv} not near any expected position'
        found.add(matched)
    assert found == expected


def test_octree_rectangle_half_n_parts_2_unachievable_raises():
    """h=0.5, n_parts=2: octree jumps 1->4->8 occupied cells; 2 never occurs."""
    mesh = _rectangle_mesh(Real(0.5), 1, PartitionMethod.OCTREE)
    with pytest.raises(Unrecoverable):
        Partitioner(
            mesh,
            mesh.mesh_functions,
            PartitionMethod.OCTREE,
            n_parts=2,
        )


# ===========================================================================
# Rectangle (4 connected sides) — h=0.5 half — kahip
# ===========================================================================

def test_kahip_rectangle_half_n_parts_4_coverage_and_invertibility():
    """h=0.5, KaHIP n_parts=4: coverage and invertibility over 8 functions."""
    pytest.importorskip('kahip')
    mesh = _rectangle_mesh(Real(0.5), 4, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    assert n_fns == 8
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


# ===========================================================================
# Two parallel line segments — h=0.1 tenth
# Each line: 2 segments of length 1.0, each giving 10 subsegments (h=0.1).
# 20 subsegments per line → 19 shared vertices per line → 38 functions total.
# Shared vertices of line A: x = 0.1, 0.2, ..., 1.9 at y=0.
# Shared vertices of line B: x = 0.1, 0.2, ..., 1.9 at y=1.
# Achievable n_parts: includes 1 and 38; n_parts=2 unachievable (both x and y
# split simultaneously at depth 1, same topology as the half case).
# ===========================================================================

def test_octree_parallel_lines_tenth_n_parts_1_function_count():
    """h=0.1: each line produces 19 function pairs -> 38 total in 1 partition."""
    mesh = _two_parallel_lines_mesh(Real(0.1), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 38


def test_octree_parallel_lines_tenth_n_parts_2_unachievable_raises():
    """h=0.1, n_parts=2: both x and y split at octree depth 1 -> 4 occupied
    cells; 2 is never achievable, same topology as the half case.
    """
    mesh = _two_parallel_lines_mesh(Real(0.1), 1, PartitionMethod.OCTREE)
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=2)


def test_octree_parallel_lines_tenth_n_parts_38_function_count():
    """h=0.1, n_parts=38: each of the 38 shared vertices in its own partition."""
    mesh = _two_parallel_lines_mesh(Real(0.1), 38, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 38
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(38))
    assert counts == [1] * 38


# ===========================================================================
# Two parallel line segments — h=0.1 tenth — kahip
# ===========================================================================

def test_kahip_parallel_lines_tenth_n_parts_4_coverage_and_invertibility():
    """h=0.1, KaHIP n_parts=4: coverage and invertibility over 38 functions."""
    pytest.importorskip('kahip')
    mesh = _two_parallel_lines_mesh(Real(0.1), 4, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    assert n_fns == 38
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))


# ===========================================================================
# Rectangle (4 connected sides) — h=0.1 tenth
# Each side has length 1.0 / h=0.1 = 10 subsegments.
# Interior shared vertices per side: 9.  Corner shared vertices: 4.
# 4 × 9 + 4 = 40 functions total.
# Achievable n_parts: includes 1 and 40; n_parts=2 unachievable (4 corners
# always occupy all 4 XY quadrants, same as the unit and half cases).
# ===========================================================================

def test_octree_rectangle_tenth_n_parts_1_function_count():
    """h=0.1: 4 corners + 4 × 9 edge-interior vertices -> 40 functions."""
    mesh = _rectangle_mesh(Real(0.1), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 1
    assert len(p.functions_in_partition(Index(0))) == 40


def test_octree_rectangle_tenth_n_parts_1_bbox_spans_unit_square():
    """h=0.1, n_parts=1: single partition endpoints span the full unit square."""
    mesh = _rectangle_mesh(Real(0.1), 1, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    bb_min, bb_max = _partition_endpoints_bbox(mesh, p, 0)
    assert float(bb_min[0]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_min[1]) == pytest.approx(0.0, abs=float(_TOL))
    assert float(bb_max[0]) == pytest.approx(1.0, abs=float(_TOL))
    assert float(bb_max[1]) == pytest.approx(1.0, abs=float(_TOL))


def test_octree_rectangle_tenth_n_parts_2_unachievable_raises():
    """h=0.1, n_parts=2: 4 corners always occupy all 4 XY quadrants; no depth
    yields exactly 2 occupied cells.
    """
    mesh = _rectangle_mesh(Real(0.1), 1, PartitionMethod.OCTREE)
    with pytest.raises(Unrecoverable):
        Partitioner(mesh, mesh.mesh_functions, PartitionMethod.OCTREE, n_parts=2)


def test_octree_rectangle_tenth_n_parts_40_function_count():
    """h=0.1, n_parts=40: each of the 40 shared vertices in its own partition."""
    mesh = _rectangle_mesh(Real(0.1), 40, PartitionMethod.OCTREE)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 40
    counts = sorted(len(p.functions_in_partition(Index(pid))) for pid in range(40))
    assert counts == [1] * 40


# ===========================================================================
# Rectangle (4 connected sides) — h=0.1 tenth — kahip
# ===========================================================================

def test_kahip_rectangle_tenth_n_parts_4_coverage_and_invertibility():
    """h=0.1, KaHIP n_parts=4: coverage and invertibility over 40 functions."""
    pytest.importorskip('kahip')
    mesh = _rectangle_mesh(Real(0.1), 4, PartitionMethod.KAHIP)
    p = mesh.mesh_functions.partitioner
    assert p.partition_count == 4
    n_fns = len(mesh.mesh_functions.function_subsegment_pairs)
    assert n_fns == 40
    all_fns = []
    for pid in range(p.partition_count):
        all_fns.extend(int(f) for f in p.functions_in_partition(Index(pid)))
    assert sorted(all_fns) == list(range(n_fns))
    for fi in range(n_fns):
        pid = int(p.partition_of_function(Index(fi)))
        assert Index(fi) in p.functions_in_partition(Index(pid))
