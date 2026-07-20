#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import Index, Real, PointRegistry, WireMesh3D, r3, unify_meshes
from doodler.errors import NeverImplement, NotYetImplemented, Unrecoverable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pt = r3.vector


# Two well-separated, non-intersecting polylines used as the "happy path".
_POLY_A = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(0, 1, 0)]
_POLY_B = [_pt(0, 0, 5), _pt(1, 0, 5), _pt(2, 0, 5)]
_H = Real(0.5)
_TOL = Real(0.01)


# ---------------------------------------------------------------------------
# WireMesh3D — parameter validation
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_zero_h_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D({'a': _POLY_A}, Real(0), _TOL)


def test_wire_mesh_3d_negative_h_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D({'a': _POLY_A}, Real(-1), _TOL)


def test_wire_mesh_3d_zero_reltol_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D({'a': _POLY_A}, _H, Real(0))


def test_wire_mesh_3d_negative_reltol_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D({'a': _POLY_A}, _H, Real(-1))


# ---------------------------------------------------------------------------
# WireMesh3D — intra-polyline: too few points
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_single_point_polyline_raises():
    with pytest.raises(Unrecoverable):
        WireMesh3D({'a': [_pt(0, 0, 0)]}, _H, _TOL)


# ---------------------------------------------------------------------------
# WireMesh3D — intra-polyline: colliding points
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_intra_duplicate_endpoints_raises():
    # First and last point are the same — collision within one polyline.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(0, 0, 0)]
    with pytest.raises(Unrecoverable):
        WireMesh3D({'loop': poly}, _H, _TOL)


def test_wire_mesh_3d_intra_nearby_interior_points_raises():
    # Two interior points closer than abstol.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(1.01, 0, 0), _pt(3, 0, 0)]
    with pytest.raises(Unrecoverable):
        WireMesh3D({'close': poly}, _H, _TOL)


# ---------------------------------------------------------------------------
# WireMesh3D — inter-polyline: shared endpoint vertices
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_inter_shared_start_start_supported():
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(0, 0, 0), _pt(0, 1, 0)]  # same start point
    mesh = WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)
    assert set(mesh.named_polylines.keys()) == {'a', 'b'}
    pairs = mesh.function_supports.support_subsegment_pairs
    shared = np.array([Real(0), Real(0), Real(0)])
    assert any(
        np.allclose(mesh.subsegment_endpoints(i)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(i)[1], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[1], shared, atol=float(_TOL))
        for i, j in pairs
    )


def test_wire_mesh_3d_inter_shared_end_end_supported():
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(0, 1, 0), _pt(1, 0, 0)]  # same end point
    mesh = WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)
    assert set(mesh.named_polylines.keys()) == {'a', 'b'}
    pairs = mesh.function_supports.support_subsegment_pairs
    shared = np.array([Real(1), Real(0), Real(0)])
    assert any(
        np.allclose(mesh.subsegment_endpoints(i)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(i)[1], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[1], shared, atol=float(_TOL))
        for i, j in pairs
    )


def test_wire_mesh_3d_inter_shared_start_end_supported():
    poly_a = [_pt(1, 0, 0), _pt(2, 0, 0)]
    poly_b = [_pt(0, 0, 0), _pt(1, 0, 0)]  # end of b == start of a
    mesh = WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)
    assert set(mesh.named_polylines.keys()) == {'a', 'b'}
    pairs = mesh.function_supports.support_subsegment_pairs
    shared = np.array([Real(1), Real(0), Real(0)])
    assert any(
        np.allclose(mesh.subsegment_endpoints(i)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(i)[1], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[0], shared, atol=float(_TOL))
        or np.allclose(mesh.subsegment_endpoints(j)[1], shared, atol=float(_TOL))
        for i, j in pairs
    )


# ---------------------------------------------------------------------------
# WireMesh3D — inter-polyline: intersecting segments
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_inter_crossing_segments_raises():
    # Two perpendicular segments that cross at the origin in the XY plane.
    poly_a = [_pt(-1, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(0, -1, 0), _pt(0, 1, 0)]
    with pytest.raises(NotYetImplemented):
        WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)


def test_wire_mesh_3d_inter_parallel_close_segments_raises():
    # Two parallel segments separated by less than reltol.
    poly_a = [_pt(0, 0, 0), _pt(2, 0, 0)]
    poly_b = [_pt(1, 0.005, 0), _pt(2, 0.005, 0)]  # 0.005 < _TOL=0.01
    with pytest.raises(NotYetImplemented):
        WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)


# ---------------------------------------------------------------------------
# WireMesh3D — valid construction
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_valid_construction():
    mesh = WireMesh3D({'a': _POLY_A, 'b': _POLY_B}, _H, _TOL)
    assert set(mesh.named_polylines.keys()) == {'a', 'b'}


def test_wire_mesh_3d_immutable_properties_raise():
    mesh = WireMesh3D({'a': _POLY_A, 'b': _POLY_B}, _H, _TOL)
    with pytest.raises(NeverImplement):
        mesh.named_polylines = {}
    with pytest.raises(NeverImplement):
        mesh.h = Real(1)
    with pytest.raises(NeverImplement):
        mesh.reltol = Real(1)
    with pytest.raises(NeverImplement):
        mesh.named_subsegment_counts = {}


# ---------------------------------------------------------------------------
# WireMesh3D — subsegment counts
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_subsegment_counts_exact_multiple():
    # Segment length 1.0, h=0.5 -> ceil(1.0/0.5)=2 subsegments each (collinear points).
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, _H, _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['p'] == [2, 2]


def test_wire_mesh_3d_subsegment_counts_non_multiple():
    # Segment length 1.0, h=0.3 -> ceil(1.0/0.3)=ceil(3.333)=4 subsegments.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(0.3), _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['p'] == [4]


def test_wire_mesh_3d_subsegment_counts_h_larger_than_segment():
    # h larger than segment length -> count must be >= 1, not 0.
    poly = [_pt(0, 0, 0), _pt(0.1, 0, 0)]
    mesh = WireMesh3D({'short': poly}, Real(1.0), _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['short'] == [1]


def test_wire_mesh_3d_subsegment_counts_multiple_polylines():
    # Two polylines; verify counts are stored independently.
    # _POLY_A: seg0 length=1.0->2, seg1 length=sqrt(2)->ceil(sqrt(2)/0.5)=3
    # _POLY_B: both segments length=1.0->2
    mesh = WireMesh3D({'a': _POLY_A, 'b': _POLY_B}, _H, _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['a'] == [2, 3]
    assert counts['b'] == [2, 2]


def test_wire_mesh_3d_subsegment_counts_all_at_least_one():
    # Construct a polyline where every segment has length well below h.
    pts = [_pt(i * 0.01, 0, 0) for i in range(5)]  # 4 segments each 0.01 long, h=1.0
    mesh = WireMesh3D({'tiny': pts}, Real(1.0), _TOL)
    for count in mesh.named_subsegment_counts['tiny']:
        assert count >= 1


# ---------------------------------------------------------------------------
# WireMesh3D — subsegment index
# ---------------------------------------------------------------------------

def test_subsegment_index_single_polyline_single_segment():
    # One segment [0,0,0]->[1,0,0] with h=0.5 gives 2 subsegments.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': poly}, Real(0.5), _TOL)
    idx = mesh.subsegment_index
    assert len(idx) == 2
    assert idx[0] == ('a', 0, 0)
    assert idx[1] == ('a', 0, 1)


def test_subsegment_index_single_polyline_two_segments():
    # Two-segment polyline each length 1, h=0.5 -> 2 subsegments per segment = 4 total.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(0.5), _TOL)
    idx = mesh.subsegment_index
    assert len(idx) == 4
    assert idx[0] == ('p', 0, 0)
    assert idx[1] == ('p', 0, 1)
    assert idx[2] == ('p', 1, 0)
    assert idx[3] == ('p', 1, 1)


def test_subsegment_index_two_polylines_lexicographic_order():
    # Names 'b' and 'a': 'a' must come first in the index regardless of insertion order.
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]        # 1 segment, h=1.0 -> 1 subsegment
    poly_b = [_pt(0, 0, 5), _pt(1, 0, 5), _pt(2, 0, 5)]  # 2 segments -> 2 subsegments
    mesh = WireMesh3D({'b': poly_b, 'a': poly_a}, Real(1.0), _TOL)
    idx = mesh.subsegment_index
    # 'a' first: 1 subsegment; 'b' second: 2 subsegments -> 3 total
    assert len(idx) == 3
    assert idx[0] == ('a', 0, 0)
    assert idx[1] == ('b', 0, 0)
    assert idx[2] == ('b', 1, 0)


def test_subsegment_index_length_matches_total_subsegment_count():
    # Total index length must equal sum of all subsegment counts.
    mesh = WireMesh3D({'a': _POLY_A, 'b': _POLY_B}, _H, _TOL)
    total = sum(
        count
        for counts in mesh.named_subsegment_counts.values()
        for count in counts
    )
    assert len(mesh.subsegment_index) == total


def test_subsegment_index_immutable():
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    with pytest.raises(NeverImplement):
        mesh.subsegment_index = []


def test_subsegment_index_returns_copy():
    # Mutating the returned list must not affect the stored index.
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    idx = mesh.subsegment_index
    original_len = len(idx)
    idx.clear()
    assert len(mesh.subsegment_index) == original_len


# ---------------------------------------------------------------------------
# PointRegistry
# ---------------------------------------------------------------------------

# Bounding box for standalone PointRegistry tests.
_REG_MIN = _pt(-1, -1, -1)
_REG_MAX = _pt(10, 10, 10)


def test_point_registry_insert_and_retrieve():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    idx0 = reg.get_or_insert(_pt(0, 0, 0))
    idx1 = reg.get_or_insert(_pt(1, 0, 0))
    assert idx0 == 0
    assert idx1 == 1
    assert reg.count == 2


def test_point_registry_deduplicates_within_tolerance():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    idx0 = reg.get_or_insert(_pt(1, 0, 0))
    idx1 = reg.get_or_insert(_pt(1.005, 0, 0))  # within 0.01 relative tolerance of (1,0,0)
    assert idx0 == idx1
    assert reg.count == 1


def test_point_registry_distinguishes_beyond_tolerance():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.001))
    idx0 = reg.get_or_insert(_pt(1, 0, 0))
    idx1 = reg.get_or_insert(_pt(1.01, 0, 0))  # 1% away, well beyond 0.001
    assert idx0 != idx1
    assert reg.count == 2


def test_point_registry_near_origin():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    idx0 = reg.get_or_insert(_pt(0, 0, 0))
    idx1 = reg.get_or_insert(_pt(1e-8, 1e-9, 0))  # both near origin
    assert idx0 == idx1
    assert reg.count == 1


def test_point_registry_point_returns_copy():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    reg.get_or_insert(_pt(1, 2, 3))
    p = reg.point(Index(0))
    p[:] = 0
    assert np.allclose(reg.point(Index(0)), [1, 2, 3])


def test_point_registry_point_out_of_range_raises():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    with pytest.raises(Unrecoverable):
        reg.point(Index(0))


def test_point_registry_morton_keys_for_aabb_contains_point_cell_key():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    p = _pt(1, 2, 3)
    key = reg._octree.morton_key(p)
    keys = reg.morton_keys_for_aabb(p, p)
    assert keys is not None
    assert key in keys


def test_point_registry_morton_keys_for_aabb_disjoint_is_empty():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    keys = reg.morton_keys_for_aabb(_pt(100, 100, 100), _pt(101, 101, 101))
    assert keys == []


def test_point_registry_morton_keys_for_aabb_max_cells_overflow_returns_none():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    keys = reg.morton_keys_for_aabb(_REG_MIN, _REG_MAX, max_cells=1)
    assert keys is None


def test_point_registry_morton_keys_for_aabb_invalid_bounds_raises():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    with pytest.raises(Unrecoverable):
        reg.morton_keys_for_aabb(_pt(1, 1, 1), _pt(0, 0, 0))


def test_point_registry_immutable_properties():
    reg = PointRegistry(_REG_MIN, _REG_MAX, Real(0.01))
    with pytest.raises(NeverImplement):
        reg.reltol = Real(0.1)
    with pytest.raises(NeverImplement):
        reg.points = []
    with pytest.raises(NeverImplement):
        reg.min_xyz = _REG_MIN
    with pytest.raises(NeverImplement):
        reg.max_xyz = _REG_MAX
    with pytest.raises(NeverImplement):
        reg.abstol = Real(0.1)


def test_point_registry_negative_reltol_raises():
    with pytest.raises(Unrecoverable):
        PointRegistry(_REG_MIN, _REG_MAX, Real(-1))


def test_point_registry_zero_reltol_raises():
    with pytest.raises(Unrecoverable):
        PointRegistry(_REG_MIN, _REG_MAX, Real(0))


# ---------------------------------------------------------------------------
# WireMesh3D — point registry and subsegment point pairs
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_point_registry_exists():
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': poly}, Real(1.0), _TOL)
    assert mesh.point_registry.count == 2


def test_wire_mesh_3d_point_pairs_length_matches_subsegments():
    mesh = WireMesh3D({'a': _POLY_A, 'b': _POLY_B}, _H, _TOL)
    assert len(mesh.subsegment_point_pairs) == len(mesh.subsegment_index)


def test_wire_mesh_3d_point_pairs_match_endpoints():
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(1.0), _TOL)
    reg = mesh.point_registry
    for mesh_idx in range(len(mesh.subsegment_index)):
        start, end = mesh.subsegment_endpoints(Index(mesh_idx))
        pi, pj = mesh.subsegment_point_pairs[mesh_idx]
        assert np.allclose(reg.point(pi), start, atol=float(_TOL))
        assert np.allclose(reg.point(pj), end, atol=float(_TOL))


def test_wire_mesh_3d_shared_endpoint_same_index():
    # Two-segment polyline: the interior vertex should share one index.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(1.0), _TOL)
    pairs = mesh.subsegment_point_pairs
    # subseg 0 end == subseg 1 start
    assert pairs[0][1] == pairs[1][0]


def test_wire_mesh_3d_inter_polyline_shared_point_same_index():
    # Two polylines sharing an endpoint at (1,0,0).
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'a': poly_a, 'b': poly_b}, Real(1.0), _TOL)
    pairs = mesh.subsegment_point_pairs
    # 'a' comes before 'b' in lexicographic order.
    # subseg 0 (a): (0,0,0)->(1,0,0), subseg 1 (b): (1,0,0)->(2,0,0)
    assert pairs[0][1] == pairs[1][0]


def test_wire_mesh_3d_unique_point_count_with_subdivisions():
    # Polyline with h=0.5: (0,0,0)->(1,0,0) becomes 2 subsegments.
    # Unique points: (0,0,0), (0.5,0,0), (1,0,0) = 3
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': poly}, Real(0.5), _TOL)
    assert mesh.point_registry.count == 3


def test_wire_mesh_3d_point_registry_immutable():
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    with pytest.raises(NeverImplement):
        mesh.point_registry = None


def test_wire_mesh_3d_subsegment_point_pairs_immutable():
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    with pytest.raises(NeverImplement):
        mesh.subsegment_point_pairs = []


def test_wire_mesh_3d_subsegment_point_pairs_returns_copy():
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    pairs = mesh.subsegment_point_pairs
    original_len = len(pairs)
    pairs.clear()
    assert len(mesh.subsegment_point_pairs) == original_len


# ---------------------------------------------------------------------------
# WireMesh3D — vertex_xyz and vertex_count
# ---------------------------------------------------------------------------

def test_wire_mesh_3d_vertex_count():
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': poly}, Real(1.0), _TOL)
    assert mesh.vertex_count == 2


def test_wire_mesh_3d_vertex_count_immutable():
    mesh = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    with pytest.raises(NeverImplement):
        mesh.vertex_count = 0


def test_wire_mesh_3d_vertex_xyz_returns_copy():
    poly = [_pt(1, 2, 3), _pt(4, 5, 6)]
    mesh = WireMesh3D({'a': poly}, Real(10.0), _TOL)
    v = mesh.vertex_xyz(Index(0))
    v[:] = 0
    assert np.allclose(mesh.vertex_xyz(Index(0)), [1, 2, 3])


def test_wire_mesh_3d_vertex_xyz_matches_endpoints():
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(1.0), _TOL)
    for mesh_idx in range(len(mesh.subsegment_index)):
        start, end = mesh.subsegment_endpoints(Index(mesh_idx))
        pi, pj = mesh.subsegment_point_pairs[mesh_idx]
        assert np.allclose(mesh.vertex_xyz(pi), start, atol=float(_TOL))
        assert np.allclose(mesh.vertex_xyz(pj), end, atol=float(_TOL))


def test_wire_mesh_3d_vertex_xyz_out_of_range_raises():
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': poly}, Real(1.0), _TOL)
    with pytest.raises(Unrecoverable):
        mesh.vertex_xyz(Index(999))


# ---------------------------------------------------------------------------
# unify_meshes
# ---------------------------------------------------------------------------

def test_unify_meshes_no_shared_points():
    mesh_a = WireMesh3D({'a': [_pt(0, 0, 0), _pt(1, 0, 0)]}, Real(1.0), _TOL)
    mesh_b = WireMesh3D({'b': [_pt(5, 0, 0), _pt(6, 0, 0)]}, Real(1.0), _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    # Both unified meshes share the same registry with 4 distinct points.
    assert ua.point_registry is ub.point_registry
    assert ua.vertex_count == 4
    assert ub.vertex_count == 4


def test_unify_meshes_with_shared_endpoint():
    mesh_a = WireMesh3D({'a': [_pt(0, 0, 0), _pt(1, 0, 0)]}, Real(1.0), _TOL)
    mesh_b = WireMesh3D({'b': [_pt(1, 0, 0), _pt(2, 0, 0)]}, Real(1.0), _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    # 3 unique points: (0,0,0), (1,0,0), (2,0,0)
    assert ua.vertex_count == 3
    # The shared point (1,0,0) maps to the same index.
    # ua subseg 0: (0,0,0)->(1,0,0), ub subseg 0: (1,0,0)->(2,0,0)
    assert ua.subsegment_point_pairs[0][1] == ub.subsegment_point_pairs[0][0]


def test_unify_meshes_shared_registry_identity():
    mesh_a = WireMesh3D({'a': _POLY_A}, _H, _TOL)
    mesh_b = WireMesh3D({'b': _POLY_B}, _H, _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    assert ua.point_registry is ub.point_registry


def test_unify_meshes_preserves_subsegment_endpoints():
    mesh_a = WireMesh3D({'a': [_pt(0, 0, 0), _pt(1, 0, 0)]}, Real(1.0), _TOL)
    mesh_b = WireMesh3D({'b': [_pt(2, 0, 0), _pt(3, 0, 0)]}, Real(1.0), _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    # Subsegment endpoints should still match the original polyline data.
    for mesh_idx in range(len(ua.subsegment_index)):
        orig_start, orig_end = mesh_a.subsegment_endpoints(Index(mesh_idx))
        new_start, new_end = ua.subsegment_endpoints(Index(mesh_idx))
        assert np.allclose(orig_start, new_start, atol=float(_TOL))
        assert np.allclose(orig_end, new_end, atol=float(_TOL))


def test_unify_meshes_preserves_mesh_functions():
    # Two polylines sharing an endpoint -> function supports should still work after unification.
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(1, 0, 0), _pt(2, 0, 0)]
    mesh_a = WireMesh3D({'a': poly_a, 'b': poly_b}, Real(1.0), _TOL)
    mesh_b = WireMesh3D({'c': [_pt(5, 0, 0), _pt(6, 0, 0)]}, Real(1.0), _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    # mesh_a has a support connecting the two subsegments; that should be preserved.
    assert len(ua.function_supports.support_subsegment_pairs) == len(mesh_a.function_supports.support_subsegment_pairs)


def test_unify_meshes_vertex_xyz_consistent():
    mesh_a = WireMesh3D({'a': [_pt(0, 0, 0), _pt(1, 0, 0)]}, Real(1.0), _TOL)
    mesh_b = WireMesh3D({'b': [_pt(1, 0, 0), _pt(2, 0, 0)]}, Real(1.0), _TOL)
    ua, ub = unify_meshes(mesh_a, mesh_b)
    # All point pair indices should resolve to valid coordinates.
    for pi, pj in ua.subsegment_point_pairs:
        ua.vertex_xyz(pi)
        ua.vertex_xyz(pj)
    for pi, pj in ub.subsegment_point_pairs:
        ub.vertex_xyz(pi)
        ub.vertex_xyz(pj)
