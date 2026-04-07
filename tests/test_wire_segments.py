#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import Real, WireSegment2D
from doodler.errors import NeverImplement, NotYetImplemented, Unrecoverable
from doodler.geometry.wire_segments import WireMesh3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt(x, y, z):
    return np.array([Real(x), Real(y), Real(z)])


# Two well-separated, non-intersecting polylines used as the "happy path".
_POLY_A = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(0, 1, 0)]
_POLY_B = [_pt(0, 0, 5), _pt(1, 0, 5), _pt(2, 0, 5)]
_H = Real(0.5)
_TOL = Real(0.01)

# ---------------------------------------------------------------------------
# WireSegment2D tests
# ---------------------------------------------------------------------------

def test_direct_construction_stores_points():
    pts = [(Real(0), Real(1)), (Real(2), Real(3))]
    seg = WireSegment2D(pts, 'test segment')
    assert seg.points == pts


def test_direct_construction_stores_description():
    seg = WireSegment2D([(Real(0), Real(0))], 'my description')
    assert seg.description == 'my description'


def test_points_setter_raises():
    seg = WireSegment2D([(Real(0), Real(0))], 'test')
    with pytest.raises(NeverImplement):
        seg.points = []


def test_description_setter_raises():
    seg = WireSegment2D([(Real(0), Real(0))], 'test')
    with pytest.raises(NeverImplement):
        seg.description = 'new'

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

def test_wire_mesh_3d_inter_shared_start_start_raises():
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(0, 0, 0), _pt(0, 1, 0)]  # same start point
    with pytest.raises(NotYetImplemented):
        WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)


def test_wire_mesh_3d_inter_shared_end_end_raises():
    poly_a = [_pt(0, 0, 0), _pt(1, 0, 0)]
    poly_b = [_pt(0, 1, 0), _pt(1, 0, 0)]  # same end point
    with pytest.raises(NotYetImplemented):
        WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)


def test_wire_mesh_3d_inter_shared_start_end_raises():
    poly_a = [_pt(1, 0, 0), _pt(2, 0, 0)]
    poly_b = [_pt(0, 0, 0), _pt(1, 0, 0)]  # end of b == start of a
    with pytest.raises(NotYetImplemented):
        WireMesh3D({'a': poly_a, 'b': poly_b}, _H, _TOL)


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
    # Segment length 1.0, h=0.5 → ceil(1.0/0.5)=2 subsegments each (collinear points).
    poly = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
    mesh = WireMesh3D({'p': poly}, _H, _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['p'] == [2, 2]


def test_wire_mesh_3d_subsegment_counts_non_multiple():
    # Segment length 1.0, h=0.3 → ceil(1.0/0.3)=ceil(3.333)=4 subsegments.
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'p': poly}, Real(0.3), _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['p'] == [4]


def test_wire_mesh_3d_subsegment_counts_h_larger_than_segment():
    # h larger than segment length → count must be >= 1, not 0.
    poly = [_pt(0, 0, 0), _pt(0.1, 0, 0)]
    mesh = WireMesh3D({'short': poly}, Real(1.0), _TOL)
    counts = mesh.named_subsegment_counts
    assert counts['short'] == [1]


def test_wire_mesh_3d_subsegment_counts_multiple_polylines():
    # Two polylines; verify counts are stored independently.
    # _POLY_A: seg0 length=1.0→2, seg1 length=sqrt(2)→ceil(sqrt(2)/0.5)=3
    # _POLY_B: both segments length=1.0→2
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

