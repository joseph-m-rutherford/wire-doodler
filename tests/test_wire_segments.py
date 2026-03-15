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
_POLY_A = [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]
_POLY_B = [_pt(0, 5, 0), _pt(1, 5, 0), _pt(2, 5, 0)]
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

