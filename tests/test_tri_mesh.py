#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import Real, WireMesh3D, r3
from doodler.errors import Unrecoverable

_pt = r3.vector


def _triangle_mesh_area(vertices, triangles) -> float:
    area = 0.0
    for i0, i1, i2 in triangles:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        area += 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
    return area


def _cone_cylinder_cone_mesh() -> WireMesh3D:
    """Straight polyline whose subsegment radii taper (0,r) -> (r,r) -> (r,0)."""
    poly = [_pt(0, 0, 0), _pt(0, 0, 10), _pt(0, 0, 20), _pt(0, 0, 30)]
    return WireMesh3D({'a': ('r=2.0', poly)}, Real(100), Real(1e-3))


def test_export_tri_mesh_invalid_num_sides_raises():
    mesh = _cone_cylinder_cone_mesh()
    with pytest.raises(Unrecoverable):
        mesh.export_tri_mesh(2)


def test_export_tri_mesh_vertex_and_triangle_counts():
    mesh = _cone_cylinder_cone_mesh()
    n = 12
    vertices, triangles = mesh.export_tri_mesh(n)
    # Two cones (ring + apex, n triangles each) and one cylinder (two rings, 2n triangles).
    assert len(vertices) == 4 * n + 2
    assert len(triangles) == 4 * n


def test_export_tri_mesh_degenerate_single_subsegment_is_empty():
    """A polyline with a single subsegment tapers to zero radius at both ends."""
    poly = [_pt(0, 0, 0), _pt(1, 0, 0)]
    mesh = WireMesh3D({'a': ('r=5.0', poly)}, Real(1.0), Real(1e-3))
    vertices, triangles = mesh.export_tri_mesh(6)
    assert vertices == []
    assert triangles == []


def test_export_tri_mesh_surface_area_matches_continuous_geometry():
    mesh = _cone_cylinder_cone_mesh()
    r = 2.0
    h = 10.0
    slant = np.sqrt(h ** 2 + r ** 2)
    continuous_area = 2 * (np.pi * r * slant) + 2 * np.pi * r * h

    vertices, triangles = mesh.export_tri_mesh(64)
    mesh_area = _triangle_mesh_area(vertices, triangles)
    relative_error = abs(mesh_area - continuous_area) / continuous_area
    assert relative_error < 1e-3


def test_export_tri_mesh_surface_area_converges_with_num_sides():
    mesh = _cone_cylinder_cone_mesh()
    r = 2.0
    h = 10.0
    slant = np.sqrt(h ** 2 + r ** 2)
    continuous_area = 2 * (np.pi * r * slant) + 2 * np.pi * r * h

    coarse_vertices, coarse_triangles = mesh.export_tri_mesh(8)
    fine_vertices, fine_triangles = mesh.export_tri_mesh(64)
    coarse_error = abs(_triangle_mesh_area(coarse_vertices, coarse_triangles) - continuous_area)
    fine_error = abs(_triangle_mesh_area(fine_vertices, fine_triangles) - continuous_area)
    assert fine_error < coarse_error


def test_export_tri_mesh_cross_sectional_area_matches_circle():
    """The polygon ring at a nonzero-radius end has the same area as the circle it approximates."""
    poly = [_pt(0, 0, 0), _pt(0, 0, 10), _pt(0, 0, 20)]
    mesh = WireMesh3D({'a': ('r=3.0', poly)}, Real(100), Real(1e-3))
    n = 8
    vertices, _triangles = mesh.export_tri_mesh(n)
    # The shared joint ring (interior vertex of the two-subsegment polyline) is the
    # first ring emitted for the second subsegment: indices [0, n) belong to the
    # first subsegment's cone apex/ring, so inspect the second subsegment's ring.
    ring = vertices[n + 1:n + 1 + n]
    center = np.array([0, 0, 10], dtype=Real)
    circumradius = float(np.linalg.norm(ring[0] - center))
    polygon_area = 0.5 * n * circumradius ** 2 * np.sin(2 * np.pi / n)
    circle_area = np.pi * 3.0 ** 2
    assert abs(polygon_area - circle_area) / circle_area < 1e-9
