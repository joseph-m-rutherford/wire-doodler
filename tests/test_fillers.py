#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np

from doodler import FillChoice, Index, Real, WireMesh3D, WireMesh3DFill, r3

_pt = r3.vector

def test_mass_filler_single_overlap_between_meshes():
    # Only one subsegment overlaps: [1, 2] on the x-axis.
    test_mesh = WireMesh3D({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}, Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D({'basis': [_pt(1, 0, 0), _pt(2, 0, 0), _pt(3, 0, 0)]}, Real(1.0), Real(0.01))

    filler = WireMesh3DFill(Real(1.0)).make_filler(test_mesh, basis_mesh, FillChoice.MASS)

    value = filler(Index(0), Index(0))
    assert np.isclose(value, Real(1.0 / 6.0))


def test_stiffness_filler_single_overlap_between_meshes():
    # Only one subsegment overlaps: [1, 2] on the x-axis.
    test_mesh = WireMesh3D({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}, Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D({'basis': [_pt(1, 0, 0), _pt(2, 0, 0), _pt(3, 0, 0)]}, Real(1.0), Real(0.01))

    filler = WireMesh3DFill(Real(1.0)).make_filler(test_mesh, basis_mesh, FillChoice.STIFFNESS)

    value = filler(Index(0), Index(0))
    assert np.isclose(value, Real(-1.0))


def test_fillers_support_reversed_overlapping_subsegment_orientation():
    # Overlap is still [1, 2], but basis orientation is reversed on that segment.
    test_mesh = WireMesh3D({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}, Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D({'basis': [_pt(3, 0, 0), _pt(2, 0, 0), _pt(1, 0, 0)]}, Real(1.0), Real(0.01))

    mass_filler = WireMesh3DFill(Real(1.0)).make_filler(test_mesh, basis_mesh, FillChoice.MASS)
    stiffness_filler = WireMesh3DFill(Real(1.0)).make_filler(test_mesh, basis_mesh, FillChoice.STIFFNESS)

    mass_value = mass_filler(Index(0), Index(0))
    stiffness_value = stiffness_filler(Index(0), Index(0))

    assert np.isclose(mass_value, Real(1.0 / 6.0))
    assert np.isclose(stiffness_value, Real(-1.0))
