#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import FillChoice, Index, Real, RuleCache, WireMesh3D, WireMesh3DFill, WireScalarFunction, r3

_pt = r3.vector


def _named(polylines: dict) -> dict:
    """Wrap plain {name: points} dicts into WireMesh3D's {name: (description, points)} format."""
    return {name: (name, points) for name, points in polylines.items()}


_cache = RuleCache()
_RULES = [
    _cache.uniform_x_gauss_rule(Index(3), Index(5)),
    _cache.uniform_x_kronrod_rule(Index(3), Index(5)),
    _cache.uniform_x_clenshaw_curtis_rule(Index(3), Index(5)),
]


@pytest.mark.parametrize('rule', _RULES)
def test_mass_filler_single_overlap_between_meshes(rule):
    # Only one subsegment overlaps: [1, 2] on the x-axis.
    test_mesh = WireMesh3D(_named({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}), Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D(_named({'basis': [_pt(1, 0, 0), _pt(2, 0, 0), _pt(3, 0, 0)]}), Real(1.0), Real(0.01))
    test_supports = test_mesh.function_supports
    basis_supports = basis_mesh.function_supports
    test_fn = WireScalarFunction(Index(1), Index(0), rule)
    basis_fn = WireScalarFunction(Index(1), Index(0), rule)

    filler = WireMesh3DFill(Real(1.0)).make_filler(
        test_mesh, test_supports, test_fn, basis_mesh, basis_supports, basis_fn, FillChoice.MASS
    )

    value = filler(Index(0), Index(0))
    assert np.isclose(value, Real(1.0 / 6.0))


@pytest.mark.parametrize('rule', _RULES)
def test_stiffness_filler_single_overlap_between_meshes(rule):
    # Only one subsegment overlaps: [1, 2] on the x-axis.
    test_mesh = WireMesh3D(_named({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}), Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D(_named({'basis': [_pt(1, 0, 0), _pt(2, 0, 0), _pt(3, 0, 0)]}), Real(1.0), Real(0.01))
    test_supports = test_mesh.function_supports
    basis_supports = basis_mesh.function_supports
    test_fn = WireScalarFunction(Index(1), Index(0), rule)
    basis_fn = WireScalarFunction(Index(1), Index(0), rule)

    filler = WireMesh3DFill(Real(1.0)).make_filler(
        test_mesh, test_supports, test_fn, basis_mesh, basis_supports, basis_fn, FillChoice.STIFFNESS
    )

    value = filler(Index(0), Index(0))
    assert np.isclose(value, Real(-1.0))


@pytest.mark.parametrize('rule', _RULES)
def test_fillers_support_reversed_overlapping_subsegment_orientation(rule):
    # Overlap is still [1, 2], but basis orientation is reversed on that segment.
    test_mesh = WireMesh3D(_named({'test': [_pt(0, 0, 0), _pt(1, 0, 0), _pt(2, 0, 0)]}), Real(1.0), Real(0.01))
    basis_mesh = WireMesh3D(_named({'basis': [_pt(3, 0, 0), _pt(2, 0, 0), _pt(1, 0, 0)]}), Real(1.0), Real(0.01))
    test_supports = test_mesh.function_supports
    basis_supports = basis_mesh.function_supports
    test_fn = WireScalarFunction(Index(1), Index(0), rule)
    basis_fn = WireScalarFunction(Index(1), Index(0), rule)

    mass_filler = WireMesh3DFill(Real(1.0)).make_filler(
        test_mesh, test_supports, test_fn, basis_mesh, basis_supports, basis_fn, FillChoice.MASS
    )
    stiffness_filler = WireMesh3DFill(Real(1.0)).make_filler(
        test_mesh, test_supports, test_fn, basis_mesh, basis_supports, basis_fn, FillChoice.STIFFNESS
    )

    mass_value = mass_filler(Index(0), Index(0))
    stiffness_value = stiffness_filler(Index(0), Index(0))

    assert np.isclose(mass_value, Real(1.0 / 6.0))
    assert np.isclose(stiffness_value, Real(-1.0))
