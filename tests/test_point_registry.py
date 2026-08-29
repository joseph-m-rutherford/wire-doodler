#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import Index, Real, PointRegistry, r3
from doodler.errors import NeverImplement, Unrecoverable

_pt = r3.vector

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
