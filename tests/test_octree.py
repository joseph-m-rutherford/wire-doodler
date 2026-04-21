#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import numpy as np
import pytest

from doodler import Real
from doodler.errors import NeverImplement, Unrecoverable
from doodler.r3 import Octree, r3vector_copy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pt(x, y, z):
    return np.array([Real(x), Real(y), Real(z)])


_MIN = _pt(0, 0, 0)
_MAX = _pt(10, 10, 10)
_TOL = Real(0.1)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_octree_zero_tolerance_raises():
    with pytest.raises(Unrecoverable):
        Octree(_MIN, _MAX, Real(0))


def test_octree_negative_tolerance_raises():
    with pytest.raises(Unrecoverable):
        Octree(_MIN, _MAX, Real(-1))


def test_octree_min_equals_max_raises():
    with pytest.raises(Unrecoverable):
        Octree(_pt(1, 0, 0), _pt(1, 10, 10), _TOL)


def test_octree_min_greater_than_max_raises():
    with pytest.raises(Unrecoverable):
        Octree(_pt(5, 0, 0), _pt(1, 10, 10), _TOL)


def test_octree_valid_construction():
    tree = Octree(_MIN, _MAX, _TOL)
    assert tree.count == 0
    assert tree.depth > 0


# ---------------------------------------------------------------------------
# Immutable properties
# ---------------------------------------------------------------------------

def test_octree_immutable_properties():
    tree = Octree(_MIN, _MAX, _TOL)
    with pytest.raises(NeverImplement):
        tree.min_xyz = _MIN
    with pytest.raises(NeverImplement):
        tree.max_xyz = _MAX
    with pytest.raises(NeverImplement):
        tree.tolerance = _TOL
    with pytest.raises(NeverImplement):
        tree.depth = 0


def test_octree_min_xyz_returns_copy():
    tree = Octree(_MIN, _MAX, _TOL)
    m = tree.min_xyz
    m[:] = 999
    assert np.allclose(tree.min_xyz, _MIN)


def test_octree_max_xyz_returns_copy():
    tree = Octree(_MIN, _MAX, _TOL)
    m = tree.max_xyz
    m[:] = 999
    assert np.allclose(tree.max_xyz, _MAX)


# ---------------------------------------------------------------------------
# Depth computation
# ---------------------------------------------------------------------------

def test_octree_depth_satisfies_tolerance():
    tree = Octree(_MIN, _MAX, _TOL)
    max_extent = float(np.max(_MAX - _MIN))
    cell_size = max_extent / (2 ** tree.depth)
    assert cell_size <= float(_TOL)


def test_octree_depth_small_box_large_tolerance():
    # Bounding box extent (1.0) <= tolerance (2.0): depth should be 0.
    tree = Octree(_pt(0, 0, 0), _pt(1, 1, 1), Real(2.0))
    assert tree.depth == 0


# ---------------------------------------------------------------------------
# Morton key computation
# ---------------------------------------------------------------------------

def test_octree_morton_key_deterministic():
    tree = Octree(_MIN, _MAX, _TOL)
    p = _pt(5, 5, 5)
    assert tree.morton_key(p) == tree.morton_key(p)


def test_octree_morton_key_same_for_nearby_points():
    tree = Octree(_MIN, _MAX, _TOL)
    p1 = _pt(5.0, 5.0, 5.0)
    p2 = _pt(5.01, 5.01, 5.01)  # well within one cell at tolerance 0.1
    assert tree.morton_key(p1) == tree.morton_key(p2)


def test_octree_morton_key_different_for_distant_points():
    tree = Octree(_MIN, _MAX, _TOL)
    p1 = _pt(1, 1, 1)
    p2 = _pt(9, 9, 9)
    assert tree.morton_key(p1) != tree.morton_key(p2)


def test_octree_morton_key_at_min_corner():
    tree = Octree(_MIN, _MAX, _TOL)
    key = tree.morton_key(_MIN)
    assert key == 0


def test_octree_morton_key_at_max_corner():
    # Should not raise; the max corner is clamped to the last cell.
    tree = Octree(_MIN, _MAX, _TOL)
    tree.morton_key(_MAX)


def test_octree_morton_key_outside_bbox_raises():
    tree = Octree(_MIN, _MAX, _TOL)
    with pytest.raises(Unrecoverable):
        tree.morton_key(_pt(-1, 5, 5))
    with pytest.raises(Unrecoverable):
        tree.morton_key(_pt(5, 5, 11))


# ---------------------------------------------------------------------------
# Insertion and retrieval
# ---------------------------------------------------------------------------

def test_octree_insert_returns_morton_key():
    tree = Octree(_MIN, _MAX, _TOL)
    p = _pt(3, 4, 5)
    key = tree.insert(p)
    assert key == tree.morton_key(p)


def test_octree_insert_stores_point():
    tree = Octree(_MIN, _MAX, _TOL)
    p = _pt(3, 4, 5)
    key = tree.insert(p)
    assert tree.count == 1
    stored = tree.point(key)
    assert np.allclose(stored, p)


def test_octree_insert_duplicate_does_not_add():
    tree = Octree(_MIN, _MAX, _TOL)
    p1 = _pt(3, 4, 5)
    p2 = _pt(3.01, 4.01, 5.01)  # same cell
    k1 = tree.insert(p1)
    k2 = tree.insert(p2)
    assert k1 == k2
    assert tree.count == 1
    # The first inserted point is retained.
    stored = tree.point(k1)
    assert np.allclose(stored, p1)


def test_octree_insert_different_cells():
    tree = Octree(_MIN, _MAX, _TOL)
    tree.insert(_pt(1, 1, 1))
    tree.insert(_pt(9, 9, 9))
    assert tree.count == 2


def test_octree_insert_stores_copy():
    tree = Octree(_MIN, _MAX, _TOL)
    p = _pt(3, 4, 5)
    key = tree.insert(p)
    p[:] = 0  # mutate original
    stored = tree.point(key)
    assert np.allclose(stored, [3, 4, 5])


def test_octree_point_returns_copy():
    tree = Octree(_MIN, _MAX, _TOL)
    key = tree.insert(_pt(3, 4, 5))
    p = tree.point(key)
    p[:] = 0
    assert np.allclose(tree.point(key), [3, 4, 5])


def test_octree_point_missing_key_raises():
    tree = Octree(_MIN, _MAX, _TOL)
    with pytest.raises(Unrecoverable):
        tree.point(9999)


# ---------------------------------------------------------------------------
# Depth-zero edge case
# ---------------------------------------------------------------------------

def test_octree_depth_zero_all_same_key():
    tree = Octree(_pt(0, 0, 0), _pt(1, 1, 1), Real(2.0))
    assert tree.depth == 0
    k1 = tree.insert(_pt(0.1, 0.2, 0.3))
    k2 = tree.insert(_pt(0.9, 0.8, 0.7))
    assert k1 == k2 == 0
    assert tree.count == 1


# ---------------------------------------------------------------------------
# Interleave bits correctness
# ---------------------------------------------------------------------------

def test_interleave_bits_known_values():
    # (1,0,0) at depth 1 -> bit 0 of x set -> position 0 -> key = 1
    assert Octree._interleave_bits(1, 0, 0, 1) == 0b001
    # (0,1,0) at depth 1 -> bit 0 of y set -> position 1 -> key = 2
    assert Octree._interleave_bits(0, 1, 0, 1) == 0b010
    # (0,0,1) at depth 1 -> bit 0 of z set -> position 2 -> key = 4
    assert Octree._interleave_bits(0, 0, 1, 1) == 0b100
    # (1,1,1) at depth 1 -> all bits set -> key = 7
    assert Octree._interleave_bits(1, 1, 1, 1) == 0b111


def test_interleave_bits_depth_two():
    # (2,0,0) = binary (10,00,00) at depth 2
    # bit 0: x=0, y=0, z=0 -> 000
    # bit 1: x=1, y=0, z=0 -> 001 at positions 3,4,5
    # key = 0b001_000 = 8
    assert Octree._interleave_bits(2, 0, 0, 2) == 8
    # (3,3,3) = binary (11,11,11) at depth 2 -> all 6 bits set -> 63
    assert Octree._interleave_bits(3, 3, 3, 2) == 63
