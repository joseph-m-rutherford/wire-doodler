#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import pytest

from doodler import Real, WireSegment2D
from doodler.errors import NeverImplement

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
