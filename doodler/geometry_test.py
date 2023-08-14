#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from geometry import vector3d, vector3d_equality, RELATIVE_TOLERANCE, Cylinder

import numpy as np

def test_vector3d():
    a = vector3d((1,2,3))
    a_reference = np.array([1.,2.,3.])
    assert np.sum(np.abs(a - a_reference)) == 0.0
    assert vector3d_equality(a,a_reference,RELATIVE_TOLERANCE)
    b = vector3d((3,2,1))
    b_reference = np.array([3.,2.,1.])
    assert np.sum(np.abs(b - b_reference)) == 0.0
    assert vector3d_equality(a,a_reference,RELATIVE_TOLERANCE)
    assert not vector3d_equality(a,b,RELATIVE_TOLERANCE)

test_vector3d()