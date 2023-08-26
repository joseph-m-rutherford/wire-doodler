#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

import doodler
from doodler import real_equality, r3vector_copy, r3vector_equality, TOLERANCE, Cylinder

import numpy as np

def test_r3vector_copy() -> None:
    '''Verify correct handling of instantiation and equality'''
    a = r3vector_copy((1,2,3))
    a_reference = np.array([1.,2.,3.])
    assert np.sum(np.abs(a - a_reference)) == 0.0
    assert r3vector_equality(a,(1,2,3),TOLERANCE)
    assert r3vector_equality(a,a_reference,TOLERANCE)
    b = r3vector_copy((3,2,1))
    b_reference = np.array([3.,2.,1.])
    assert np.sum(np.abs(b - b_reference)) == 0.0
    assert r3vector_equality(b,(3,2,1.),TOLERANCE)
    assert r3vector_equality(a,a_reference,TOLERANCE)
    assert not r3vector_equality(a,b,TOLERANCE)

def test_tangent_coordinates() -> None:
    from doodler.geometry import valid_tangent_coordinates, InvalidTangentCoordinates
    import pytest
    import random
    tested_points = dict()
    assert valid_tangent_coordinates(0,0)
    for i in range(1000):
        assert valid_tangent_coordinates(random.random(),random.random())
    assert valid_tangent_coordinates(1,1)
    assert not valid_tangent_coordinates(0.0,-0.01)
    assert not valid_tangent_coordinates(-0.01,0.01)
    assert not valid_tangent_coordinates(-0.01,-0.01)

    # Confirm that Cylinder raises if bad parameters are used
    htwo_rhalf_z = Cylinder((0.,0.,-1.),(0.,0.,1.),0.5)
    with pytest.raises(InvalidTangentCoordinates):
        htwo_rhalf_z.surface_differential_area(-1,0)
    with pytest.raises(InvalidTangentCoordinates):
        htwo_rhalf_z.surface_position_local(0,-1)        
       

def test_cylinder_bounding_box() -> None:
    '''Verify correct interpretation of position and orientation using bounding box vertices'''
    htwo_rhalf_z = Cylinder((0.,0.,-1.),(0.,0.,1.),0.5)
    min_uvw = np.zeros((3,))
    max_uvw = np.zeros((3,))
    htwo_rhalf_z.bounding_box_local(min_uvw,max_uvw)
    assert r3vector_equality((-0.5,-0.5,-1),min_uvw,TOLERANCE)
    assert r3vector_equality((0.5,0.5,1),max_uvw,TOLERANCE)

    min_xyz = np.zeros((3,))
    max_xyz = np.zeros((3,))
    htwo_rhalf_z.bounding_box_global(min_xyz,max_xyz)
    assert r3vector_equality((-0.5,-0.5,-1),min_xyz,TOLERANCE)
    assert r3vector_equality((0.5,0.5,1),max_xyz,TOLERANCE)

    # Flip along z-axis: bounding box unchanged
    htwo_rhalf_z_flip = Cylinder((0.,0.,1.),(0.,0.,-1.),0.5)
    htwo_rhalf_z_flip.bounding_box_local(min_uvw,max_uvw)
    assert r3vector_equality((-0.5,-0.5,-1),min_uvw,TOLERANCE)
    assert r3vector_equality((0.5,0.5,1),max_uvw,TOLERANCE)
    htwo_rhalf_z_flip.bounding_box_global(min_xyz,max_xyz)
    assert r3vector_equality((-0.5,-0.5,-1),min_xyz,TOLERANCE)
    assert r3vector_equality((0.5,0.5,1),max_xyz,TOLERANCE)

    # Interchange role of x,z
    alternate_htwo_rhalf_z_flip = Cylinder((1.,0.,0.),(-1.,0.,0.),0.5)
    alternate_htwo_rhalf_z_flip.bounding_box_local(min_uvw,max_uvw)
    assert r3vector_equality((-0.5,-0.5,-1),min_uvw,TOLERANCE)
    assert r3vector_equality((0.5,0.5,1),max_uvw,TOLERANCE)
    alternate_htwo_rhalf_z_flip.bounding_box_global(min_xyz,max_xyz)
    assert r3vector_equality((-1,-0.5,-0.5),min_xyz,TOLERANCE)
    assert r3vector_equality((1,0.5,0.5),max_xyz,TOLERANCE)


def test_cylinder_surface_tangent() -> None:
    # Verify positions on outer cylinder wall.
    htwo_rhalf_z = Cylinder((0.,0.,0.),(0.,0.,2.),0.5)
    assert r3vector_equality((0.5,0,0),htwo_rhalf_z.surface_position_global(0,0),TOLERANCE)
    assert r3vector_equality((0,0.5,0),htwo_rhalf_z.surface_position_global(0.25,0),TOLERANCE)
    assert r3vector_equality((-0.5,0,0),htwo_rhalf_z.surface_position_global(0.5,0),TOLERANCE)
    assert r3vector_equality((0,-0.5,0),htwo_rhalf_z.surface_position_global(0.75,0),TOLERANCE)
    assert r3vector_equality((0.5,0,0),htwo_rhalf_z.surface_position_global(1,0),TOLERANCE)
    assert r3vector_equality((0.5,0,1),htwo_rhalf_z.surface_position_global(0,0.5),TOLERANCE)
    assert r3vector_equality((0,0.5,1),htwo_rhalf_z.surface_position_global(0.25,0.5),TOLERANCE)
    assert r3vector_equality((-0.5,0,1),htwo_rhalf_z.surface_position_global(0.5,0.5),TOLERANCE)
    assert r3vector_equality((0,-0.5,1),htwo_rhalf_z.surface_position_global(0.75,0.5),TOLERANCE)
    assert r3vector_equality((0.5,0,1),htwo_rhalf_z.surface_position_global(1,0.5),TOLERANCE)
    # Differential area such that integration over s in [0,1] and t in [0,1] yields surface area of cylinder
    # Constant for all s,t
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(0,0),TOLERANCE)
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(0.1,0.9),TOLERANCE)
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(1,1),TOLERANCE)