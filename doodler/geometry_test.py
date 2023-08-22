#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from geometry import real_equality, vector3d, vector3d_equality, valid_tangent_coordinates, TOLERANCE, Cylinder

import numpy as np

def test_vector3d() -> None:
    '''Verify correct handling of instantiation and equality'''
    a = vector3d((1,2,3))
    a_reference = np.array([1.,2.,3.])
    assert np.sum(np.abs(a - a_reference)) == 0.0
    assert vector3d_equality(a,a_reference,TOLERANCE)
    b = vector3d((3,2,1))
    b_reference = np.array([3.,2.,1.])
    assert np.sum(np.abs(b - b_reference)) == 0.0
    assert vector3d_equality(a,a_reference,TOLERANCE)
    assert not vector3d_equality(a,b,TOLERANCE)

def test_cylinder_bounding_box() -> None:
    '''Verify correct interpretation of position and orientation using bounding box vertices'''
    htwo_rhalf_z = Cylinder((0.,0.,-1.),(0.,0.,1.),0.5)
    min_uvw = np.zeros((3,))
    max_uvw = np.zeros((3,))
    htwo_rhalf_z.bounding_box_local(min_uvw,max_uvw)
    assert vector3d_equality(vector3d((-0.5,-0.5,-1)),min_uvw,TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0.5,1)),max_uvw,TOLERANCE)

    min_xyz = np.zeros((3,))
    max_xyz = np.zeros((3,))
    htwo_rhalf_z.bounding_box_global(min_xyz,max_xyz)
    assert vector3d_equality(vector3d((-0.5,-0.5,-1)),min_xyz,TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0.5,1)),max_xyz,TOLERANCE)

    # Flip along z-axis: bounding box unchanged
    htwo_rhalf_z_flip = Cylinder((0.,0.,1.),(0.,0.,-1.),0.5)
    htwo_rhalf_z_flip.bounding_box_local(min_uvw,max_uvw)
    assert vector3d_equality(vector3d((-0.5,-0.5,-1)),min_uvw,TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0.5,1)),max_uvw,TOLERANCE)
    htwo_rhalf_z_flip.bounding_box_global(min_xyz,max_xyz)
    assert vector3d_equality(vector3d((-0.5,-0.5,-1)),min_xyz,TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0.5,1)),max_xyz,TOLERANCE)

    # Interchange role of x,z
    alternate_htwo_rhalf_z_flip = Cylinder((1.,0.,0.),(-1.,0.,0.),0.5)
    alternate_htwo_rhalf_z_flip.bounding_box_local(min_uvw,max_uvw)
    assert vector3d_equality(vector3d((-0.5,-0.5,-1)),min_uvw,TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0.5,1)),max_uvw,TOLERANCE)
    alternate_htwo_rhalf_z_flip.bounding_box_global(min_xyz,max_xyz)
    assert vector3d_equality(vector3d((-1,-0.5,-0.5)),min_xyz,TOLERANCE)
    assert vector3d_equality(vector3d((1,0.5,0.5)),max_xyz,TOLERANCE)


def test_cylinder_surface_tangent() -> None:
    assert not valid_tangent_coordinates(-0.01,-0.01)
    assert not valid_tangent_coordinates(0.01,-0.01)
    assert not valid_tangent_coordinates(-0.01,0.01)
    assert not valid_tangent_coordinates(-0.01,-0.01)
    assert valid_tangent_coordinates(0,0)
    assert valid_tangent_coordinates(0,1)
    assert valid_tangent_coordinates(1,0)
    assert valid_tangent_coordinates(1,1)
    assert not valid_tangent_coordinates(1.01,0.99)
    assert not valid_tangent_coordinates(0.99,1.01)
    assert not valid_tangent_coordinates(1.01,1.01)
    # Verify positions on outer cylinder wall.
    htwo_rhalf_z = Cylinder((0.,0.,0.),(0.,0.,2.),0.5)
    assert vector3d_equality(vector3d((0.5,0,0)),htwo_rhalf_z.surface_position_global(0,0),TOLERANCE)
    assert vector3d_equality(vector3d((0,0.5,0)),htwo_rhalf_z.surface_position_global(0.25,0),TOLERANCE)
    assert vector3d_equality(vector3d((-0.5,0,0)),htwo_rhalf_z.surface_position_global(0.5,0),TOLERANCE)
    assert vector3d_equality(vector3d((0,-0.5,0)),htwo_rhalf_z.surface_position_global(0.75,0),TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0,0)),htwo_rhalf_z.surface_position_global(1,0),TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0,1)),htwo_rhalf_z.surface_position_global(0,0.5),TOLERANCE)
    assert vector3d_equality(vector3d((0,0.5,1)),htwo_rhalf_z.surface_position_global(0.25,0.5),TOLERANCE)
    assert vector3d_equality(vector3d((-0.5,0,1)),htwo_rhalf_z.surface_position_global(0.5,0.5),TOLERANCE)
    assert vector3d_equality(vector3d((0,-0.5,1)),htwo_rhalf_z.surface_position_global(0.75,0.5),TOLERANCE)
    assert vector3d_equality(vector3d((0.5,0,1)),htwo_rhalf_z.surface_position_global(1,0.5),TOLERANCE)
    # Differential area such that integration over s in [0,1] and t in [0,1] yields surface area of cylinder
    # Constant for all s,t
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(0,0),TOLERANCE)
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(0.1,0.9),TOLERANCE)
    assert real_equality(2*np.pi*0.5*2,htwo_rhalf_z.surface_differential_area(1,1),TOLERANCE)



test_vector3d()
test_cylinder_bounding_box()
test_cylinder_surface_tangent() 