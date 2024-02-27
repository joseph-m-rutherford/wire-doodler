#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from doodler import geometry, quadrature, real_equality
from doodler.geometry import ClippedSphere, Cylinder, Shape3DSampler, LeftHanded, RightHanded

import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

RNG_SEED = 20240115
generator = np.random.default_rng(RNG_SEED)

def export_samples(sampler:geometry.Shape3DSampler, filename:str) -> None:
        quadrature_weights = sampler.weights.flatten()
        with open(filename,'w') as outlet:
            outlet.write('x,y,z,weight\n')
            for i in range(len(quadrature_weights)):
                point = sampler.shape.surface_position_global(sampler.samples_s[i],
                                                              sampler.samples_t[i])
                outlet.write('{},{},{},{}\n'.format(point[0],point[1],point[2],quadrature_weights[i]))

def test_z_aligned_cylinder_areas() -> None:
    '''Verify that simple cylinder inputs have correct surface area calculations'''
    rules = quadrature.RuleCache()
    TEST_COUNT = 10
    for i in range(TEST_COUNT):
        bottom_height = -1*generator.random()
        top_height = generator.random()
        radius = 0.1+generator.random()
        cylinder_sampler = Shape3DSampler(rules, Cylinder((0.,0.,bottom_height),(0.,0.,top_height),radius), 0.1)
        cylinder_area_reference = 2*math.pi*radius*(top_height-bottom_height)
        cylinder_quadrature_weights = cylinder_sampler.weights.flatten()
        cylinder_differential_areas = np.zeros_like(cylinder_quadrature_weights)
        for j in range(len(cylinder_quadrature_weights)):
            cylinder_differential_areas[j] = \
                cylinder_sampler.shape.surface_differential_area(cylinder_sampler.samples_s[j],
                                                                 cylinder_sampler.samples_t[j])
        cylinder_area_test = np.dot(cylinder_quadrature_weights,cylinder_differential_areas)
        assert real_equality(cylinder_area_reference,cylinder_area_test,geometry.TOLERANCE)
        #export_samples(sampler=cylinder_sampler,filename='cylinder_{}.csv'.format(i))

def test_z_aligned_clipped_sphere_areas() -> None:
    '''Verify that simple inputs have correct surface area calculations'''
    rules = quadrature.RuleCache()
    TEST_COUNT = 10
    for i in range(TEST_COUNT):
        center = (0.,0.,0.)
        radius = 0.1+generator.random()
        clip_bottom = ClippedSphere.ClipPlane(LeftHanded(),radius,(0.,0.,-1.),generator.random()*radius)
        clip_top = ClippedSphere.ClipPlane(RightHanded(),radius,(0.,0.,1.),generator.random()*radius)
        clipped_sphere_sampler = Shape3DSampler(rules, ClippedSphere(center,radius,[clip_bottom,clip_top]), 0.05)
        # by Archimedes' hat-box theorem https://mathworld.wolfram.com/ArchimedesHat-BoxTheorem.html
        clipped_sphere_area_reference = (2*math.pi*radius)*(clip_top.distance+clip_bottom.distance)
        clipped_sphere_quadrature_weights = clipped_sphere_sampler.weights
        clipped_sphere_differential_areas = np.zeros_like(clipped_sphere_quadrature_weights)
        for j in range(len(clipped_sphere_quadrature_weights)):
            clipped_sphere_differential_areas[j] = \
                clipped_sphere_sampler.shape.surface_differential_area(clipped_sphere_sampler.samples_s[j],
                                                                       clipped_sphere_sampler.samples_t[j])
        clipped_sphere_area_test = np.dot(clipped_sphere_quadrature_weights,clipped_sphere_differential_areas)
        assert real_equality(clipped_sphere_area_reference,clipped_sphere_area_test,0.001)
        #export_samples(sampler=clipped_sphere_sampler,filename='clipped_sphere_{}.csv'.format(i))

def compute_transformation(generator):
    azimuthal_angle = generator.uniform(-np.pi,np.pi) # Any angle to rotate about z-axis
    zenithal_angle = generator.uniform(0.,np.pi*0.25) # Stick to simple clips around poles
    return Rotation.from_rotvec([[zenithal_angle,0,0],[0,0,azimuthal_angle]])

def test_rotated_cylinder_areas() -> None:
    '''Verify that rotated cylinder inputs have correct surface area calculations.
    
    Use randomized tilt (from z) and spin (from x)'''
    rules = quadrature.RuleCache()
    TEST_COUNT = 10
    for i in range(TEST_COUNT):
        bottom_height = -1*generator.random()
        top_height = generator.random()
        radius = 0.1+generator.random()
        tilt_angle = generator.uniform(0.,0.5*np.pi)
        tilt = Rotation.from_rotvec([tilt_angle,0.,0.])
        spin_angle = generator.uniform(-np.pi,np.pi)
        spin = Rotation.from_rotvec([0.,0.,spin_angle])
        bottom_point = spin.apply(tilt.apply((0.,0.,bottom_height)))
        top_point = spin.apply(tilt.apply((0.,0.,top_height)))
        cylinder_sampler = Shape3DSampler(rules, Cylinder(bottom_point,top_point,radius), 0.1)
        cylinder_area_reference = 2*math.pi*radius*(top_height-bottom_height)
        cylinder_quadrature_weights = cylinder_sampler.weights.flatten()
        cylinder_differential_areas = np.zeros_like(cylinder_quadrature_weights)
        for j in range(len(cylinder_quadrature_weights)):
            cylinder_differential_areas[j] = \
                cylinder_sampler.shape.surface_differential_area(cylinder_sampler.samples_s[j],
                                                                 cylinder_sampler.samples_t[j])
        cylinder_area_test = np.dot(cylinder_quadrature_weights,cylinder_differential_areas)
        assert real_equality(cylinder_area_reference,cylinder_area_test,geometry.TOLERANCE)
        #export_samples(sampler=cylinder_sampler,filename='cylinder_{}.csv'.format(i))

def test_rotated_clipped_sphere_areas() -> None:
    '''Verify that tilted sphere inputs have correct surface area calculations'''
    rules = quadrature.RuleCache()
    TEST_COUNT = 10
    for i in range(TEST_COUNT):
        center = (0.,0.,0.)
        radius = generator.uniform(0.5,1.5)
        tilt_angle = generator.uniform(0.,0.25*np.pi)
        tilt = Rotation.from_rotvec([tilt_angle,0.,0.])
        spin_angle = generator.uniform(-np.pi,np.pi)
        spin = Rotation.from_rotvec([0.,0.,spin_angle])
        bottom_point = spin.apply(tilt.apply((0.,0.,-1.)))
        tilt_angle = generator.uniform(0.,0.*np.pi)
        tilt = Rotation.from_rotvec([tilt_angle,0.,0.])
        spin_angle = generator.uniform(-np.pi,np.pi)
        spin = Rotation.from_rotvec([0.,0.,spin_angle])
        top_point = spin.apply(tilt.apply((0.,0.,1.)))
        clip_bottom = ClippedSphere.ClipPlane(LeftHanded(),radius,bottom_point,generator.uniform(0.25,0.75)*radius)
        clip_top = ClippedSphere.ClipPlane(RightHanded(),radius,top_point,generator.uniform(0.5,1.)*radius)
        clipped_sphere_sampler = Shape3DSampler(rules, ClippedSphere(center,radius,[clip_bottom,clip_top]), 0.025*radius)
        # Use Archimedes' hat-box theorem https://mathworld.wolfram.com/ArchimedesHat-BoxTheorem.html
        # Consider different clipped spheres, one with the top clip only, the other with the bottom only.
        # The area associated either hemisphere is unchanged, regardless clip's orientation in it.
        # Add the two sphere's areas, and then subtract out the extra sphere surface area
        clipped_sphere_area_reference = (2*math.pi*radius)*(clip_top.distance+clip_bottom.distance)
        clipped_sphere_quadrature_weights = clipped_sphere_sampler.weights
        clipped_sphere_differential_areas = np.zeros_like(clipped_sphere_quadrature_weights)
        for j in range(len(clipped_sphere_quadrature_weights)):
            clipped_sphere_differential_areas[j] = \
                clipped_sphere_sampler.shape.surface_differential_area(clipped_sphere_sampler.samples_s[j],
                                                                       clipped_sphere_sampler.samples_t[j])
        clipped_sphere_area_test = np.dot(clipped_sphere_quadrature_weights,clipped_sphere_differential_areas)
        if not real_equality(clipped_sphere_area_reference,clipped_sphere_area_test,0.01):
            export_samples(sampler=clipped_sphere_sampler,filename='clipped_sphere_{}.csv'.format(i))
        assert real_equality(clipped_sphere_area_reference,clipped_sphere_area_test,0.01)
