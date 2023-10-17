#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from doodler import geometry, quadrature, real_equality
from doodler.geometry import ClippedSphere, Cylinder, Shape3DSampler

import math
import numpy as np
import pytest

def test_shape_areas() -> None:
    '''Verify that normal inputs have correct surface area calculations'''
    rules = quadrature.RuleCache()
    cylinder_sampler = Shape3DSampler(rules, Cylinder((0.,0.,-0.1),(0.,0.,0.9),0.5), 0.1)
    cylinder_area_reference = 2*math.pi*0.5*1
    cylinder_quadrature_weights = cylinder_sampler.weights.flatten()
    cylinder_differential_areas = np.zeros_like(cylinder_quadrature_weights)
    for i in range(len(cylinder_quadrature_weights)):
        cylinder_differential_areas[i] = \
            cylinder_sampler.shape.surface_differential_area(cylinder_sampler.samples_s[i],
                                                             cylinder_sampler.samples_t[i])
    cylinder_area_test = np.dot(cylinder_quadrature_weights,cylinder_differential_areas)
    assert real_equality(cylinder_area_reference,cylinder_area_test,geometry.TOLERANCE)

    center = (0.,0.,0.)
    radius = 1.0
    clip_radius = 0.9*radius
    clip_bottom = ClippedSphere.ClipPlane(radius,(0.,0.,-1.),clip_radius)
    clip_top = ClippedSphere.ClipPlane(radius,(0.,0.,1.),clip_radius)   
    clipped_sphere_sampler = Shape3DSampler(rules, ClippedSphere(center,radius,[clip_bottom,clip_top]), 0.1)
    # area = int_0^pi dtheta 2*pi r*r*sin_theta
    darea_dtheta = lambda theta: 2*math.pi*radius*radius*math.sin(theta)
    from scipy import integrate
    clipped_sphere_area_reference,clipped_sphere_area_reference_error = \
        integrate.quad(darea_dtheta,math.acos(clip_radius/radius),math.acos(-clip_radius/radius))
    assert clipped_sphere_area_reference_error < 1e-4
    clipped_sphere_quadrature_weights = clipped_sphere_sampler.weights
    clipped_sphere_differential_areas = np.zeros_like(clipped_sphere_quadrature_weights)
    for i in range(len(clipped_sphere_quadrature_weights)):
        clipped_sphere_differential_areas[i] = \
            clipped_sphere_sampler.shape.surface_differential_area(clipped_sphere_sampler.samples_s[i],
                                                                   clipped_sphere_sampler.samples_t[i])
    print(clipped_sphere_differential_areas)
    clipped_sphere_area_test = np.dot(clipped_sphere_quadrature_weights,clipped_sphere_differential_areas)
    assert real_equality(clipped_sphere_area_reference,clipped_sphere_area_test,0.0001)
