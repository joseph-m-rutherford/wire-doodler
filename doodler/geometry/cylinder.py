#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .common import Shape3D, InvalidTangentCoordinates, TOLERANCE, valid_tangent_coordinates

from doodler.errors import Unrecoverable
from doodler.r3 import Real, R3Vector, r3vector_copy
import math
import numpy as np

class Cylinder(Shape3D):
    '''Defines the locus of points a fixed axial radius from a line segment'''
    def __init__(self,start_center:R3Vector, stop_center:R3Vector, radius:Real):
        try:
            self._radius = Real(radius)
        except Exception as e:
            raise Unrecoverable(''.join(['Invalid cylinder radius \'',repr(radius),'\':',str(e)]))
        if self._radius <= 0:
            raise Unrecoverable(''.join(['Invalid cylinder radius ', str(self._radius),' <= 0']))
        start = r3vector_copy(start_center)
        stop = r3vector_copy(stop_center)
        segment = stop - start
        height = math.sqrt(np.dot(segment,segment))
        if height/radius <= TOLERANCE or radius/height <= TOLERANCE:
            raise Unrecoverable(''.join(['Invalid cylinder height/radius ratio ',str(height),'/',str(radius)]))
        self._height = height

        w_axis = segment/height
        # Pick least component of w for computing u,v axes
        v_segment = None
        if abs(w_axis[0]) <= abs(w_axis[1]) and abs(w_axis[0]) <= abs(w_axis[2]):
            v_segment = r3vector_copy((0,w_axis[2],-w_axis[1]))
        elif abs(w_axis[1]) <= abs(w_axis[2]) and abs(w_axis[1]) <= abs(w_axis[0]):
            v_segment = r3vector_copy((-w_axis[2],0,w_axis[0]))
        else: # abs(w_axis[2]) <= abs(w_axis[0]) and abs(w_axis[2]) <= abs(w_axis[1]) or they are equal
            v_segment = r3vector_copy((w_axis[1],-w_axis[0],0))
        v_axis = v_segment/math.sqrt(np.dot(v_segment,v_segment))
        u_axis = np.cross(v_axis,w_axis)
        super().__init__((start+stop)/2,(u_axis,v_axis,w_axis))
    
    @Shape3D.periodicity.getter
    def periodicity(self) -> tuple[bool]:
        '''Cylinder is periodic in s, not periodic in t; returns (True,False)'''
        return (True,False)

    def bounding_box_local(self, min_uvw:R3Vector, max_uvw:R3Vector) -> None:
        '''Return bounding box in local coordinate system'''
        min_uvw[:] = (-self._radius,-self._radius,-self._height/2)
        max_uvw[:] = (self._radius,self._radius,self._height/2)
    
    def surface_position_local(self, s:Real, t:Real) -> R3Vector:
        '''Cylinder is traversed by orthogonal coordinates in unit square [0,1] x [0,1]
        
        s argument is scaled linearly in polar angle range [0,2*pi]
        t argument is scaled linearly in height position [0,height]'''
        if valid_tangent_coordinates(s,t):
            return np.array([np.cos(2*np.pi*s)*self._radius,np.sin(2*np.pi*s)*self._radius,self._height*(t-0.5)],dtype=Real)
        else:
            raise InvalidTangentCoordinates('surface_position_local() requested at invalid coordinate ({},{})'.format(s,t))

    def surface_differential_area(self, s:Real, t:Real) -> Real:
        '''Differential area in orthogonal coordinates in unit square [0,1] x [0,1]
        
        s,t arguments are ignored because a cylinder is constant in differential area'''
        if valid_tangent_coordinates(s,t):
            return Real(2*np.pi*self._radius*self._height)
        else:
            raise InvalidTangentCoordinates('surface_differential_area() requested at invalid coordinate ({},{})'.format(s,t))
