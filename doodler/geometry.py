#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from errors import Recoverable, Unrecoverable, NeverImplement

import math
import numpy as np
from numpy import typing as npt
import typing

Real = np.float64
R3Vector = npt.NDArray # Shape (3,) and dtype=Real, pending https://peps.python.org/pep-0646/
R3Axes = npt.NDArray  # Shape (3,3) and dtype=Real, pending https://peps.python.org/pep-0646/

TOLERANCE = Real(1e-6)

def vector3d(xyz: any) -> R3Vector:
    '''Cast the argument into a length-3 array of floating point data'''
    result = None
    try:
        result = R3Vector((3,),dtype=Real)
        input = np.asarray(xyz,dtype=Real)
        if input.shape != (3,):
            # raise Recoverable because something could be done here in the future
            raise Recoverable('Argument was not interpreted into length-3 array')
        else:
            result[:] = input
    except Exception as e:
        raise Unrecoverable(''.join(['Cannot instantiate a 3D vector structure with argument: \'',
                                     str(xyz),':\n\t',str(e)]))
    return result

def axes3d(axes: any) -> R3Axes:
    result = None
    try:
        result = R3Axes((3,3),dtype=Real)
        input = np.asarray(axes,dtype=Real)
        if input.shape != (3,3):
            # raise Recoverable because something could be done here in the future
            raise Recoverable('Argument was not interpreted into 3 length-3 arrays')
        else:
            bad_uv = max(abs(np.dot(input[0],input[1])), np.max(np.abs(input[2]-np.cross(input[0],input[1])))) > TOLERANCE
            bad_vw = max(abs(np.dot(input[1],input[2])), np.max(np.abs(input[0]-np.cross(input[1],input[2])))) > TOLERANCE
            bad_wu = max(abs(np.dot(input[2],input[0])), np.max(np.abs(input[1]-np.cross(input[2],input[0])))) > TOLERANCE
            if bad_uv or bad_vw or bad_wu:
                raise Unrecoverable(''.join(['Axes argument was not found to be 3 orthnormal vectors:',str(axes)]))
            else:
                result[:] = input
    except Exception as e:
        raise Unrecoverable(''.join(['Cannot instantiate a 3D axes structure with argument: \'',
                                     str(axes),':\n\t',str(e)]))
    return result            

def real_equality(a: Real, b: Real, tolerance: Real) -> bool:
    '''For values near the origin, use absolute comparison; otherwise do relative comparison'''
    if abs(a) < tolerance and abs(b) < tolerance:
        return True
    else:
        return abs(a-b)/max(abs(a),abs(b)) < tolerance

def vector3d_equality(a: R3Vector, b: R3Vector, tolerance: Real) -> bool:
    '''Compute relative difference between vector3d results'''
    if tolerance < TOLERANCE:
        raise Unrecoverable('Cannot perform relative equality with finer tolerance than module relative tolerance')
    max_length_squared = max(np.dot(a,a),np.dot(b,b))
    if max_length_squared < tolerance*tolerance: # detect comparison of values at origin
        return true
    difference = b-a
    difference_length_squared = np.dot(difference,difference)
    return difference_length_squared < max_length_squared*tolerance*tolerance


def flip_element(argument: R3Vector, index: int) -> None:
    '''Reverses sign of one element in array'''
    result = np.copy(argument)
    result[index] *= -1
    return result


def valid_tangent_coordinates(s,t) -> bool:
    '''Method to confirm tangent coordinates are in unit square [0,1]x[0,1]'''
    return s >= 0. and s <= 1. and t >= 0. and t <= 1.


class Shape3D:
    '''Common base class for all 3D geometry'''

    def __init__(self,origin: R3Vector, axes: R3Axes):
        '''Primarily intended as a base class for position, orientation'''
        self.origin = None # Origin is in global coordinates
        self.axes = None # Axes are orthonormal vectors in global coordinates
        try:
            self.origin = vector3d(origin)
            self.axes = axes3d(axes)
        except Exception as e:
            raise Unrecoverable(''.join(['Failure defining 3D shape:\n',str(e)]))
        
    def point_local_to_global(self,point) -> R3Vector:
        '''Given u,v,w positions compute global position in x,y,z'''
        return self.origin + np.dot(self.axes.T,point)

    def bounding_box_local(self,min_uvw,max_uvw) -> None:
        '''Return bounding box in local coordinate system'''
        raise NeverImplement('Abstract method bounding_box_local()')
    
    def bounding_box_global(self,min_xyz,max_xyz) -> None:
        '''Return bounding box in global coordinate system'''
        # This could be implemented for a minimal bounding box using specific knowledge of subclass.
        # For the time being, accept expanding the global bounding box using outer corners of local voxel.
        min_uvw = np.zeros((3,),dtype=Real)
        max_uvw = np.zeros((3,),dtype=Real)
        self.bounding_box_local(min_uvw,max_uvw)
        # Compute 8 corners of voxel in local coordinates
        centroid_uvw = (min_uvw+max_uvw)/2
        span_uvw = (max_uvw-min_uvw)

        vertices_uvw = np.zeros((3,8),dtype=Real)
        vertices_uvw[:,0] = -span_uvw/2
        vertices_uvw[:,1] = flip_element(vertices_uvw[:,0],0)
        vertices_uvw[:,2] = flip_element(vertices_uvw[:,0],1)
        vertices_uvw[:,3] = flip_element(vertices_uvw[:,0],2)
        vertices_uvw[:,4:] = -vertices_uvw[:,:4]
        for column in range(8):
            vertices_uvw[:,column] += centroid_uvw
        # Convert from local to global coordinates
        vertices_xyz = np.zeros((3,8),dtype=Real)
        for column in range(8):
            vertices_xyz[:,column] = self.point_local_to_global(vertices_uvw[:,column])
        # Extract min/max from all 8 vertices
        min_xyz[0] = np.min(vertices_xyz[0,:])
        max_xyz[0] = np.max(vertices_xyz[0,:])
        min_xyz[1] = np.min(vertices_xyz[1,:])
        max_xyz[1] = np.max(vertices_xyz[1,:])
        min_xyz[2] = np.min(vertices_xyz[2,:])
        max_xyz[2] = np.max(vertices_xyz[2,:])

    def surface_position_global(self,s,t) -> R3Vector:
        '''Every 3D shape surface is traversed by orthogonal coordinates in unit square [0,1] x [0,1]'''
        return self.point_local_to_global(self.surface_position_local(s,t))
    
    def surface_position_local(self,s,t) -> R3Vector:
        '''Every 3D shape surface is traversed by orthogonal coordinates in unit square [0,1] x [0,1]'''
        raise NeverImplement('Abstract method surface position local()')
    
    def surface_differential_area(self,s,t) -> Real:
        '''Differential area in orthogonal coordinates in unit square [0,1] x [0,1]'''
        raise NeverImplement('Abstract method surface_differential_area()')
    
    
class Cylinder(Shape3D):
    '''Defines the locus of points a fixed axial radius from a line segment'''
    def __init__(self,start_center, stop_center, radius):
        try:
            self.radius = Real(radius)
        except Exception as e:
            raise Unrecoverable(''.join(['Invalid cylinder radius \'',repr(radius),'\':',str(e)]))
        if self.radius <= 0:
            raise Unrecoverable(''.join(['Invalid cylinder radius ', str(self.radius),' <= 0']))
        start = vector3d(start_center)
        stop = vector3d(stop_center)
        segment = stop - start
        height = math.sqrt(np.dot(segment,segment))
        if height/radius <= TOLERANCE or radius/height <= TOLERANCE:
            raise Unrecoverable(''.join(['Invalid cylinder height/radius ratio ',str(height),'/',str(radius)]))
        self.height = height

        w_axis = segment/height
        # Pick least component of w for computing u,v axes
        v_segment = None
        if abs(w_axis[0]) <= abs(w_axis[1]) and abs(w_axis[0]) <= abs(w_axis[2]):
            v_segment = vector3d((0,w_axis[2],-w_axis[1]))
        elif abs(w_axis[1]) <= abs(w_axis[2]) and abs(w_axis[1]) <= abs(w_axis[0]):
            v_segment = vector3d((-w_axis[2],0,w_axis[0]))
        else: # abs(w_axis[2]) <= abs(w_axis[0]) and abs(w_axis[2]) <= abs(w_axis[1]) or they are equal
            v_segment = vector3d((w_axis[1],-w_axis[0],0))
        v_axis = v_segment/math.sqrt(np.dot(v_segment,v_segment))
        u_axis = np.cross(v_axis,w_axis)
        super().__init__((start+stop)/2,(u_axis,v_axis,w_axis))

    def bounding_box_local(self,min_uvw,max_uvw) -> None:
        '''Return bounding box in local coordinate system'''
        min_uvw[:] = (-self.radius,-self.radius,-self.height/2)
        max_uvw[:] = (self.radius,self.radius,self.height/2)
    
    def surface_position_local(self,s,t) -> R3Vector:
        '''Cylinder is traversed by orthogonal coordinates in unit square [0,1] x [0,1]
        
        s argument is scaled linearly in polar angle range [0,2*pi]
        t argument is scaled linearly in height position [0,height]'''
        if valid_tangent_coordinates(s,t):
            return np.array([np.cos(2*np.pi*s)*self.radius,np.sin(2*np.pi*s)*self.radius,self.height*(t-0.5)],dtype=Real)
        else:
            raise Unrecoverable('surface_position_local() requested at invalid coordinate ({},{})'.format(s,t))

    def surface_differential_area(self,s,t) -> Real:
        '''Differential area in orthogonal coordinates in unit square [0,1] x [0,1]
        
        s,t arguments are ignored because a cylinder is constant in differential area'''
        return 2*np.pi*self.radius*self.height