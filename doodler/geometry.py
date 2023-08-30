#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .errors import Unrecoverable, NeverImplement
from .r3 import Real, R3Axes, R3Vector, r3vector_copy, axes3d_copy
import math
import numpy as np
from numpy import typing as npt

TOLERANCE = Real(1e-6)

def flip_element(argument: R3Vector, index: int) -> None:
    '''Reverses sign of one element in array'''
    result = np.copy(argument)
    result[index] *= -1
    return result


def valid_tangent_coordinates(s,t) -> bool:
    '''Method to confirm tangent coordinates are in unit square [0,1]x[0,1]'''
    return s >= 0. and s <= 1. and t >= 0. and t <= 1.

class InvalidTangentCoordinates(Unrecoverable):
    '''Raised when a coordinate is outside the unit square [0,1]x[0,1]'''
    pass

class Shape3D:
    '''Common base class for all 3D geometry'''

    def __init__(self,origin: R3Vector, axes: R3Axes):
        '''Primarily intended as a base class for position, orientation'''
        self.origin = None # Origin is in global coordinates
        self.axes = None # Axes are orthonormal vectors in global coordinates
        try:
            self.origin = r3vector_copy(origin)
            self.axes = axes3d_copy(axes)
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
        start = r3vector_copy(start_center)
        stop = r3vector_copy(stop_center)
        segment = stop - start
        height = math.sqrt(np.dot(segment,segment))
        if height/radius <= TOLERANCE or radius/height <= TOLERANCE:
            raise Unrecoverable(''.join(['Invalid cylinder height/radius ratio ',str(height),'/',str(radius)]))
        self.height = height

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
            raise InvalidTangentCoordinates('surface_position_local() requested at invalid coordinate ({},{})'.format(s,t))

    def surface_differential_area(self,s,t) -> Real:
        '''Differential area in orthogonal coordinates in unit square [0,1] x [0,1]
        
        s,t arguments are ignored because a cylinder is constant in differential area'''
        if valid_tangent_coordinates(s,t):
            return Real(2*np.pi*self.radius*self.height)
        else:
            raise InvalidTangentCoordinates('surface_differential_area() requested at invalid coordinate ({},{})'.format(s,t))

class ClippedSphere(Shape3D):
    '''Defines locus of points on a sphere and between two clipping planes'''
    def __init__(self, start_center, start_direction, start_radius, stop_center, stop_direction, stop_radius):
        # Confirm translation of radius values
        try:
            self.start_radius = Real(start_radius)
        except Exception as e:
            raise Unrecoverable(''.join(['Invalid sphere first clip radius \'',repr(start_radius),'\':',str(e)]))
        if self.start_radius <= 0:
            raise Unrecoverable(''.join(['Invalid sphere first clip radius ', str(self.start_radius),' <= 0']))
        try:
            self.stop_radius = Real(stop_radius)
        except Exception as e:
            raise Unrecoverable(''.join(['Invalid sphere second clip radius \'',repr(stop_radius),'\':',str(e)]))
        if self.stop_radius <= 0:
            raise Unrecoverable(''.join(['Invalid sphere second clip radius ', str(self.stop_radius),' <= 0']))
        
        # Define clipping plane for start, stop faces
        self.start = r3vector_copy(start_center)
        self.start_normal = r3vector_copy(start_direction)
        start_normal_length = math.sqrt(np.dot(self.start_normal,self.start_normal))
        if abs(start_normal_length-1.) > TOLERANCE:
            raise Unrecoverable('Invalid start normal vector with non-unity length')
        else:
            self.start_normal /= start_normal_length
        self.stop = r3vector_copy(stop_center)
        self.stop_normal = r3vector_copy(stop_direction)
        stop_normal_length = math.sqrt(np.dot(self.stop_normal,self.stop_normal))
        if abs(stop_normal_length-1.) > TOLERANCE:
            raise Unrecoverable('Invalid stop normal vector with non-unity length')
        else:
            self.stop_normal /= stop_normal_length

        # The clipping planes with fixed radii define 2 circles
        #   - one in the start (South) hemisphere
        #   - one in the stop (North) hemisphere
        # TODO: Compute the corresponding unclipped sphere

        # Define local coordinate system
        segment = self.stop - self.start
        height = math.sqrt(np.dot(segment,segment))
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