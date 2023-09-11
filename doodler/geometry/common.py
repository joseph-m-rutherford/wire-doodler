#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from doodler import common, errors, r3
from doodler.common import Index, Real
from doodler.errors import Unrecoverable, NeverImplement
from doodler.r3 import R3Axes, R3Vector, axes3d_copy, r3vector_copy, TOLERANCE

import numpy as np

def flip_element(argument:R3Vector, index: Index) -> None:
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

class InvalidClippingPlane(Unrecoverable):
    '''Raised when a clipping plane is outside the geometry or has bad orientation'''
    pass

class Shape3D:
    '''Common base class for all 3D geometry support sampling for quadrature'''

    def __init__(self,origin:R3Vector, axes:R3Axes):
        '''Primarily intended as a base class for position, orientation'''
        self.origin = None # Origin is in global coordinates
        self.axes = None # Axes are orthonormal vectors in global coordinates
        try:
            self.origin = r3vector_copy(origin)
            self.axes = axes3d_copy(axes)
        except Exception as e:
            raise Unrecoverable(''.join(['Failure defining 3D shape:\n',str(e)]))
    
    @property
    def periodicity(self) -> tuple[bool]:
        '''Return a pair of Boolean values true if the corresponding local dimension is periodic (s, t)'''
        raise NeverImplement('Abstract property periodicity')
    
    @periodicity.setter
    def periodicity(self,value) -> None:
        raise NeverImplement('Periodicity is immutable')
        
    def point_local_to_global(self,point:R3Vector) -> R3Vector:
        '''Given u,v,w positions compute global position in x,y,z'''
        return self.origin + np.dot(self.axes.T,point)

    def bounding_box_local(self,min_uvw:R3Vector,max_uvw:R3Vector) -> None:
        '''Return bounding box in local coordinate system'''
        raise NeverImplement('Abstract method bounding_box_local()')
    
    def bounding_box_global(self,min_xyz:R3Vector,max_xyz:R3Vector) -> None:
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

    def surface_position_global(self, s:Real, t:Real) -> R3Vector:
        '''Every 3D shape surface is traversed by orthogonal coordinates in unit square [0,1] x [0,1]'''
        return self.point_local_to_global(self.surface_position_local(s,t))
    
    def surface_position_local(self, s:Real, t:Real) -> R3Vector:
        '''Every 3D shape surface is traversed by orthogonal coordinates in unit square [0,1] x [0,1]'''
        raise NeverImplement('Abstract method surface position local()')
    
    def surface_differential_area(self, s:Real,t:Real) -> Real:
        '''Differential area in orthogonal coordinates in unit square [0,1] x [0,1]'''
        raise NeverImplement('Abstract method surface_differential_area()')
  