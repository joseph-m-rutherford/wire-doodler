#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from errors import Recoverable, Unrecoverable

import math
import numpy as np

FloatType = np.float64
RELATIVE_TOLERANCE = FloatType(1e-6)

def vector3d(xyz):
    '''Cast the argument into a length-3 array of floating point data'''
    result = None
    try:
        result = np.array(xyz,dtype=FloatType)
        if result.shape != (3,):
            # raise Recoverable because something could be done here in the future
            raise Recoverable('Argument was not interpreted into length-3 array')
    except Exception as e:
        raise Unrecoverable(''.join(['Cannot instantiate a 3D point structure with argument: \'',
                                     str(xyz),':\n\t',str(e)]))
    return result

def vector3d_equality(a,b,tolerance):
    '''Compute relative difference between vector3d results'''
    if tolerance < RELATIVE_TOLERANCE:
        raise Unrecoverable('Cannot perform relative equality with finer tolerance than module relative tolerance')
    aa = vector3d(a)
    bb = vector3d(b)
    max_length_squared = max(np.dot(aa,aa),np.dot(bb,bb))
    if max_length_squared < tolerance*tolerance: # detect comparison of values at origin
        return true
    difference = bb-aa
    difference_length_squared = np.dot(difference,difference)
    return difference_length_squared < max_length_squared*tolerance*tolerance

class Shape3D:
    '''Common base class for all 3D geometry'''

    def __init__(self,origin,axes):
        self.origin = None # Origin is in global coordinates
        self.axes = None # Axes are orthonormal vectors in global coordinates
        try:
            self.origin = np.array(origin,dtype=FloatType)
            if self.origin.shape != (3,): # raise Recoverable because something could be done here in the future
                raise Recoverable(''.join(['Argument for origin was not interpreted into a length-3 array:',str(origin)]))
            self.axes = np.array(axes,dtype=FloatType)
            if self.axes.shape != (3,3):
                # raise Recoverable because something could be done here in the future
                raise Recoverable(''.join(['Argument for axes was not interpreted into 3 length-3 arrays:',str(axes)]))
            bad_uv = max(abs(np.dot(self.axes[0],self.axes[1])), np.max(np.abs(self.axes[2]-np.cross(self.axes[0],self.axes[1])))) > RELATIVE_TOLERANCE
            bad_vw = max(abs(np.dot(self.axes[1],self.axes[2])), np.max(np.abs(self.axes[0]-np.cross(self.axes[1],self.axes[2])))) > RELATIVE_TOLERANCE
            bad_wu = max(abs(np.dot(self.axes[2],self.axes[0])), np.max(np.abs(self.axes[1]-np.cross(self.axes[2],self.axes[0])))) > RELATIVE_TOLERANCE
            if bad_uv or bad_vw or bad_wu:
                raise Unrecoverable(''.join(['Axes argument was not found to be 3 orthnormal vectors:',str(axes)]))
        except Exception as e:
            raise Unrecoverable(''.join(['Failure defining 3D shape:\n',str(e)]))
        
class Cylinder(Shape3D):
    '''Defines the locus of points a fixed axial radius from a line segment'''
    def __init__(self,start_center, stop_center, radius):
        try:
            self.radius = FloatType(radius)
        except Exception as e:
            raise Unrecoverable(''.join(['Invalid cylinder radius \'',repr(radius),'\':',str(e)]))
        if self.radius <= 0:
            raise Unrecoverable(''.join(['Invalid cylinder radius ', str(self.radius),' <= 0']))
        start = vector3d(start_center)
        stop = vector3d(stop_center)
        segment = stop - start
        height = math.sqrt(np.dot(segment,segment))
        if height/radius <= RELATIVE_TOLERANCE or radius/height <= RELATIVE_TOLERANCE:
            raise Unrecoverable(''.join(['Invalid cylinder height/radius ratio ',str(height),'/',str(radius)]))
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
