#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .errors import Recoverable, Unrecoverable

import numpy as np
from numpy import typing as npt

Real = np.float64
R3Vector = npt.NDArray # Shape (3,) and dtype=Real, pending https://peps.python.org/pep-0646/
R3Axes = npt.NDArray  # Shape (3,3) and dtype=Real, pending https://peps.python.org/pep-0646/

TOLERANCE = Real(1e-6)

def r3vector_copy(xyz: any) -> R3Vector:
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

def axes3d_copy(axes: any) -> R3Axes:
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

def r3vector_equality(a: R3Vector, b: R3Vector, tolerance: Real) -> bool:
    '''Compute relative difference between r3vector results'''
    if tolerance < TOLERANCE:
        raise Unrecoverable('Cannot perform relative equality with finer tolerance than module relative tolerance')
    max_length_squared = max(np.dot(a,a),np.dot(b,b))
    if max_length_squared < tolerance*tolerance: # detect comparison of values at origin
        return true
    difference = b-a
    difference_length_squared = np.dot(difference,difference)
    return difference_length_squared < max_length_squared*tolerance*tolerance
