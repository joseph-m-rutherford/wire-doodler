#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .common import Shape3D, InvalidClippingPlane, TOLERANCE
from doodler.errors import Unrecoverable
from doodler.r3 import Real, R3Vector, r3vector_copy, r3vector_equality
import copy
import math
import numpy as np

class ClippedSphere(Shape3D):
    '''Defines locus of points on a sphere and between two clipping planes'''
    SPHERE_OVERSIZE_SCALE = 1.5

    class ClipPlane:
        '''Clipping planes defined in CartesianR3 for use on a sphere'''
        def __init__(self, direction:R3Vector, radius:Real):
            '''
            Define clipping plane as a radial unit normal and a distance along normal

            :param R3Vector direction: unit direction interpreted from sphere center in global R3
            :param Real radius: distance along sphere radius at which to clip'''
            self._direction = r3vector_copy(direction) # Uses setter method for sanity checks
            self._radius = Real(radius) # Uses setter method for sanity checks

        @property
        def direction(self) -> R3Vector:
            '''Unit direction for clip interpreted in global R3'''
            return self._direction
        
        @direction.setter
        def direction(self, direction:R3Vector) -> None:
            if abs(Real(1) - math.sqrt(np.dot(direction,direction))) > TOLERANCE:
                raise Unrecoverable('Cannot clip at non unit direction |{}| != 1.'.format(direction))
            self._direction = direction
        
        @property
        def radius(self) -> Real:
            '''Distance from sphere center to clip with plane'''
            return self._radius
        
        @radius.setter
        def radius(self,radius:Real) -> None:
            if radius < Real(0):
                raise Unrecoverable('Cannot clip sphere at radius < 0')
            # Snap radius to 0 if within TOLERANCE
            if radius < TOLERANCE:
                self._radius = Real(0)
            else:
                self._radius = radius

        @property
        def offset(self) -> R3Vector:
            '''Position in global coordinates: direction*radius'''
            return self._direction*self._radius

    def __init__(self, center:R3Vector, radius:Real, clips: list[ClipPlane]):
        '''
        Initialize a clipped sphere

        :param R3Vector sphere_center: Position in Cartesian R3 at center of unclipped sphere
        :param Real sphere_radius: Radius of unclipped sphere
        :param list[ClipPlane] clips: two clipping planes to apply to sphere
        '''
        self._center = r3vector_copy(center)
        self._radius = Real(radius)
        
        # Sanity check contents of clips arument
        if len(clips) != 2:
            raise Unrecoverable('Sphere clip count must equal two')
        self._clips = []
        for clip in clips:
            if clip.radius > self._radius+TOLERANCE:
                raise InvalidClippingPlane('Cannot clip sphere at radial distance {} > radius {}'.format(clip.radius,self._radius))
            if clip.radius < -TOLERANCE:
                raise InvalidClippingPlane('Cannot clip with negative radius')
            copied_clip = copy.deepcopy(clip)
            # snap radius to tangent to sphere if it's close
            if abs(copied_clip.radius-self._radius) < TOLERANCE:
                copied_clip.radius = self._radius
            self._clips += [copied_clip]

        if np.dot(self._clips[0].direction, self._clips[1].direction) > -TOLERANCE:
            raise InvalidClippingPlane('Clipping planes must reside in opposite hemispheres')

        # Define local coordinate system as degenerate
        u_axis = None
        v_axis = None
        w_axis = None
        # Identify w-axis direction
        segment = self._clips[1].offset - self._clips[0].offset
        height = math.sqrt(np.dot(segment,segment))
        if height < TOLERANCE:
            # Clips are both at origin is no clipping at all or degenerate
            if abs(np.dot(self._clips[0].direction,self._clips[1].direction)+1.) > TOLERANCE:
                raise InvalidClippingPlane('Clipping all of sphere requires opposite surface normals')
            # Align with global coordinates
            u_axis = R3Vector(1,0,0)
            v_axis = R3Vector(0,1,0)
            w_axis = R3Vector(0,0,1)
        else:
            w_axis = segment/height
            # Identify special case of collinear cuts
            if r3vector_equality(self._clips[0].offset,self._clips[1].offset,TOLERANCE):
                # same pole is not supported
                raise InvalidClippingPlane('An entire sphere may be define by cuts on opposite poles, not the same pole')
            if r3vector_equality(self._clips[0].offset,-1*self._clips[1].offset,TOLERANCE):
                # opposite poles is supported
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
            else:
                # Define coordinate system using plane of two cuts
                # u perpendicular to plane of 2 cut offsets
                # v in plane of 2 cut offsets such that their position relative to center is positive in v
                # w parallel to the segment between the cut offset points
                u_axis = np.cross(self._clips[0].offset,self._clips[1].offset)
                parallelogram_area = math.sqrt(np.dot(u_axis,u_axis))
                if parallelogram_area < TOLERANCE:
                    raise InvalidClippingPlane('Degenerate plane computed from sphere clip vectors')
                u_axis /= parallelogram_area
                v_axis = np.cross(w_axis,u_axis)
            super().__init__(self._center,(u_axis,v_axis,w_axis))

    @property
    def radius(self):
        '''Unclipped sphere radius'''
        return self._radius
    
    @property
    def center(self):
        '''Global center point of unclipped sphere used as origin in local coordinate'''
        return self._center

    def bounding_box_local(self, min_uvw:R3Vector, max_uvw:R3Vector) -> None:
        '''Return bounding box in local coordinate system'''
        # All dimensions are limited by self.radius
        # N,S hemispheres each may be clipped by a tilted plane.
        # Clipping along radial vector results in a circle flattening that side.
        # Center of clip is at self.offset
        # Addition to w dir is sin(theta)*circle_radius
        # Addition to u,v dirs is cos_theta*circle_radius

        min_uvw[:] = (-self.radius,-self.radius,-self._radius)
        max_uvw[:] = (self.radius,self.radius,self.radius)