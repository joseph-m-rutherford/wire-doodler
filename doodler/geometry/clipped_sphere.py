#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from .common import Shape3D, TOLERANCE
from doodler.errors import Unrecoverable
from doodler.r3 import Real, R3Vector, r3vector_copy, r3vector_equality
import copy
import math
import numpy as np

class InvalidClipPlane(Unrecoverable):
    '''Raised when a clipping plane is outside the geometry or has bad orientation'''
    pass

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
                raise InvalidClipPlane('Cannot clip at non unit direction |{}| != 1.'.format(direction))
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
        self._clips = [] # Clipping planes
        self._circle_radii = [] # Circles at each clip
        for clip in clips:
            if clip.radius > self._radius-TOLERANCE:
                raise InvalidClipPlane('Cannot clip sphere at radial distance {} >= radius {}'.format(clip.radius,self._radius))
            if clip.radius < -TOLERANCE:
                raise InvalidClipPlane('Cannot clip with negative radius')
            copied_clip = copy.deepcopy(clip)
            # snap radius to near-tangent to sphere if it's close
            if abs(copied_clip.radius-self._radius) < TOLERANCE:
                copied_clip.radius = self._radius-TOLERANCE
            self._clips.append(copied_clip)
            # Circle eqn is x^2 + y^2 = r^2 or y = +/- sqrt(r^2 - x^2)
            # On a sphere great circle perpendicular to clip plane, clip circle radius is sqrt(sphere_radius^2-clip_radius^2)
            # Limit small values to avoid taking sqrt() of a small negative number
            inward_offset_squared = max(TOLERANCE*TOLERANCE,
                                        self._radius*self._radius-copied_clip.radius*copied_clip.radius)
            circle_radius = math.sqrt(inward_offset_squared)
            self._circle_radii.append(circle_radius)

        if np.dot(self._clips[0].direction, self._clips[1].direction) > -TOLERANCE:
            raise InvalidClipPlane('Clipping planes must reside in opposite hemispheres')

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
                raise InvalidClipPlane('Clipping all of sphere requires opposite surface normals')
            # Align with global coordinates
            u_axis = R3Vector(1,0,0)
            v_axis = R3Vector(0,1,0)
            w_axis = R3Vector(0,0,1)
        else:
            w_axis = segment/height
            # Identify special case of collinear cuts
            if r3vector_equality(self._clips[0].offset,self._clips[1].offset,TOLERANCE):
                # same pole is not supported
                raise InvalidClipPlane('An entire sphere may be define by cuts on opposite poles, not the same pole')
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
                    raise InvalidClipPlane('Degenerate plane computed from sphere clip vectors')
                u_axis /= parallelogram_area
                v_axis = np.cross(w_axis,u_axis)
            super().__init__(self._center,(u_axis,v_axis,w_axis))

    @Shape3D.periodicity.getter
    def periodicity(self) -> tuple[bool]:
        '''Cylinder is periodic in s, not periodic in t; returns (True,False)'''
        return (True,False)

    @Shape3D.max_spans.getter
    def max_spans(self) -> tuple[Real]:
        '''Clipped sphere is circular in s, a zenithal arc in t; returns (2*pi*radius,radius*longest_arc_angle)'''
        min_w, max_w = self._longest_w_spans()
        horizontal_offset_min_w = max(0.,math.sqrt(self._radius*self._radius-min_w*min_w))
        horizontal_offset_max_w = max(0.,math.sqrt(self._radius*self._radius-max_w*max_w))
        angle_span = math.atan2(max_w,horizontal_offset_max_w)-math.atan2(min_w,horizontal_offset_min_w)
        # Equator may represent largest s-span
        if abs(min_w) < TOLERANCE or abs(max_w) < TOLERANCE or (min_w < 0.) != (max_w < 0.):
            return (2*math.pi*self._radius,angle_span*self._radius)
        else:
            return (2*math.pi*max(self._circle_radii),angle_span*self._radius)

    @Shape3D.min_spans.getter
    def min_spans(self) -> tuple[Real]:
        '''Clipped sphere is circular in s, a zenithal arc in t; returns (2*pi*minimum_clipped_radius,radius*shortest_arc_angle)'''
        min_w, max_w = self._shortest_w_spans()
        horizontal_offset_min_w = max(0.,math.sqrt(self._radius*self._radius-min_w*min_w))
        horizontal_offset_max_w = max(0.,math.sqrt(self._radius*self._radius-max_w*max_w))
        angle_span = math.atan2(max_w,horizontal_offset_max_w)-math.atan2(min_w,horizontal_offset_min_w)
        return (2*math.pi*min(self._circle_radii),angle_span*self._radius)

    def _shortest_w_spans(self) -> tuple[Real]:
        '''Returns min,max pair of w-values for shortest zenithal arc'''
        # N,S hemispheres each may be clipped by a tilted plane.
        # Clipping along radial vector results in a circle flattening that side.
        # Center of clip is at self.offset
        # Radius reduction in w dir is sin(theta)*circle_radius
        # Radius reduction to u,v dirs is 0
        cos_theta_0 = np.dot(self.axes[:,2],self._clips[0].direction)
        sin_theta_0_squared = max(Real(0),1.-cos_theta_0*cos_theta_0)
        sin_theta_0 = math.sqrt(sin_theta_0_squared)
        deviation_0 = self._circle_radii[0]*sin_theta_0
        cos_theta_1 = np.dot(self.axes[:,2],self._clips[1].direction)
        sin_theta_1_squared = max(Real(0),1.-cos_theta_1*cos_theta_1)
        sin_theta_1 = math.sqrt(sin_theta_1_squared)
        deviation_1 = self._circle_radii[1]*sin_theta_1
        max_w, min_w = Real(0),Real(0)
        if self._clips[0].offset[2] > Real(0):
            max_w = self._clips[0].offset[2] - deviation_0
        else:
            min_w = self._clips[0].offset[2] + deviation_0
        if self._clips[1].offset[2] > Real(0):
            max_w = self._clips[1].offset[2] - deviation_1
        else:
            min_w = self._clips[1].offset[2] + deviation_1       
        return (min_w,max_w)

    def _longest_w_spans(self) -> tuple[Real]:
        '''Returns min,max pair of w-values for longest zenithal arc'''
        # N,S hemispheres each may be clipped by a tilted plane.
        # Clipping along radial vector results in a circle flattening that side.
        # Center of clip is at self.offset
        # Radius reduction in w dir is sin(theta)*circle_radius
        # Radius reduction to u,v dirs is 0
        cos_theta_0 = np.dot(self.axes[:,2],self._clips[0].direction)
        sin_theta_0_squared = max(Real(0),1.-cos_theta_0*cos_theta_0)
        sin_theta_0 = math.sqrt(sin_theta_0_squared)
        deviation_0 = self._circle_radii[0]*sin_theta_0
        cos_theta_1 = np.dot(self.axes[:,2],self._clips[1].direction)
        sin_theta_1_squared = max(Real(0),1.-cos_theta_1*cos_theta_1)
        sin_theta_1 = math.sqrt(sin_theta_1_squared)
        deviation_1 = self._circle_radii[1]*sin_theta_1
        max_w = max(self._clips[0].offset[2] + deviation_0, self._clips[1].offset[2] + deviation_1)
        min_w = min(self._clips[0].offset[2] - deviation_0, self._clips[1].offset[2] - deviation_1)
        return (min_w,max_w)

    def bounding_box_local(self, min_uvw:R3Vector, max_uvw:R3Vector) -> None:
        '''Return bounding box in local coordinate system'''
        # All dimensions are limited by radius
        # N,S hemispheres each may be clipped by a tilted plane.
        # Clipping along radial vector results in a circle flattening that side.
        # Center of clip is at self.offset
        # Radius reduction in w dir is sin(theta)*circle_radius
        # Radius reduction to u,v dirs is 0
        min_w, max_w = self._longest_w_spans()
        min_uvw[:] = (-self._radius,-self._radius,min_w)
        max_uvw[:] = (self._radius,self._radius,max_w)