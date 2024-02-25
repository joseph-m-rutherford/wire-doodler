#!/usr/bin/env python3
# Copyright (c) 2023-2024, Joseph M. Rutherford

from .common import Shape3D, InvalidTangentCoordinates, TOLERANCE, valid_tangent_coordinates
from doodler.errors import Unrecoverable, NeverImplement
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
        def __init__(self, sphere_radius:Real, direction:R3Vector, distance:Real):
            '''
            Define clipping plane as a radial unit normal and a distance along normal
            :param Real sphere_radius: radius of associated sphere
            :param R3Vector direction: unit direction interpreted from sphere center in global R3
            :param Real distance: distance along sphere radius at which to clip'''

            if abs(Real(1) - math.sqrt(np.dot(direction,direction))) > TOLERANCE:
                raise InvalidClipPlane('Cannot clip at non unit direction |{}| != 1.'.format(direction))
            self._direction = r3vector_copy(direction)

            if distance < TOLERANCE:
                raise InvalidClipPlane('Cannot clip sphere at radius < {}'.format(TOLERANCE))
            if distance > sphere_radius-TOLERANCE:
                raise InvalidClipPlane('Cannot clip near or beyond sphere radius')
            self._distance = Real(distance)

            # Circle eqn is x^2 + y^2 = r^2 or y = +/- sqrt(r^2 - x^2)
            # On a sphere great circle perpendicular to clip plane, clip circle radius is sqrt(sphere_radius^2-clip_radius^2)
            # Limit small values to avoid taking sqrt() of a small negative number
            inward_offset_squared = max(TOLERANCE*TOLERANCE,
                                        sphere_radius*sphere_radius-self.distance*self.distance)
            self._circle_radius = math.sqrt(inward_offset_squared)

            # Define plane in a local uv space with w as normal
            self._circle_u = None
            self._circle_v = None
            if abs(self._direction[2]) < TOLERANCE:
                # normal has no w component, which is an error case
                raise InvalidClipPlane('Clip plane must have non-trivial local w-direction')
            # Arbitrary alignment: best match to parent u-direction
            self._circle_u = np.cross(r3vector_copy((0.,1.,0.)),self._direction)
            self._circle_u /= math.sqrt(np.dot(self._circle_u,self._circle_u))
            self._circle_v = np.cross(self._direction,self._circle_u)
            # Local circle positions lie in a polar arc; compute maximum deviation on that circle in w-direction
            circle_delta_w = self._circle_radius*math.sqrt(self._circle_u[2]*self._circle_u[2]+self._circle_v[2]*self._circle_v[2])
            if abs(self._direction[2])-circle_delta_w < TOLERANCE:
                raise InvalidClipPlane('Clip plane passes through local w=0 plane')

        @property
        def direction(self) -> Real:
            '''Direction from sphere center for plane in sphere's uvw axes'''
            return self._direction
        
        @direction.setter
        def direction(self,value):
            raise NeverImplement('Clipping plane direction cannot be directly set')

        @property
        def distance(self) -> Real:
            '''Distance along sphere-radial clip direction for plane'''
            return self._distance

        @distance.setter
        def distance(self,value):
            raise NeverImplement('Clipping plane distance cannot be directly set')
        
        @property
        def offset(self):
            '''Convenience method returns the vector relative to sphere origin for center of circle'''
            return self._distance*self._direction

        @property
        def circle_radius(self) -> Real:
            '''Radius of clip plane circle intersecting sphere'''
            return self._circle_radius
        
        @circle_radius.setter
        def circle_radius(self,value):
            raise NeverImplement('Circle radius cannot be directly set')

        def circle_position(self,value) -> R3Vector:
            '''Position in 3D given parametric space [-1,1]'''
            arc_angle = math.pi*value
            return self._distance*self._direction + \
                self._circle_radius * (math.cos(arc_angle)*self._circle_u + math.sin(arc_angle)*self._circle_v)

    def __init__(self, center:R3Vector, radius:Real, clips: list[ClipPlane]):
        '''
        Initialize a clipped sphere

        :param R3Vector sphere_center: Position in Cartesian R3 at center of unclipped sphere in parent coordinates
        :param Real sphere_radius: Radius of unclipped sphere
        :param list[ClipPlane] clips: two clipping planes to apply to sphere in parent coordinates
        '''
        # Sanity check radius
        if radius < 10*TOLERANCE:
            raise Unrecoverable('Sphere radius must exceed {}'.format(10*TOLERANCE))
        # Sanity check contents of clips arument
        if len(clips) != 2:
            raise Unrecoverable('Sphere clip count must equal two')
        if np.dot(clips[0].direction, clips[1].direction) > -TOLERANCE:
            raise InvalidClipPlane('Clipping planes must reside in opposite hemispheres')
        for clip in clips:
            if clip.distance > radius-TOLERANCE:
                raise InvalidClipPlane('Cannot clip sphere at radial distance {} >= radius {}'.format(clip.distance,self._radius))
            if clip.distance < -TOLERANCE:
                raise InvalidClipPlane('Cannot clip with negative radius')
        # Define local coordinate system as degenerate
        u_axis = None
        v_axis = None
        w_axis = None
        # Identify w-axis direction
        segment = clips[1].offset - clips[0].offset
        height = math.sqrt(np.dot(segment,segment))
        if height < TOLERANCE:
            # Clips are both at origin is no clipping at all or degenerate
            if abs(np.dot(clips[0].direction,clips[1].direction)+1.) > TOLERANCE:
                raise InvalidClipPlane('Clipping all of sphere requires opposite surface normals')
            # Align with global coordinates
            u_axis = r3vector_copy(1,0,0)
            v_axis = r3vector_copy(0,1,0)
            w_axis = r3vector_copy(0,0,1)
        else:
            w_axis = segment/height
            # Identify special case of collinear cuts
            if r3vector_equality(clips[0].offset,clips[1].offset,TOLERANCE):
                # same pole is not supported
                raise InvalidClipPlane('An entire sphere may be define by cuts on opposite poles, not the same pole')
            if r3vector_equality(clips[0].direction,-1*clips[1].direction,TOLERANCE):
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
                # w parallel to the segment between the cut offset points
                # v at cross product of 2 cut offsets
                # u in plane of 2 cut offsets such that their position relative to center is positive in u, zer in v
                v_axis = np.cross(clips[0].offset,clips[1].offset)
                parallelogram_area = math.sqrt(np.dot(v_axis,v_axis))
                if parallelogram_area < TOLERANCE:
                    raise InvalidClipPlane('Degenerate plane computed from sphere clip vectors')
                v_axis /= parallelogram_area
                u_axis = np.cross(v_axis,w_axis)
        # Store local coordinate system info
        super().__init__(center,(u_axis,v_axis,w_axis))
        self._center = r3vector_copy(center)
        self._radius = Real(radius)
        self._clips = [] # Clipping planes to be transcribed
        self._circle_radii = [] # Circles at each clip to be computed

        # Define clipping planes in the local uvw space
        parent_to_uvw = np.array([u_axis,v_axis,w_axis])
        for clip in clips:
            self._clips.append(ClippedSphere.ClipPlane(radius,np.matmul(parent_to_uvw,clip.direction),clip.distance))
            self._circle_radii.append(clip.circle_radius)

    @Shape3D.periodicity.getter
    def periodicity(self) -> tuple[bool]:
        '''Clipped sphere is periodic in s, not periodic in t; returns (True,False)'''
        return (True,False)

    @Shape3D.max_spans.getter
    def max_spans(self) -> tuple[Real]:
        '''Clipped sphere is circular in s, a zenithal arc in t; returns (2*pi*radius,radius*longest_arc_angle)'''
        min_w, max_w = self._longest_w_spans()
        horizontal_offset_min_w = math.sqrt(max(0.,self._radius*self._radius-min_w*min_w))
        horizontal_offset_max_w = math.sqrt(max(0.,self._radius*self._radius-max_w*max_w))
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
        min_w, max_w = self._longest_w_spans()
        min_uvw[:] = (-self._radius,-self._radius,min_w)
        max_uvw[:] = (self._radius,self._radius,max_w)

    def phi_theta(self, s:Real, t:Real) -> tuple[Real]:
        '''Given tangential coordinates s,t return azimuthal angle phi, zenithal angle theta, theta span'''
        if valid_tangent_coordinates(s,t):
            # Extract theta = acos(w_component/radius) of point on circles at azimuthal coordinate s
            clip_thetas = [Real(math.acos(clip.circle_position(s)[2]/self._radius)) for clip in self._clips]
            theta_span = max(clip_thetas) - min(clip_thetas)
            midpoint_theta = np.mean(clip_thetas)
            # At this choice of s, -1<=t<=1 spans [southern_theta,northern_theta]
            theta_t = midpoint_theta + t*theta_span*0.5
            return math.pi*s,theta_t
        else:
            raise InvalidTangentCoordinates('Cannot compute local position on sphere at point ({},{})'.format(s,t))

    def surface_position_local(self, s:Real, t:Real) -> R3Vector:
        '''Clipped sphere is traversed by orthogonal coordinates in square [-1,1]x[-1,1]
        
        s argument is scaled linearly in polar angle range (-pi,pi) about w-axis
        t argument is scaled linearly on arc between clip plane circles (-height/2,height/2)'''
        phi,theta = self.phi_theta(s,t)
        return r3vector_copy((math.sin(theta)*math.cos(phi)*self._radius,math.sin(theta)*math.sin(phi)*self._radius,math.cos(theta)*self._radius))

        
    def surface_differential_area(self, s:Real, t:Real) -> Real:
        '''Differential area in orthogonal coordinates in square [-1,1]x[-1,1]

        s argument is scaled linearly in polar angle range (-pi,pi) about w-axis
        t argument is scaled linearly on arc between clip plane circles (-height/2,height/2)'''
        phi,theta = self.phi_theta(s,t)
        theta_span = abs(self.phi_theta(s,1.)[1]-self.phi_theta(s,-1.)[1])

        # for integral over 0 < theta < pi and 0 < phi < 2*pi, sphere differential area is r*r*sin(theta) dtheta dphi
        # s spans 2*pi with range of 2; multiply by pi
        # t spans theta_span with a range of 2, so multiply by theta_span/2
        return Real(math.pi*theta_span*0.5*self._radius*self._radius*math.sin(theta))