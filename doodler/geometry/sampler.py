#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from doodler import common, errors, r3
from doodler.common import Index, Integer, Real
from doodler.errors import Unrecoverable, NeverImplement
from doodler.quadrature import RuleCache, Rule2D
from doodler.r3 import R3Vector

from .common import Shape3D, TOLERANCE

import math
import numpy as np

class Shape3DSampler:
    '''Sample generator for Shape3D instances'''

    MINIMUM_SAMPLE_SIZE = math.sqrt(TOLERANCE)
    MINIMUM_SAMPLE_COUNT = 3

    def __init__(self, rules:RuleCache, shape:Shape3D, separation:Real):
        '''Generates sample points for a specific shape using shared rule cache'''
        self._rules = rules
        self._samples_2d = None # Position in local R^2 (s,t) = [-1,1]x[-1,1] space
        self._samples_3d = None # Position in global R^3 (x,y,z) space
        self._quadrature_weights = None # Quadrature weight for each sample
        if not isinstance(rules,RuleCache):
            raise Unrecoverable('Cannot use quadrature rule source outside RuleCache type')
        if not isinstance(shape,Shape3D):
            raise Unrecoverable('Cannot use shapes outside Shape3D type')
        self._shape = shape
        # setter tests value, computes samples
        self.separation = separation 

    def _compute_samples(self) -> None:
        '''Revise the samples data based upon the new separation'''
        if (self._samples_2d is None) != (self._samples_3d is None):
            raise Unrecoverable('Cannot compute samples with invalid state')
        max_span_s, max_span_t = self.shape.max_spans
        # Compute number of samples for s,t dimensions
        sample_count_s = max(Shape3DSampler.MINIMUM_SAMPLE_COUNT,Integer(math.ceil(max_span_s/self.separation)))
        sample_count_t = max(Shape3DSampler.MINIMUM_SAMPLE_COUNT,Integer(math.ceil(max_span_t/self.separation)))
        # Obtain 2D quadrature rules
        lower_order_rule = self._rules.uniform_x_gauss_rule(sample_count_s,sample_count_t)
        self._quadrature_weights = lower_order_rule.weights
        # Populate 2D samples
        self._samples_2d = lower_order_rule.positions
        # Populate 3D samples
        sample_count_s,sample_count_t = lower_order_rule.size
        self._samples_3d = np.zeros((sample_count_s*sample_count_t,3),dtype=Real)
        for index in range(sample_count_s*sample_count_t):
            self._samples_3d[index,:] = \
                self._shape.surface_position_global(self._samples_2d[0,index],
                                                    self._samples_2d[1,index])
        return

    @property
    def shape(self) -> Shape3D:
        return self._shape
    
    @shape.setter
    def shape(self, value):
        raise NeverImplement('Sampler cannot change shapes after construction')

    @property
    def separation(self):
        return self._separation
    
    @separation.setter
    def separation(self, value:Real):
        if value < Shape3DSampler.MINIMUM_SAMPLE_SIZE:
            raise Unrecoverable('Sampler cannot use a separation value < {}'.format(Sampler.MINIMUM_SAMPLE_SIZE))
        self._separation = Real(value)
        self._compute_samples()

    @property
    def samples_s(self):
        return self._samples_2d[0,:].flatten()
    
    @samples_s.setter
    def samples_s(self,values):
        raise NeverImplement('Sampler cannot set samples directly; assign the quadrature rule')
    
    @property
    def samples_t(self):
        return self._samples_2d[1,:].flatten()

    @samples_t.setter
    def samples_t(self,values):
        raise NeverImplement('Sampler cannot set samples directly; set the sample density to assign the quadrature rule')
    
    @property
    def weights(self):
        return self._quadrature_weights.flatten()

    @weights.setter
    def weights(self,values):
        raise NeverImplement('Sampler cannot set quadrature weights directly; set the sample density to assign the quadrature rule')