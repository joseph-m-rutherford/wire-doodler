#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford
from doodler.common import Index, Integer, Real
from doodler.errors import NeverImplement, Recoverable, Unrecoverable

import numpy as np
import os
from pyarrow import parquet
import threading

class InvalidQuadratureDefinition(Unrecoverable):
    '''Nonsense inputs require raising this type.'''
    pass

class MissingQuadratureDefinition(Recoverable):
    '''Quadrature rules not available precomputed require raising this type'''
    pass

class Rule1D:
    '''Holds 1D quadrature rule positions and weights for positions (-1,1)'''
    def __init__(self,label,size,positions,weights):
        if isinstance(label,str):
            self._label = label
        else:
            raise InvalidQuadratureDefinition('Rule label must be a string')
        try:
            self._size = Integer(size)
            self._positions = np.array(positions,dtype=Real)
            self._weights = np.array(weights,dtype=Real)
        except Exception as e:
            raise InvalidQuadratureDefinition('Bad quadarutre rule argument: {}'.format(e))
        if size < 1:
            raise InvalidQuadratureDefinition('Quadrature size must be > 0')
        if len(self._positions.shape) != 1:
            raise InvalidQuadratureDefinition('Invalid positions array shape')
        if len(self._weights.shape) != 1:
            raise InvalidQuadratureDefinition('Invalid weights array shape')
        if self._positions.shape != self._weights.shape:
            raise InvalidQuadratureDefinition('Positions and weights arrays must both be shaped the same')
    
    @property
    def size(self):
        '''Polynomial size of quadrature rule'''
        return self._size
    @size.setter
    def size(self,value):
        raise NeverImplement('Quadrature rule size fixed after initialization.')

    @property
    def positions(self):
        '''Quadrature rule positions for 1D range (-1.,1.)'''
        return self._positions
    @positions.setter
    def positions(self,value):
        raise NeverImplement('Quadrature rule positions are fixed after initialization.')
    
    @property
    def weights(self):
        '''Quadrature rule weights'''
        return self._weights
    @weights.setter
    def weights(self,value):
        raise NeverImplement('Quadrature weights are fixed after initialization.')

class Rule2D:
    '''Holds 2D quadrature rule positions and weights for positions (-1,1)x(-1,1)'''
    def __init__(self,rule_1:Rule1D,rule_2:Rule1D):
        if isinstance(rule_1,Rule1D) and isinstance(rule_2,Rule1D):
            self._rule_1 = rule_1
            self._rule_2 = rule_2
            self._weights = np.outer(rule_1.weights,rule_2.weights)
            # match positions to output of np.outer()
            positions_2, positions_1 = np.meshgrid(self._rule_1.positions,self._rule_2.positions)
            self._positions = np.array([positions_1.flatten(),positions_2.flatten()],dtype=Real)
        else:
            raise InvalidQuadratureDefinition('Rule2D arguments must be Rule1D')
    
    @property
    def label(self):
        '''Label for each quadrature rule'''
        return self._rule_1.label,self.rule_2.label
    @label.setter
    def label(self):
        raise NeverImplement('Quadrature labels are fixed after initialization.')

    @property
    def size(self):
        '''Size for each quadrature rule'''
        return self._rule_1.size,self._rule_2.size
    @size.setter
    def size(self,value):
        raise NeverImplement('Quadrature rule sizes are fixed after initialization.')
    
    @property
    def positions(self):
        '''Quadrature rule positions for 2D range (-1.,1.)x(-1.,1.)'''
        return self._positions
    @positions.setter
    def positions(self,value):
        raise NeverImplement('Quadrature rule positions are fixed after initialization.')
    
    @property
    def weights(self):
        '''Quadrature rule weights for each position'''
        return self._weights
    @weights.setter
    def weights(self,value):
        raise NeverImplement('Quadrature weights are fixed after initialization.')
