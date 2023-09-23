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

class Rule:
    '''Holds 1D quadrature rule positions and weights for positions (-1,1)'''
    def __init__(self,label,order,positions,weights):
        if isinstance(label,str):
            self._label = label
        else:
            raise InvalidQuadratureDefinition('Rule label must be a string')
        try:
            self._order = Integer(order)
            self._positions = np.array(positions,dtype=Real)
            self._weights = np.array(weights,dtype=Real)
        except Exception as e:
            raise InvalidQuadratureDefinition('Bad quadarutre rule argument: {}'.format(e))
        if order < 1:
            raise InvalidQuadratureDefinition('Quadrature order must be > 0')
        if len(self._positions.shape) != 1:
            raise InvalidQuadratureDefinition('Invalid positions array shape')
        if len(self._weights.shape) != 1:
            raise InvalidQuadratureDefinition('Invalid weights array shape')
        if self._positions.shape != self._weights.shape:
            raise InvalidQuadratureDefinition('Positions and weights arrays must both be shaped the same')
    
    @property
    def order(self):
        '''Polynomial order of quadrature rule'''
        return self._order
    @order.setter
    def order(self,value):
        raise NeverImplement('Quadrature rule order fixed after initialization.')

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

class RuleCache:
    '''Manages access to read-only quadrature rules.
    
    Mutex locked write access.'''

    _gauss_label = 'gauss'
    _kronrod_label = 'kronrod'
    _file_name_format = os.path.join(os.path.dirname(__file__),'{rule_type}_{rule_order}.parquet')

    def __init__(self):
        self._gauss_cache = dict[Index,Rule]()
        self._kronrod_cache = dict[Index,Rule]()
        self._lock = threading.RLock()

    def _cache_rule(self,name: str, order: Index) -> None:
        '''Locks for thread safety, loads the contents from disk, stores in cache, and unlocks'''
        caches = {'gauss':self._gauss_cache, 'kronrod':self._kronrod_cache}
        file_name = RuleCache._file_name_format.format(rule_type=name,rule_order=order)
        if not os.path.exists(file_name):
            raise MissingQuadratureDefinition('Cannot find quadrature rule file {}'.format(file_name))
        with self._lock:
            cache = caches[name]
            table = parquet.read_table(file_name)
            # Copy table contents into new Rule, insert into cache
            cache[order] = Rule(name,order,table['position'],table['weight'])     

    def gauss_rule(self, order:Index) -> Rule:
        '''If the Gauss rule is in memory, return it; else find on disk and return it'''
        if order not in self._gauss_cache:
            self._cache_rule(RuleCache._gauss_label,order)
        return self._gauss_cache[order]

    def kronrod_rule(self, order:Index) -> Rule:
        '''If the Kronrod rule is in memory, return it; else find on disk and return it'''
        if order not in self._kronrod_cache:
            self._cache_rule(RuleCache._kronrod_label,order)
        return self._kronrod_cache[order]

