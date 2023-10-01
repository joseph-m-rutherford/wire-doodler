#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford
from doodler.common import Index, Real
from doodler.errors import Recoverable
from .rules import Rule1D, Rule2D

import numpy as np
import os
from pyarrow import parquet
import threading

class MissingQuadratureDefinition(Recoverable):
    '''Quadrature rules not available precomputed require raising this type'''
    pass

class RuleCache:
    '''Manages access to read-only quadrature rules.
    
    Mutex locked write access.'''

    _gauss_label = 'gauss'
    _kronrod_label = 'kronrod'
    _uniform_label = 'uniform'
    _file_name_format = os.path.join(os.path.dirname(__file__),'{rule_type}_{rule_size}.parquet')

    def __init__(self):
        self._gauss_cache = dict[Index,Rule1D]()
        self._kronrod_cache = dict[Index,Rule1D]()
        self._uniform_cache = dict[Index,Rule1D]()
        self._lock = threading.RLock()

    def _cache_rule(self,name: str, size: Index) -> None:
        '''Locks for thread safety, loads the contents from disk, stores in cache, and unlocks'''
        if name is self._uniform_label:
            # Uniform rule is trivial to compute
            with self._lock:
                delta_position = 2./size
                self._uniform_cache[size] = \
                    Rule1D(RuleCache._uniform_label, size,
                           positions=np.linspace(-1.+0.5*delta_position,1-0.5*delta_position,size),
                           weights=np.ones((size,),dtype=Real)*(2./size))
        else:
            file_caches = {
                RuleCache._gauss_label:self._gauss_cache,
                RuleCache._kronrod_label:self._kronrod_cache}
            file_name = RuleCache._file_name_format.format(rule_type=name,rule_size=size)
            if not os.path.exists(file_name):
                raise MissingQuadratureDefinition('Cannot find quadrature rule file {}'.format(file_name))
            with self._lock:
                cache = file_caches[name]
                table = parquet.read_table(file_name)
                # Copy table contents into new Rule, insert into cache
                cache[size] = Rule1D(name,size,table['position'],table['weight'])     

    def gauss_rule(self, size:Index) -> Rule1D:
        '''If the Gauss rule is in memory, return it; else find on disk and return it'''
        if size not in self._gauss_cache:
            self._cache_rule(RuleCache._gauss_label,size)
        return self._gauss_cache[size]

    def kronrod_rule(self, size:Index) -> Rule1D:
        '''If the Kronrod rule is in memory, return it; else find on disk and return it'''
        if size not in self._kronrod_cache:
            self._cache_rule(RuleCache._kronrod_label,size)
        return self._kronrod_cache[size]
    
    def uniform_rule(self, size:Index) -> Rule1D:
        '''If the uniform rule is in memory, return it; else compute and return it'''
        if size not in self._uniform_cache:
            self._cache_rule(RuleCache._uniform_label,size)
        return self._uniform_cache[size]

    def uniform_x_gauss_rule(self, size_uniform:Index, size_gauss:Index) -> Rule2D:
        '''Compute and return compound rule uniform(size_uniform) x rule_gauss(size_gauss)'''
        return Rule2D(self.uniform_rule(size_uniform),self.gauss_rule(size_gauss))
    
    def uniform_x_kronrod_rule(self, size_uniform:Index, size_kronrod:Index) -> Rule2D:
        '''Compute and return compound rule uniform(size_uniform) x rule_kronrod(size_kronrod)'''
        return Rule2D(self.uniform_rule(size_uniform),self.kronrod_rule(size_kronrod))

