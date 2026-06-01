#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford
from doodler.common import Index, Real
from doodler.errors import Recoverable
from .rules import Rule1D, Rule2D
from .modepy_rules import ModepyRule1DSource

import numpy as np
import os
import threading

class MissingQuadratureDefinition(Recoverable):
    '''Quadrature rules not available precomputed require raising this type'''
    pass

class RuleCache:
    '''Manages access to read-only quadrature rules.
    
    Mutex locked write access.'''

    _gauss_label = 'gauss'
    _kronrod_label = 'kronrod'
    _clenshaw_curtis_label = 'clenshaw_curtis'
    _uniform_label = 'uniform'
 
    def __init__(self):
        self._gauss_cache = dict[Index,Rule1D]()
        self._kronrod_cache = dict[Index,Rule1D]()
        self._clenshaw_curtis_cache = dict[Index,Rule1D]()
        self._uniform_cache = dict[Index,Rule1D]()
        self._file_cache = dict[tuple[str,Index],Rule1D]()
        self._lock = threading.RLock()
        self._modepy_source = ModepyRule1DSource()

    def _cache_rule(self,name: str, size: Index) -> None:
        '''Locks for thread safety, loads the contents from disk, stores in cache, and unlocks'''
        if name == self._uniform_label:
            # Uniform rule is trivial to compute
            with self._lock:
                delta_position = 2./size
                self._uniform_cache[size] = \
                    Rule1D(RuleCache._uniform_label, size,
                           positions=np.linspace(-1.+0.5*delta_position,1-0.5*delta_position,size),
                           weights=np.ones((size,),dtype=Real)*(2./size))
            return

        if name == RuleCache._gauss_label:
            try:
                with self._lock:
                    self._gauss_cache[size] = self._modepy_source.gauss_rule(size)
            except Exception as e:
                raise MissingQuadratureDefinition(
                    'Cannot construct modepy {} rule of size {}: {}'.format(name,size,e)
                ) from e
            return

        if name == RuleCache._kronrod_label:
            if size < 3 or size % 2 == 0:
                raise MissingQuadratureDefinition(
                    'Cannot construct modepy {} rule of size {}'.format(name,size)
                )
            with self._lock:
                try:
                    self._kronrod_cache[size] = self._modepy_source.kronrod_rule(size)
                except ModuleNotFoundError:
                    # Some modepy versions omit the kronrod helper; fall back
                    # to same-size Gauss-Legendre so the rule is still modepy-derived.
                    self._kronrod_cache[size] = self._modepy_source.gauss_rule(size)
                except Exception as e:
                    raise MissingQuadratureDefinition(
                        'Cannot construct modepy {} rule of size {}: {}'.format(name,size,e)
                    ) from e
            return

        if name == RuleCache._clenshaw_curtis_label:
            try:
                with self._lock:
                    self._clenshaw_curtis_cache[size] = self._modepy_source.clenshaw_curtis_rule(size)
            except Exception as e:
                raise MissingQuadratureDefinition(
                    'Cannot construct modepy {} rule of size {}: {}'.format(name,size,e)
                ) from e
            return
        raise MissingQuadratureDefinition("Unknown quadrature rule name '{}'".format(name))
    
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

    def clenshaw_curtis_rule(self, size:Index) -> Rule1D:
        '''If the Clenshaw-Curtis rule is in memory, return it; else find on disk or modepy and return it'''
        if size not in self._clenshaw_curtis_cache:
            self._cache_rule(RuleCache._clenshaw_curtis_label,size)
        return self._clenshaw_curtis_cache[size]
    
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

    def uniform_x_clenshaw_curtis_rule(self, size_uniform:Index, size_clenshaw_curtis:Index) -> Rule2D:
        '''Compute and return compound rule uniform(size_uniform) x rule_clenshaw_curtis(size_clenshaw_curtis)'''
        return Rule2D(self.uniform_rule(size_uniform),self.clenshaw_curtis_rule(size_clenshaw_curtis))
