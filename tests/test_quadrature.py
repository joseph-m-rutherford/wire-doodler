#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford

from doodler import quadrature, real_equality

import math
import numpy as np
import pytest

def test_invalid_quadrature_constructor():
    from doodler.quadrature import InvalidQuadratureDefinition, MissingQuadratureDefinition
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad label
        r = quadrature.Rule(1,1,[-0.5, 0., 0.5], [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule('bad',-1,[-0.5, 0., 0.5], [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad positions shape
        r = quadrature.Rule('bad',-1,np.zeros((3,1)), [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule('bad',-1,[-0.5, 0., 0.5], np.zeros((3,1)))
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule('bad',-1,np.zeros((3,1)), np.zeros((1,3)))
    rules = quadrature.RuleCache()
    with pytest.raises(MissingQuadratureDefinition):
        r = rules.gauss_rule(1)
    with pytest.raises(MissingQuadratureDefinition):
        r = rules.kronrod_rule(1)        

def exp_abs_n_x(n,x):
    '''exp(abs(100*x)); definite integral from -1 to 1 is 2*(exp(n)-exp(0))/n'''
    return np.exp(np.abs(n*x))

def test_quadrature_rules():
    f1 = lambda x : exp_abs_n_x(1,x)
    rules = quadrature.RuleCache()
    rule_g20 = rules.gauss_rule(20)
    assert real_equality(np.dot(rule_g20.weights,f1(rule_g20.positions)), 2.*(math.exp(1.)-1.), 1e-3)
    