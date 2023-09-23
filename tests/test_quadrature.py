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

def exp_n_x(n,x):
    '''exp(n*x); definite integral from -1 to 1 is (exp(n)-exp(-n))/n'''
    return np.exp(n*x)

from scipy import integrate

FINE_TOLERANCE = 1e-12
COARSE_TOLERANCE = 1e-8

def test_quadrature_rules():
    f1 = lambda x : exp_n_x(10,x)
    rules = quadrature.RuleCache()
    rule_g20 = rules.gauss_rule(20)
    integration_g = np.dot(rule_g20.weights,f1(rule_g20.positions))
    reference = (math.exp(10.)-math.exp(-10.))/10.
    reference_quad, reference_quad_error = integrate.quad(f1,-1.,1.)
    # Confirm that quadrature has converged to default tolerances of 1e-8
    assert real_equality(reference_quad_error,0.,COARSE_TOLERANCE)
    assert real_equality(reference_quad,reference,COARSE_TOLERANCE)
    assert real_equality(integration_g, reference, COARSE_TOLERANCE)
    rule_k41 = rules.kronrod_rule(41)
    integration_k = np.dot(rule_k41.weights,f1(rule_k41.positions))
    assert real_equality(integration_k, reference, FINE_TOLERANCE)
    assert real_equality(abs(integration_g-integration_k),0.,COARSE_TOLERANCE)
    