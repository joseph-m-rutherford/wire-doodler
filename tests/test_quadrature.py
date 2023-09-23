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

from numpy import typing as npt

def absolute_error(a,b):
    return np.abs(a-b)

def relative_error(a,b,epsilon):
    '''Compare two values; if within epsilon of 0 magnitude, treat both as zero'''
    magnitude_a = np.abs(a)
    magnitude_b = np.abs(b)
    if max(magnitude_a, magnitude_b) < epsilon:
        return True
    return np.abs(a-b)/max(magnitude_a,magnitude_b)

def test_quadrature_rules():
    f10 = lambda x : exp_n_x(10,x)
    reference = (math.exp(10.)-math.exp(-10.))/10.
    reference_quad, reference_quad_error = integrate.quad(f10,-1.,1.)
    # Confirm that quadrature has converged to default tolerances of 1e-8
    assert real_equality(reference_quad_error,0.,COARSE_TOLERANCE)
    assert real_equality(reference_quad,reference,COARSE_TOLERANCE)
    # Having confirmed a correct refernece value, check the fixed Gaussian quadrature
    rules = quadrature.RuleCache()
    rule_g10 = rules.gauss_rule(10)
    integration_g = np.dot(rule_g10.weights,f10(rule_g10.positions))
    assert real_equality(integration_g, reference, math.sqrt(FINE_TOLERANCE))
    rule_k21 = rules.kronrod_rule(21)
    integration_k = np.dot(rule_k21.weights,f10(rule_k21.positions))
    assert real_equality(integration_k, reference, FINE_TOLERANCE)
    assert relative_error(integration_g,integration_k,COARSE_TOLERANCE) < math.sqrt(COARSE_TOLERANCE)
