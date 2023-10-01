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
        r = quadrature.Rule1D(1,1,[-0.5, 0., 0.5], [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule1D('bad',-1,[-0.5, 0., 0.5], [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad positions shape
        r = quadrature.Rule1D('bad',-1,np.zeros((3,1)), [1./3., 1./3, 1./3.])
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule1D('bad',-1,[-0.5, 0., 0.5], np.zeros((3,1)))
    with pytest.raises(InvalidQuadratureDefinition):
        # Bad order
        r = quadrature.Rule1D('bad',-1,np.zeros((3,1)), np.zeros((1,3)))
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

def test_1D_quadrature_rules():
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

def exp_n_rho_sin_phi(n:float, rho:float,phi:float):
    return np.exp(n*rho*np.sin(phi))

def test_2D_quadrature_rules():
    # Volume under function over area is iint f(rho,phi)*rho*dphi*drho
    f1_rho_phi = lambda rho,phi : rho*exp_n_rho_sin_phi(1,rho,phi)
    reference, reference_error = integrate.dblquad(f1_rho_phi,-math.pi,math.pi,0,1)
    # Confirm that quadrature has converged to within factor of 10 of default tolerance 1e-8
    assert real_equality(reference_error,0.,10*COARSE_TOLERANCE)
    # Test conversion from actual range of interest to (-1,1)x(-1,1) space
    # s range from -1,1 converted to phi=-pi,pi: phi = s*pi, dphi = pi*ds
    # t range from -1,1 converted to rho=0,1 : rho = t*0.5+0.5, drho = 0.5*dt
    f1_s_t = lambda s,t : (0.5*math.pi)*(t*0.5+0.5)*exp_n_rho_sin_phi(1,t*0.5+0.5,math.pi*s)
    reference_quad, reference_quad_error = integrate.dblquad(f1_s_t,-1.,1.,-1.,1.)
    # Confirm that remapped quadrature also converged to within a factor of 10 of default 1e-8 tolerance
    assert real_equality(reference_quad_error,0.,10*COARSE_TOLERANCE)
    # Confirm that remapped adaptive quadrature agrees with original
    assert real_equality(reference_quad,reference,COARSE_TOLERANCE)
    # Having confirmed a correct reference value, check fixed rule behavior
    rules = quadrature.RuleCache()
    rule_u20g20 = rules.uniform_x_gauss_rule(20,20)
    assert rule_u20g20.size == (20,20)
    values_u20g20 = f1_s_t(rule_u20g20.positions[0,:],rule_u20g20.positions[1,:])
    integral_u20g20 = np.dot(values_u20g20.flatten(),rule_u20g20.weights.flatten())
    # This u10g10 rule should be inaccurate; test error estimation using coresponding u20k21 rule
    rule_u40k41 = rules.uniform_x_kronrod_rule(40,41)
    assert rule_u40k41.size == (40,41)
    values_u40k41 = f1_s_t(rule_u40k41.positions[0,:],rule_u40k41.positions[1,:])
    integral_u40k41 = np.dot(values_u40k41.flatten(),rule_u40k41.weights.flatten())
    error_u20g20 = relative_error(integral_u20g20,reference,FINE_TOLERANCE)
    error_estimated_u20g20 = relative_error(integral_u20g20,integral_u40k41,FINE_TOLERANCE)
    # error estimates should both be < 0.1
    assert absolute_error(error_estimated_u20g20,error_u20g20) < 0.1
    # error estimates should both be > 0.01
    assert absolute_error(error_estimated_u20g20,error_u20g20) > 0.01
