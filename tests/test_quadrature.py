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
    # Construct common rule cache
    rules = quadrature.RuleCache()

    # Volume under function over area is iint f(rho,phi)*rho*dphi*drho
    f1_rho_phi = lambda rho,phi : rho*exp_n_rho_sin_phi(1,rho,phi)
    reference, reference_error = integrate.dblquad(f1_rho_phi,-math.pi,math.pi,0,1,epsabs=FINE_TOLERANCE,epsrel=FINE_TOLERANCE)
    # Confirm that quadrature has converged to within factor of 10 of default tolerance 1e-8
    assert reference_error < COARSE_TOLERANCE
    # Test conversion from actual range of interest to [-1,1]x[-1,1] space
    # s range from -1,1 converted to phi=-pi,pi: phi = s*pi, dphi = pi*ds
    # t range from -1,1 converted to rho=0,1 : rho = t*0.5+0.5, drho = 0.5*dt
    f1_s_t = lambda s,t : (0.5*math.pi)*(t*0.5+0.5)*exp_n_rho_sin_phi(1,t*0.5+0.5,math.pi*s)
    reference_quad, reference_quad_error = integrate.dblquad(f1_s_t,-1.,1.,-1.,1.,epsabs=FINE_TOLERANCE,epsrel=FINE_TOLERANCE)
    # Confirm that remapped quadrature also converged to within a factor of 10 of default 1e-8 tolerance
    assert reference_quad_error < COARSE_TOLERANCE
    # Confirm that remapped adaptive quadrature agrees with original
    assert real_equality(reference_quad,reference,COARSE_TOLERANCE)
    # Having confirmed a correct reference value, check fixed rule behavior
    rule_u10g10 = rules.uniform_x_gauss_rule(10,10)
    assert rule_u10g10.size == (10,10)
    values_u10g10 = f1_s_t(rule_u10g10.positions[0,:],rule_u10g10.positions[1,:])
    integral_u10g10 = np.dot(values_u10g10.flatten(),rule_u10g10.weights.flatten())
    # This u10g10 rule should be inaccurate; test error estimation using coresponding u20k21 rule
    rule_u20k21 = rules.uniform_x_kronrod_rule(20,21)
    assert rule_u20k21.size == (20,21)
    values_u20k21 = f1_s_t(rule_u20k21.positions[0,:],rule_u20k21.positions[1,:])
    integral_u20k21 = np.dot(values_u20k21.flatten(),rule_u20k21.weights.flatten())
    error_u10g10 = relative_error(integral_u10g10,reference,FINE_TOLERANCE)
    error_estimated_u10g10 = relative_error(integral_u10g10,integral_u20k21,FINE_TOLERANCE)
    # Accuracy achieved only to COARSE_TOLERANCE
    assert real_equality(error_estimated_u10g10,error_u10g10,COARSE_TOLERANCE)
    assert real_equality(integral_u10g10,reference,COARSE_TOLERANCE)

    # Move up exponent scaling by 10, recycle quadrature rules
    # Volume under function over area is iint f(rho,phi)*rho*dphi*drho
    f10_rho_phi = lambda rho,phi : rho*exp_n_rho_sin_phi(10,rho,phi)
    reference, reference_error = integrate.dblquad(f10_rho_phi,-math.pi,math.pi,0,1,epsabs=FINE_TOLERANCE,epsrel=FINE_TOLERANCE)
    # Confirm that quadrature has converged to within factor of 100 of default tolerance 1e-8
    assert reference_error < 100*COARSE_TOLERANCE
    # Test conversion from actual range of interest to [-1,1]x[-1,1] space
    # s range from -1,1 converted to phi=-pi,pi: phi = s*pi, dphi = pi*ds
    # t range from -1,1 converted to rho=0,1 : rho = t*0.5+0.5, drho = 0.5*dt
    f10_s_t = lambda s,t : (0.5*math.pi)*(t*0.5+0.5)*exp_n_rho_sin_phi(10,t*0.5+0.5,math.pi*s)
    reference_quad, reference_quad_error = integrate.dblquad(f10_s_t,-1.,1.,-1.,1.,epsabs=FINE_TOLERANCE,epsrel=FINE_TOLERANCE)
    # Confirm that remapped quadrature also converged to within a factor of 100 of default 1.5e-8 tolerance
    assert reference_quad_error < COARSE_TOLERANCE
    # Confirm that remapped adaptive quadrature agrees with original
    assert real_equality(reference_quad,reference,COARSE_TOLERANCE)
    # Having confirmed a correct reference value, check fixed rule behavior
    values_u10g10 = f10_s_t(rule_u10g10.positions[0,:],rule_u10g10.positions[1,:])
    integral_u10g10 = np.dot(values_u10g10.flatten(),rule_u10g10.weights.flatten())
    # This u10g10 rule should be inaccurate; test error estimation using coresponding u20k21 rule
    values_u20k21 = f10_s_t(rule_u20k21.positions[0,:],rule_u20k21.positions[1,:])
    integral_u20k21 = np.dot(values_u20k21.flatten(),rule_u20k21.weights.flatten())
    error_u10g10 = relative_error(integral_u10g10,reference,FINE_TOLERANCE)
    error_estimated_u10g10 = relative_error(integral_u10g10,integral_u20k21,FINE_TOLERANCE)
    # Error estimate is larger; use very coarse tolerance
    assert real_equality(error_estimated_u10g10,error_u10g10,math.sqrt(COARSE_TOLERANCE))
    # Low order solution is not adequate
    assert not real_equality(integral_u10g10,reference,COARSE_TOLERANCE)
    # High order solution in very coarse adequate
    assert real_equality(integral_u20k21,reference,math.sqrt(COARSE_TOLERANCE))
    # Move up in rule orders
    rule_u30g30 = rules.uniform_x_gauss_rule(30,30)
    assert rule_u30g30.size == (30,30)
    values_u30g30 = f10_s_t(rule_u30g30.positions[0,:],rule_u30g30.positions[1,:])
    integral_u30g30 = np.dot(values_u30g30.flatten(),rule_u30g30.weights.flatten())
    # This u30g30 rule should be inaccurate; test error estimation using coresponding u60k61 rule
    rule_u60k61 = rules.uniform_x_kronrod_rule(60,61)
    assert rule_u60k61.size == (60,61)
    values_u60k61 = f10_s_t(rule_u60k61.positions[0,:],rule_u60k61.positions[1,:])
    integral_u60k61 = np.dot(values_u60k61.flatten(),rule_u60k61.weights.flatten())
    error_u30g30 = relative_error(integral_u30g30,reference,FINE_TOLERANCE)
    error_estimated_u30g30 = relative_error(integral_u30g30,integral_u60k61,FINE_TOLERANCE)
    # Error estimate is tiny, use fine tolerance
    assert real_equality(error_estimated_u30g30,error_u30g30,COARSE_TOLERANCE)
    # Reference value is accurate only to within fine tolerance
    assert real_equality(integral_u30g30,reference,FINE_TOLERANCE)
    # Higher order solution is within tolerance
    assert real_equality(integral_u60k61,reference,FINE_TOLERANCE)