#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

from ..common import Index
from ..errors import NeverImplement, Unrecoverable
from ..quadrature.rules import Rule2D


class WireScalarFunction:
    def __init__(self, axial_order:Index, azimuthal_order:Index, quadrature_rule:Rule2D) -> None:
        if not isinstance(quadrature_rule, Rule2D):
            raise Unrecoverable('WireScalarFunction: quadrature_rule must be Rule2D')
        self._axial_order = axial_order
        self._azimuthal_order = azimuthal_order
        self._quadrature_rule = quadrature_rule
    @property
    def axial_order(self) -> Index:
        return self._axial_order
    
    @axial_order.setter
    def axial_order(self, value: Index) -> None:
        raise NeverImplement('WireScalarFunction axial_order is immutable')
    
    @property
    def azimuthal_order(self) -> Index:
        return self._azimuthal_order
    
    @azimuthal_order.setter
    def azimuthal_order(self, value: Index) -> None:
        raise NeverImplement('WireScalarFunction azimuthal_order is immutable')

    @property
    def quadrature_rule(self) -> Rule2D:
        return self._quadrature_rule

    @quadrature_rule.setter
    def quadrature_rule(self, value: Rule2D) -> None:
        raise NeverImplement('WireScalarFunction quadrature_rule is immutable')
