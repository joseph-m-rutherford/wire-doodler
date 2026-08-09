#!/usr/bin/env python3
# Copyright (c) 2023-2026, Joseph M. Rutherford

from collections.abc import Callable
from enum import Enum

import numpy as np

from ..common import Index, Real
from ..errors import NeverImplement
from ..errors import NotYetImplemented
from ..errors import Unrecoverable
from ..discretization.function_supports import FunctionSupports
from ..discretization.wire_functions import WireScalarFunction
from ..discretization.wire_mesh import WireMesh3D
from ..quadrature.rules import Rule1D
from ..r3 import vector_equality

class FillChoice(str, Enum):
    MASS = 'mass'
    STIFFNESS = 'stiffness'


class WireMesh3DFill:
    def __init__(self, frequency: Real) -> None:
        frequency = Real(frequency)
        if not np.isfinite(frequency):
            raise Unrecoverable('WireMesh3DFill: frequency must be finite')
        self._frequency = frequency

    @property
    def frequency(self) -> Real:
        return self._frequency

    @frequency.setter
    def frequency(self, value) -> None:
        raise NeverImplement('WireMesh3DFill frequency is immutable')

    @staticmethod
    def _local_entry(
        fill_choice: FillChoice,
        length: Real,
        test_local_index: int,
        basis_local_index: int,
        axial_rule: Rule1D,
    ) -> Real:
        positions = axial_rule.positions
        weights = axial_rule.weights

        phi = np.array([(1.0 - positions) / 2.0, (1.0 + positions) / 2.0], dtype=Real)
        dphi = np.array([-0.5, 0.5], dtype=Real)

        if fill_choice == FillChoice.MASS:
            integrand = phi[test_local_index] * phi[basis_local_index]
            return Real((length / Real(2.0)) * np.sum(weights * integrand))

        if fill_choice == FillChoice.STIFFNESS:
            integrand = dphi[test_local_index] * dphi[basis_local_index]
            return Real((Real(2.0) / length) * np.sum(weights * integrand))

        raise Unrecoverable('WireMesh3DFill: unsupported FillChoice')

    @staticmethod
    def _function_support(mesh: WireMesh3D, supports: FunctionSupports, support_index: Index) -> list[tuple[Index, int]]:
        s0, s1 = supports.pair(support_index)
        a0, a1 = mesh.subsegment_endpoints(s0)
        b0, b1 = mesh.subsegment_endpoints(s1)
        reltol = mesh.reltol

        if vector_equality(a0, b0, reltol):
            return [(s0, 0), (s1, 0)]
        if vector_equality(a0, b1, reltol):
            return [(s0, 0), (s1, 1)]
        if vector_equality(a1, b0, reltol):
            return [(s0, 1), (s1, 0)]
        if vector_equality(a1, b1, reltol):
            return [(s0, 1), (s1, 1)]

        raise Unrecoverable(
            'WireMesh3DFill: mapped function subsegments must share one vertex'
        )

    def make_filler(
        self,
        test_mesh: WireMesh3D,
        test_supports: FunctionSupports,
        test_function: WireScalarFunction,
        basis_mesh: WireMesh3D,
        basis_supports: FunctionSupports,
        basis_function: WireScalarFunction,
        fill_choice: FillChoice = FillChoice.MASS,
    ) -> Callable[[Index, Index], Real]:
        if not isinstance(fill_choice, FillChoice):
            raise Unrecoverable('WireMesh3DFill: fill_choice must be FillChoice')
        if test_function.axial_order != 1 or basis_function.axial_order != 1:
            raise NotYetImplemented(
                'WireMesh3DFill: only axial_order=1 is currently supported'
            )
        if test_function.azimuthal_order != 0 or basis_function.azimuthal_order != 0:
            raise NotYetImplemented(
                'WireMesh3DFill: only azimuthal_order=0 is currently supported'
            )
        reltol = Real(max(float(test_mesh.reltol), float(basis_mesh.reltol)))
        axial_rule = test_function.quadrature_rule.rule_2

        def filler(test_support_index: Index, basis_support_index: Index) -> Real:
            test_support = self._function_support(test_mesh, test_supports, test_support_index)
            basis_support = self._function_support(basis_mesh, basis_supports, basis_support_index)

            value = Real(0)
            for test_sub_idx, test_local_idx in test_support:
                test_start, test_end = test_mesh.subsegment_endpoints(test_sub_idx)
                length = Real(np.linalg.norm(test_end - test_start))

                for basis_sub_idx, basis_local_idx in basis_support:
                    basis_start, basis_end = basis_mesh.subsegment_endpoints(basis_sub_idx)
                    same_orientation = (
                        vector_equality(test_start, basis_start, reltol)
                        and vector_equality(test_end, basis_end, reltol)
                    )
                    reversed_orientation = (
                        vector_equality(test_start, basis_end, reltol)
                        and vector_equality(test_end, basis_start, reltol)
                    )
                    if not same_orientation and not reversed_orientation:
                        continue

                    basis_j = basis_local_idx if same_orientation else (1 - basis_local_idx)
                    value = Real(
                        value + self._local_entry(
                            fill_choice,
                            length,
                            test_local_idx,
                            basis_j,
                            axial_rule,
                        )
                    )

            return value

        return filler
