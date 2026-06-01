#!/usr/bin/env python3
# Copyright (c) 2023, Joseph M. Rutherford
from doodler.common import Index, Real

from .rules import Rule1D

import numpy as np

class ModepyRule1DSource:
    '''Constructs Rule1D quadratures from modepy.'''

    _gauss_label = 'gauss'
    _kronrod_label = 'kronrod'
    _clenshaw_curtis_label = 'clenshaw_curtis'

    def gauss_rule(self, size: Index) -> Rule1D:
        '''Create an N-node Gauss-Legendre rule on (-1,1).'''
        from modepy import LegendreGaussQuadrature

        if size < 1:
            raise ValueError('Gauss rule size must be >= 1')

        # modepy uses an order N that produces N+1 nodes.
        rule = LegendreGaussQuadrature(int(size) - 1, force_dim_axis=True)
        positions = np.array(rule.nodes[0], dtype=Real)
        weights = np.array(rule.weights, dtype=Real)
        return Rule1D(self._gauss_label, size, positions, weights)

    def kronrod_rule(self, size: Index) -> Rule1D:
        '''Create a (2*n+1)-node Gauss-Kronrod rule on (-1,1).'''
        from modepy.quadrature.kronrod import make_kronrod_quadrature

        if size < 3 or size % 2 == 0:
            raise ValueError('Kronrod rule size must be odd and >= 3')

        order = (int(size) - 1) // 2
        rule = make_kronrod_quadrature(order)
        positions = np.array(rule.nodes, dtype=Real)
        weights = np.array(rule.weights, dtype=Real)

        # Keep the ascending position convention used by cached parquet rules.
        sorted_indices = np.argsort(positions)
        positions = positions[sorted_indices]
        weights = weights[sorted_indices]
        return Rule1D(self._kronrod_label, size, positions, weights)

    def clenshaw_curtis_rule(self, size: Index) -> Rule1D:
        '''Create an N-node Clenshaw-Curtis rule on (-1,1).'''
        from modepy import ClenshawCurtisQuadrature

        if size < 2:
            raise ValueError('Clenshaw-Curtis rule size must be >= 2')

        # modepy uses an order N that produces N+1 nodes.
        rule = ClenshawCurtisQuadrature(int(size) - 1, force_dim_axis=True)
        positions = np.array(rule.nodes[0], dtype=Real)
        weights = np.array(rule.weights, dtype=Real)

        # Keep ascending position convention used by cached parquet rules.
        sorted_indices = np.argsort(positions)
        positions = positions[sorted_indices]
        weights = weights[sorted_indices]
        return Rule1D(self._clenshaw_curtis_label, size, positions, weights)
