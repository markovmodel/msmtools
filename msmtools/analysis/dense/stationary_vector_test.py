
# This file is part of PyEMMA.
#
# Copyright 2015, Martin K. Scherer, Benjamin Trendelkamp-Schroer, Frank Noé,
# Fabian Paul, Guillermo Pérez-Hernández
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, division

import unittest
import numpy as np
from msmtools.util.numeric import assert_allclose

from .stationary_vector import stationary_distribution_from_eigenvector
from .stationary_vector import stationary_distribution_from_backward_iteration

from .birth_death_chain import BirthDeathChain


class TestStationaryVector(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim / 2 - 1] = 0.001
        q[self.dim / 2 + 1] = 0.001

        self.bdc = BirthDeathChain(q, p)

    def test_statdist_decomposition(self):
        P = self.bdc.transition_matrix()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_eigenvector(P)
        assert_allclose(mu, mun)

    def test_statdist_iteration(self):
        P = self.bdc.transition_matrix()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_backward_iteration(P)
        assert_allclose(mu, mun)

if __name__ == "__main__":
    unittest.main()