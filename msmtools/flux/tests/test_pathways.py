
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
#
# MSMTools is free software: you can redistribute it and/or modify
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

"""Unit test for the reaction pathway decomposition

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import

import warnings
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from msmtools.util.numeric import assert_allclose
from msmtools.flux import pathways
from six.moves import range


class TestPathways(unittest.TestCase):

    def setUp(self):
        """Small flux-network"""
        F = np.zeros((8, 8))
        F[0, 2] = 10.0
        F[2, 6] = 10.0
        F[1, 3] = 100.0
        F[3, 4] = 30.0
        F[3, 5] = 70.0
        F[4, 6] = 5.0
        F[4, 7] = 25.0
        F[5, 6] = 30.0
        F[5, 7] = 40.0
        """Reactant and product states"""
        A = [0, 1]
        B = [6, 7]

        self.F = F
        self.F_sparse = csr_matrix(F)
        self.A = A
        self.B = B
        self.paths = []
        self.capacities = []
        p1 = np.array([1, 3, 5, 7])
        c1 = 40.0
        self.paths.append(p1)
        self.capacities.append(c1)
        p2 = np.array([1, 3, 5, 6])
        c2 = 30.0
        self.paths.append(p2)
        self.capacities.append(c2)
        p3 = np.array([1, 3, 4, 7])
        c3 = 25.0
        self.paths.append(p3)
        self.capacities.append(c3)
        p4 = np.array([0, 2, 6])
        c4 = 10.0
        self.paths.append(p4)
        self.capacities.append(c4)
        p5 = np.array([1, 3, 4, 6])
        c5 = 5.0
        self.paths.append(p5)
        self.capacities.append(c5)

    def test_pathways_dense(self):
        paths, capacities = pathways(self.F, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_dense_incomplete(self):
        paths, capacities = pathways(self.F, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])
            assert issubclass(w[-1].category, RuntimeWarning)

    def test_pathways_sparse(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_sparse_incomplete(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])
            assert issubclass(w[-1].category, RuntimeWarning)

if __name__ == "__main__":
    unittest.main()