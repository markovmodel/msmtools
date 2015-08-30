r"""Test package for the decomposition module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import, division
import unittest

import numpy as np
from msmtools.util.numeric import assert_allclose

from scipy.linalg import eig, eigvals

from .birth_death_chain import BirthDeathChain

from .stationary_vector import stationary_distribution_from_eigenvector
from .stationary_vector import stationary_distribution_from_backward_iteration


class TestStationaryVector(unittest.TestCase):
    def setUp(self):
        self.dim = 100
        self.k = 10
        self.ncv = 40

        """Set up meta-stable birth-death chain"""
        p = np.zeros(self.dim)
        p[0:-1] = 0.5

        q = np.zeros(self.dim)
        q[1:] = 0.5

        p[self.dim / 2 - 1] = 0.001
        q[self.dim / 2 + 1] = 0.001

        self.bdc = BirthDeathChain(q, p)

    def test_statdist_decomposition(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_eigenvector(P, ncv=self.ncv)
        assert_allclose(mu, mun)

    def test_statdist_iteration(self):
        P = self.bdc.transition_matrix_sparse()
        mu = self.bdc.stationary_distribution()
        mun = stationary_distribution_from_backward_iteration(P)
        assert_allclose(mu, mun)

if __name__ == "__main__":
    unittest.main()
