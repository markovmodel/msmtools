# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
