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

r"""Unit test for the pathways-module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import
from __future__ import division

import unittest
import numpy as np
from scipy.sparse import csr_matrix

from msmtools.flux.sparse.pathways import pathways
from msmtools.flux.sparse.tpt import flux_matrix
from msmtools.analysis import committor, statdist
from msmtools.util.numeric import assert_allclose

class TestPathways(unittest.TestCase):

    def setUp(self):
        P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
                      [0.1,  0.75, 0.05, 0.05, 0.05],
                      [0.05,  0.1,  0.8,  0.0,  0.05],
                      [0.0,  0.2, 0.0,  0.8,  0.0],
                      [0.0,  0.02, 0.02, 0.0,  0.96]])
        P = csr_matrix(P)
        A = [0]
        B = [4]
        mu = statdist(P)
        qminus = committor(P, A, B, forward=False, mu=mu)
        qplus = committor(P, A, B, forward=True, mu=mu)    
        self.A = A
        self.B = B
        self.F = flux_matrix(P, mu, qminus, qplus, netflux=True)

        self.paths = [np.array([0, 1, 4]), np.array([0, 2, 4]), np.array([0, 1, 2, 4])]
        self.capacities = [0.0072033898305084252, 0.0030871670702178975, 0.00051452784503631509]                
    def test_pathways(self):
        paths, capacities = pathways(self.F, self.A, self.B)
        assert_allclose(capacities, self.capacities)
        N = len(paths)
        for i in range(N):
            self.assertTrue(np.all(paths[i] == self.paths[i]))
            

if __name__=="__main__":
    unittest.main()
    

