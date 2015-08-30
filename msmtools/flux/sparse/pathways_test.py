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
    

