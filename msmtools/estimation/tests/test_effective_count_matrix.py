import unittest

import numpy as np
from os.path import abspath, join
from os import pardir

from msmtools.estimation import count_matrix, effective_count_matrix

"""Unit tests for the transition_matrix module"""



class TestEffectiveCountMatrix(unittest.TestCase):
    def setUp(self):
        testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'
        self.dtraj_long = np.loadtxt(testpath + 'dtraj.dat', dtype=int)

    def tearDown(self):
        pass

    def test_singletraj(self):
        # lag 1
        C = count_matrix(self.dtraj_long, 1)
        Ceff = effective_count_matrix(self.dtraj_long, 1)
        assert np.array_equal(Ceff.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff.nonzero())
        assert np.all(Ceff.toarray() <= C.toarray())
        # lag 100
        C = count_matrix(self.dtraj_long, 100)
        Ceff = effective_count_matrix(self.dtraj_long, 100)
        assert np.array_equal(Ceff.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff.nonzero())
        assert np.all(Ceff.toarray() <= C.toarray())

    def test_multitraj(self):
        dtrajs = [[1,0,1,0,1,1,0,0,0,1], [2], [0,1,0,1]]
        # lag 1
        C = count_matrix(dtrajs, 1)
        Ceff = effective_count_matrix(dtrajs, 1)
        assert np.array_equal(Ceff.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff.nonzero())
        assert np.all(Ceff.toarray() <= C.toarray())
        # lag 2
        C = count_matrix(dtrajs, 2)
        Ceff = effective_count_matrix(dtrajs, 2)
        assert np.array_equal(Ceff.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff.nonzero())
        assert np.all(Ceff.toarray() <= C.toarray())

if __name__ == "__main__":
    unittest.main()