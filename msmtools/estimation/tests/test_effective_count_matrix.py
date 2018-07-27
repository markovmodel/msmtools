
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

import unittest

import numpy as np
from os.path import abspath, join
from os import pardir

from msmtools.estimation import count_matrix, effective_count_matrix

"""Unit tests for the transition_matrix module"""

have_multiprocess_lib = True
try:
    import multiprocess
    del multiprocess
except ImportError:
    have_multiprocess_lib = False


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
        dtrajs = [[1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [2], [0, 1, 0, 1]]
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

    @unittest.skipIf(not have_multiprocess_lib, 'multiprocess lib missing')
    def test_multitraj_njobs(self):
        dtrajs = [[1, 0, 1, 0, 1, 1, 0, 0, 0, 1], [2], [0, 1, 0, 1]]
        # lag 1
        C = count_matrix(dtrajs, 1)
        Ceff = effective_count_matrix(dtrajs, 1, n_jobs=1)
        assert np.array_equal(Ceff.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff.nonzero())
        assert np.all(Ceff.toarray() <= C.toarray())

        Ceff2 = effective_count_matrix(dtrajs, 1, n_jobs=2)
        assert np.array_equal(Ceff2.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff2.nonzero())
        assert np.all(Ceff2.toarray() <= C.toarray())

        # lag 2
        C = count_matrix(dtrajs, 2)
        Ceff2 = effective_count_matrix(dtrajs, 2)
        assert np.array_equal(Ceff2.shape, C.shape)
        assert np.array_equal(C.nonzero(), Ceff2.nonzero())
        assert np.all(Ceff2.toarray() <= C.toarray())


class TestEffectiveCountMatrix_old_impl(unittest.TestCase):

    def test_compare_with_old_impl(self):
        # generated with v1.1@ from
        # pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10_n6good
        Ceff_ref = np.array([[2.21353316e+04,   2.13659736e+03,   4.63558176e+02,
                              1.56043628e+02,   3.88680098e+01,   1.14317676e+01],
                             [1.84456322e+03,   3.74107190e+02,   1.79811199e+02,
                              9.29024530e+01,   5.59412620e+01,   2.59727288e+01],
                             [3.45678646e+02,   1.42148228e+02,   8.19775293e+01,
                              7.75353971e+01,   5.73438875e+01,   8.19775293e+01],
                             [9.08206988e+01,   6.53466003e+01,   7.82682445e+01,
                              7.71606750e+01,   8.38060919e+01,   2.84276171e+02],
                             [3.56219388e+01,   3.43186971e+01,   7.64568442e+01,
                              1.13816439e+02,   2.51960055e+02,   1.33451946e+03],
                             [1.57044024e+01,   3.26168358e+01,   1.12346879e+02,
                              4.34287128e+02,   1.88573632e+03,   2.35837843e+04]])
        import pkg_resources
        f = pkg_resources.resource_filename('msmtools.estimation', 'tests/testfiles/dwell.npz')
        ref_dtraj = np.load(f)['dtraj_T100K_dt10_n6good'].astype('int32')
        Ceff = effective_count_matrix(ref_dtraj, lag=10, average='row', mact=1.0).toarray()

        np.testing.assert_allclose(Ceff, Ceff_ref, atol=1e-15, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
