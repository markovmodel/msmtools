
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