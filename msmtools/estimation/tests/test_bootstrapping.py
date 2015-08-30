
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

'''
Created on Jul 25, 2014

@author: noe
'''
from __future__ import absolute_import
from __future__ import division
import unittest
import numpy as np
import msmtools.estimation as msmest
from six.moves import range


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def validate_counts(self, ntraj, length, n, tau):
        dtrajs = []
        for i in range(ntraj):
            dtrajs.append(np.random.random_integers(0, n-1, size=length))
        for i in range(10):
            C = msmest.bootstrap_counts(dtrajs, tau).toarray()
            assert(np.shape(C) == (n, n))
            assert(np.sum(C) == (ntraj*length) / tau)

    def test_bootstrap_counts(self):
        self.validate_counts(1, 10000, 10, 10)
        self.validate_counts(1, 10000, 100, 1000)
        self.validate_counts(10, 100, 2, 10)
        self.validate_counts(10, 1000, 100, 100)
        self.validate_counts(1000, 10, 1000, 1)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()