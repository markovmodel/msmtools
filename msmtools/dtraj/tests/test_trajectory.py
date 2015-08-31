
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

r"""This module contains unit tests for the trajectory module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import
import os
import unittest

import numpy as np

from os.path import abspath, join
from os import pardir

from msmtools.dtraj import read_discrete_trajectory, write_discrete_trajectory, \
    load_discrete_trajectory, save_discrete_trajectory

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestReadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj_np = np.loadtxt(self.filename, dtype=int)
        dtraj = read_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_np == dtraj))


class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.dat'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


class TestLoadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.npy'

    def tearDown(self):
        pass

    def test_load_discrete_trajectory(self):
        dtraj_n = np.load(self.filename)
        dtraj = load_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_n == dtraj))


class TestSaveDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.npy'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.load(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


if __name__ == "__main__":
    unittest.main()