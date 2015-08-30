
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
Created on 06.12.2013

@author: jan-hendrikprinz

This module provides the unittest for the pcca module

'''
from __future__ import absolute_import
from __future__ import division
import unittest
import numpy as np

from msmtools.util.numeric import assert_allclose
from .pcca import pcca


class TestPCCA(unittest.TestCase):
    def test_pcca_no_transition_matrix(self):
        P = np.array([[1.0, 1.0],
                      [0.1, 0.9]])

        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_no_detailed_balance(self):
        P = np.array([[0.8, 0.1, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.6, 0.3, 0.1]])
        with self.assertRaises(ValueError):
            pcca(P, 2)

    def test_pcca_1(self):
        P = np.array([[1, 0],
                      [0, 1]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_2(self):
        P = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.999, 0.001],
                      [0.0, 0.001, 0.999]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=3
        chi = pcca(P, 3)
        sol = np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_4(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_5(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[1., 0.],
                        [1., 0.],
                        [0.5, 0.5],
                        [0., 1.],
                        [0., 1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0.5, 0.5, 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        assert_allclose(chi, sol)

    def test_pcca_large(self):
        import os

        P = np.loadtxt(os.path.split(__file__)[0] + '/../tests/P_rev_251x251.dat')
        # n=2
        chi = pcca(P, 2)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=3
        chi = pcca(P, 3)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))
        # n=4
        chi = pcca(P, 4)
        assert (np.alltrue(chi >= 0))
        assert (np.alltrue(chi <= 1))

    def test_pcca_coarsegrain(self):
        # fine-grained transition matrix
        P = np.array([[0.9,  0.1,  0.0,  0.0,  0.0],
                      [0.1,  0.89, 0.01, 0.0,  0.0],
                      [0.0,  0.1,  0.8,  0.1,  0.0],
                      [0.0,  0.0,  0.01, 0.79, 0.2],
                      [0.0,  0.0,  0.0,  0.2,  0.8]])
        from msmtools.analysis import stationary_distribution
        pi = stationary_distribution(P)
        Pi = np.diag(pi)
        m = 3
        # Susanna+Marcus' expression ------------
        M = pcca(P, m)
        pi_c = np.dot(M.T, pi)
        Pi_c_inv = np.diag(1.0/pi_c)
        # restriction and interpolation operators
        R = M.T
        I = np.dot(np.dot(Pi, M), Pi_c_inv)
        # result
        ms1 = np.linalg.inv(np.dot(R,I)).T
        ms2 = np.dot(np.dot(I.T, P), R.T)
        Pc_ref = np.dot(ms1,ms2)
        # ---------------------------------------

        from .pcca import coarsegrain
        Pc = coarsegrain(P, 3)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(Pc - Pc_ref)) < 1e-10
        # test mass conservation
        assert np.allclose(Pc.sum(axis=1), np.ones(m))

        from .pcca import PCCA
        p = PCCA(P, m)
        # test against Marcus+Susanna's expression
        assert np.max(np.abs(p.coarse_grained_transition_matrix - Pc_ref)) < 1e-10
        # test against the present coarse-grained stationary dist
        assert np.max(np.abs(p.coarse_grained_stationary_probability - pi_c)) < 1e-10
        # test mass conservation
        assert np.allclose(p.coarse_grained_transition_matrix.sum(axis=1), np.ones(m))

if __name__ == "__main__":
    unittest.main()