
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

r"""Unit tests for the connectivity API functions

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import

import unittest

import numpy as np
from msmtools.util.numeric import assert_allclose
import scipy.sparse

from msmtools.estimation import connected_sets, largest_connected_set, largest_connected_submatrix, is_connected
from six.moves import range

################################################################################
# Dense
################################################################################


class TestConnectedSetsDense(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.toarray()

        self.cc_directed = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        self.cc_undirected = [np.array([0, 1, 2, 3, 4, 5])]

    def tearDown(self):
        pass

    def test_connected_sets(self):
        """Directed"""
        cc = connected_sets(self.C)
        for i in range(len(cc)):
            self.assertTrue(np.all(self.cc_directed[i] == np.sort(cc[i])))

        """Undirected"""
        cc = connected_sets(self.C, directed=False)
        for i in range(len(cc)):
            self.assertTrue(np.all(self.cc_undirected[i] == np.sort(cc[i])))


class TestLargestConnectedSetSparse(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.toarray()

        self.cc_directed = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        self.cc_undirected = [np.array([0, 1, 2, 3, 4, 5])]

        self.lcc_directed = self.cc_directed[0]
        self.lcc_undirected = self.cc_undirected[0]

    def tearDown(self):
        pass

    def test_largest_connected_set(self):
        """Directed"""
        lcc = largest_connected_set(self.C)
        self.assertTrue(np.all(self.lcc_directed == np.sort(lcc)))

        """Undirected"""
        lcc = largest_connected_set(self.C, directed=False)
        self.assertTrue(np.all(self.lcc_undirected == np.sort(lcc)))


class TestConnectedCountMatrixDense(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.toarray()

        self.C_cc_directed = C1
        self.C_cc_undirected = self.C

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        """Directed"""
        C_cc = largest_connected_submatrix(self.C)
        assert_allclose(C_cc, self.C_cc_directed)

        """Undirected"""
        C_cc = largest_connected_submatrix(self.C, directed=False)
        assert_allclose(C_cc, self.C_cc_undirected)


class TestIsConnectedDense(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.tocoo()

        self.C_connected = C1
        self.C_not_connected = self.C.toarray()

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        """Directed"""
        is_c = is_connected(self.C_not_connected)
        self.assertFalse(is_c)
        is_c = is_connected(self.C_connected)
        self.assertTrue(is_c)

        """Undirected"""
        is_c = is_connected(self.C_not_connected, directed=False)
        self.assertTrue(is_c)


################################################################################
# Sparse
################################################################################

class TestConnectedSetsSparse(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.tocoo()

        self.cc_directed = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        self.cc_undirected = [np.array([0, 1, 2, 3, 4, 5])]

    def tearDown(self):
        pass

    def test_connected_sets(self):
        """Directed"""
        cc = connected_sets(self.C)
        for i in range(len(cc)):
            self.assertTrue(np.all(self.cc_directed[i] == np.sort(cc[i])))

        """Undirected"""
        cc = connected_sets(self.C, directed=False)
        for i in range(len(cc)):
            self.assertTrue(np.all(self.cc_undirected[i] == np.sort(cc[i])))


class TestLargestConnectedSetSparse(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.tocoo()

        self.cc_directed = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]
        self.cc_undirected = [np.array([0, 1, 2, 3, 4, 5])]

        self.lcc_directed = self.cc_directed[0]
        self.lcc_undirected = self.cc_undirected[0]

    def tearDown(self):
        pass

    def test_largest_connected_set(self):
        """Directed"""
        lcc = largest_connected_set(self.C)
        self.assertTrue(np.all(self.lcc_directed == np.sort(lcc)))

        """Undirected"""
        lcc = largest_connected_set(self.C, directed=False)
        self.assertTrue(np.all(self.lcc_undirected == np.sort(lcc)))


class TestConnectedCountMatrixSparse(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.tocoo()

        self.C_cc_directed = C1
        self.C_cc_undirected = self.C.toarray()

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        """Directed"""
        C_cc = largest_connected_submatrix(self.C)
        assert_allclose(C_cc.toarray(), self.C_cc_directed)

        """Directed with user specified lcc"""
        C_cc = largest_connected_submatrix(self.C, lcc=np.array([0, 1]))
        assert_allclose(C_cc.toarray(), self.C_cc_directed[0:2, 0:2])

        """Undirected"""
        C_cc = largest_connected_submatrix(self.C, directed=False)
        assert_allclose(C_cc.toarray(), self.C_cc_undirected)

        """Undirected with user specified lcc"""
        C_cc = largest_connected_submatrix(self.C, lcc=np.array([0, 1]), directed=False)
        assert_allclose(C_cc.toarray(), self.C_cc_undirected[0:2, 0:2])


class TestIsConnectedSparse(unittest.TestCase):
    def setUp(self):
        C1 = np.array([[1, 4, 3], [3, 2, 4], [4, 5, 1]])
        C2 = np.array([[0, 1], [1, 0]])
        C3 = np.array([[7]])

        self.C = scipy.sparse.block_diag((C1, C2, C3))

        self.C = self.C.tolil()
        """Forward transition block 1 -> block 2"""
        self.C[2, 3] = 1
        """Forward transition block 2 -> block 3"""
        self.C[4, 5] = 1
        self.C = self.C.tocoo()

        self.C_connected = scipy.sparse.csr_matrix(C1)
        self.C_not_connected = self.C

    def tearDown(self):
        pass

    def test_connected_count_matrix(self):
        """Directed"""
        is_c = is_connected(self.C_not_connected)
        self.assertFalse(is_c)
        is_c = is_connected(self.C_connected)
        self.assertTrue(is_c)

        """Undirected"""
        is_c = is_connected(self.C_not_connected, directed=False)
        self.assertTrue(is_c)


if __name__ == "__main__":
    unittest.main()