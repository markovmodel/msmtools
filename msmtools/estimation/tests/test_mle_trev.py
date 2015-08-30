
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

from __future__ import absolute_import
import unittest
import numpy as np
from msmtools.util.numeric import assert_allclose
import scipy
import scipy.sparse
import warnings
import msmtools.util.exceptions

from os.path import abspath, join
from os import pardir

from msmtools.estimation.sparse.mle_trev import mle_trev as impl_sparse
from msmtools.estimation.dense.transition_matrix import estimate_transition_matrix_reversible as impl_dense
from msmtools.estimation import tmatrix as apicall

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class Test_mle_trev(unittest.TestCase):
    def test_mle_trev(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')

        T_impl_algo_sparse_type_sparse = impl_sparse(scipy.sparse.csr_matrix(C)).toarray()
        T_impl_algo_dense_type_dense = impl_dense(C)

        T_api_algo_dense_type_dense = apicall(C, reversible=True, method='dense')
        T_api_algo_sparse_type_dense = apicall(C, reversible=True, method='sparse')
        T_api_algo_dense_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='dense').toarray()
        T_api_algo_sparse_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='sparse').toarray()
        T_api_algo_auto_type_dense = apicall(C, reversible=True, method='auto')
        T_api_algo_auto_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, method='auto').toarray()

        assert_allclose(T_impl_algo_sparse_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_dense_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_sparse_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_dense_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_sparse_type_sparse, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_auto_type_dense, T_impl_algo_dense_type_dense)
        assert_allclose(T_api_algo_auto_type_sparse, T_impl_algo_dense_type_dense)

    def test_warnings(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            impl_sparse(scipy.sparse.csr_matrix(C), maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, msmtools.util.exceptions.NotConvergedWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            impl_dense(C, maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, msmtools.util.exceptions.NotConvergedWarning)

    def test_noninteger_counts_sparse(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_sparse_reference = impl_sparse(scipy.sparse.csr_matrix(C)).toarray()
        T_sparse_scaled_1 = impl_sparse(scipy.sparse.csr_matrix(C*10.0)).toarray()
        T_sparse_scaled_2 = impl_sparse(scipy.sparse.csr_matrix(C*0.1)).toarray()
        assert_allclose(T_sparse_reference, T_sparse_scaled_1)
        assert_allclose(T_sparse_reference, T_sparse_scaled_2)

    def test_noninteger_counts_dense(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        T_dense_reference = impl_dense(C)
        T_dense_scaled_1 = impl_dense(C*10.0)
        T_dense_scaled_2 = impl_dense(C*0.1)
        assert_allclose(T_dense_reference, T_dense_scaled_1)
        assert_allclose(T_dense_reference, T_dense_scaled_2)


if __name__ == '__main__':
    unittest.main()