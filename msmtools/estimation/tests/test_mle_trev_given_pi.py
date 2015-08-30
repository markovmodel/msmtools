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

from msmtools.estimation.dense.mle_trev_given_pi import mle_trev_given_pi as impl_dense
from msmtools.estimation.sparse.mle_trev_given_pi import mle_trev_given_pi as impl_sparse
from msmtools.estimation.dense.transition_matrix import transition_matrix_reversible_fixpi as impl_dense_Frank
from msmtools.estimation import tmatrix as apicall
from msmtools.analysis import statdist, is_transition_matrix

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class Test_mle_trev_given_pi(unittest.TestCase):
    def setUp(self):
        pass

    def test_mle_trev_given_pi(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        pi = np.loadtxt(testpath + 'pi.dat')

        T_impl_algo_dense_type_dense = impl_dense(C, pi)
        T_impl_algo_sparse_type_sparse = impl_sparse(scipy.sparse.csr_matrix(C), pi).toarray()
        T_Frank = impl_dense_Frank(C, pi)
        T_api_algo_dense_type_dense = apicall(C, reversible=True, mu=pi, method='dense')
        T_api_algo_sparse_type_dense = apicall(C, reversible=True, mu=pi, method='sparse')
        T_api_algo_dense_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='dense').toarray()
        T_api_algo_sparse_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='sparse').toarray()
        T_api_algo_auto_type_dense = apicall(C, reversible=True, mu=pi, method='auto')
        T_api_algo_auto_type_sparse = apicall(scipy.sparse.csr_matrix(C), reversible=True, mu=pi, method='auto').toarray()

        assert_allclose(T_impl_algo_dense_type_dense, T_Frank)
        assert_allclose(T_impl_algo_sparse_type_sparse, T_Frank)
        assert_allclose(T_api_algo_dense_type_dense, T_Frank)
        assert_allclose(T_api_algo_sparse_type_dense, T_Frank)
        assert_allclose(T_api_algo_dense_type_sparse, T_Frank)
        assert_allclose(T_api_algo_sparse_type_sparse, T_Frank)
        assert_allclose(T_api_algo_auto_type_dense, T_Frank)
        assert_allclose(T_api_algo_auto_type_sparse, T_Frank)

        assert is_transition_matrix(T_Frank)
        assert is_transition_matrix(T_impl_algo_dense_type_dense)
        assert is_transition_matrix(T_impl_algo_sparse_type_sparse)
        assert is_transition_matrix(T_api_algo_dense_type_dense)
        assert is_transition_matrix(T_api_algo_sparse_type_dense)
        assert is_transition_matrix(T_api_algo_dense_type_sparse)
        assert is_transition_matrix(T_api_algo_sparse_type_sparse)
        assert is_transition_matrix(T_api_algo_auto_type_dense)
        assert is_transition_matrix(T_api_algo_auto_type_sparse)

        assert_allclose(statdist(T_Frank), pi)
        assert_allclose(statdist(T_impl_algo_dense_type_dense), pi)
        assert_allclose(statdist(T_impl_algo_sparse_type_sparse), pi)
        assert_allclose(statdist(T_api_algo_dense_type_dense), pi)
        assert_allclose(statdist(T_api_algo_sparse_type_dense), pi)
        assert_allclose(statdist(T_api_algo_dense_type_sparse), pi)
        assert_allclose(statdist(T_api_algo_sparse_type_sparse), pi)
        assert_allclose(statdist(T_api_algo_auto_type_dense), pi)
        assert_allclose(statdist(T_api_algo_auto_type_sparse), pi)

    def test_warnings(self):
        C = np.loadtxt(testpath + 'C_1_lag.dat')
        pi = np.loadtxt(testpath + 'pi.dat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            impl_sparse(scipy.sparse.csr_matrix(C), pi, maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, msmtools.util.exceptions.NotConvergedWarning)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            impl_dense(C, pi, maxiter=1)
            assert len(w) == 1
            assert issubclass(w[-1].category, msmtools.util.exceptions.NotConvergedWarning)


if __name__ == '__main__':
    unittest.main()
