'''
Created on 28.10.2013

@author: marscher
'''
from __future__ import absolute_import
import numpy as np
from numpy.testing import assert_allclose as assert_allclose_np

__all__ = ['allclose_sparse',
           'assert_allclose',
           ]


def assert_allclose(actual, desired, rtol=1.e-5, atol=1.e-8,
                    err_msg='', verbose=True):
    r"""wrapper for numpy.testing.allclose with default tolerances of
    numpy.allclose. Needed since testing method has different values."""
    return assert_allclose_np(actual, desired, rtol=rtol, atol=atol,
                              err_msg=err_msg, verbose=True)


def allclose_sparse(A, B, rtol=1e-5, atol=1e-8):
    """
    Compares two sparse matrices in the same matter like numpy.allclose()
    Parameters
    ----------
    A : scipy.sparse matrix
        first matrix to compare
    B : scipy.sparse matrix
        second matrix to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance

    Returns
    -------
    True, if given matrices are equal in bounds of rtol and atol
    False, otherwise

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.
    """
    A = A.tocsr()
    B = B.tocsr()

    """Shape"""
    same_shape = (A.shape == B.shape)

    """Data"""
    if same_shape:
        diff = (A - B).data
        same_data = np.allclose(diff, 0.0, rtol=rtol, atol=atol)

        return same_data
    else:
        return False
