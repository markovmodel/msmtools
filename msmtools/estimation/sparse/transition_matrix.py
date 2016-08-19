
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

"""This module implements the transition matrix functionality"""
from __future__ import absolute_import
from __future__ import division

import numpy
import scipy.sparse


def transition_matrix_non_reversible(C):
    """implementation of transition_matrix"""
    if not scipy.sparse.issparse(C):
        C = scipy.sparse.csr_matrix(C)
    rowsum = C.tocsr().sum(axis=1)
    # catch div by zero
    if(numpy.min(rowsum) == 0.0):
        raise ValueError("matrix C contains rows with sum zero.")
    rowsum = numpy.array(1. / rowsum).flatten()
    norm = scipy.sparse.diags(rowsum, 0)
    return norm * C

def correct_transition_matrix(T, reversible=None):
    r"""Normalize transition matrix

    Fixes a the row normalization of a transition matrix.
    To be used with the reversible estimators to fix an almost coverged
    transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray
        matrix to correct
    reversible : boolean
        for future use

    Returns
    -------
    (M, M) ndarray
        corrected transition matrix
    """
    row_sums = T.sum(axis=1).A1
    max_sum = numpy.max(row_sums)
    if max_sum == 0.0:
         max_sum = 1.0
    return (T + scipy.sparse.diags(-row_sums+max_sum, 0)) / max_sum
