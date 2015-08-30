
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

r"""This module provides functions for the computation of forward and
backward comittors using dense linear algebra.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import, division
from six.moves import range

import numpy as np
from scipy.linalg import solve

from .stationary_vector import stationary_distribution


def forward_committor(T, A, B):
    r"""Forward committor between given sets.

    The forward committor u(x) between sets A and B is the probability
    for the chain starting in x to reach B before reaching A.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B

    Returns
    -------
    u : (M, ) ndarray
        Vector of forward committor probabilities

    Notes
    -----
    The forward committor is a solution to the following
    boundary-value problem

    .. math::

        \sum_j L_{ij} u_{j}=0    for i in X\(A u B) (I)
                      u_{i}=0    for i \in A        (II)
                      u_{i}=1    for i \in B        (III)

    with generator matrix L=(P-I).

    """
    X = set(range(T.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    L = T - np.eye(T.shape[0])  # Generator matrix

    """Assemble left hand-side W for linear system"""
    """Equation (I)"""
    W = 1.0 * L
    """Equation (II)"""
    W[list(A), :] = 0.0
    W[list(A), list(A)] = 1.0
    """Equation (III)"""
    W[list(B), :] = 0.0
    W[list(B), list(B)] = 1.0

    """Assemble right hand side r for linear system"""
    """Equation (I+II)"""
    r = np.zeros(T.shape[0])
    """Equation (III)"""
    r[list(B)] = 1.0

    u = solve(W, r)
    return u


def backward_committor(T, A, B, mu=None):
    r"""Backward committor between given sets.

    The backward committor u(x) between sets A and B is the
    probability for the chain starting in x to have come from A last
    rather than from B.

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M, ) ndarray (optional)
        Stationary vector

    Returns
    -------
    u : (M, ) ndarray
        Vector of forward committor probabilities

    Notes
    -----
    The forward committor is a solution to the following
    boundary-value problem

    .. math::

        \sum_j K_{ij} \pi_{j} u_{j}=0    for i in X\(A u B) (I)
                                  u_{i}=1    for i \in A        (II)
                                  u_{i}=0    for i \in B        (III)

    with adjoint of the generator matrix K=(D_pi(P-I))'.

    """
    X = set(range(T.shape[0]))
    A = set(A)
    B = set(B)
    AB = A.intersection(B)
    notAB = X.difference(A).difference(B)
    if len(AB) > 0:
        raise ValueError("Sets A and B have to be disjoint")
    if mu is None:
        mu = stationary_distribution(T)
    K = np.transpose(mu[:, np.newaxis] * (T - np.eye(T.shape[0])))

    """Assemble left-hand side W for linear system"""
    """Equation (I)"""
    W = 1.0 * K
    """Equation (II)"""
    W[list(A), :] = 0.0
    W[list(A), list(A)] = 1.0
    """Equation (III)"""
    W[list(B), :] = 0.0
    W[list(B), list(B)] = 1.0

    """Assemble right-hand side r for linear system"""
    """Equation (I)+(III)"""
    r = np.zeros(T.shape[0])
    """Equation (II)"""
    r[list(A)] = 1.0

    u = solve(W, r)

    return u