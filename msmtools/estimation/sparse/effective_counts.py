# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
r"""This module implements effective transition counts

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""
from __future__ import absolute_import, division

from six.moves import range
import numpy as np
import scipy.sparse

from msmtools.util.statistics import statistical_inefficiency
from msmtools.estimation.sparse.count_matrix import count_matrix_mult
from msmtools.dtraj.discrete_trajectory import number_of_states

__author__ = 'noe'


def _split_sequences_singletraj(dtraj, nstates, lag):
    """ splits the discrete trajectory into conditional sequences by starting state

    Parameters
    ----------
    dtraj : int-iterable
        discrete trajectory
    nstates : int
        total number of discrete states
    lag : int
        lag time

    """
    sall = []
    for i in range(nstates):
        sall.append([])
    for t in range(len(dtraj)-lag):
        sall[dtraj[t]].append(dtraj[t+lag])
    res_states = []
    res_seqs = []
    for i in range(nstates):
        if len(sall[i]) > 0:
            res_states.append(i)
            res_seqs.append(np.array(sall[i]))
    return res_states, res_seqs

def _split_sequences_multitraj(dtrajs, lag):
    """ splits the discrete trajectories into conditional sequences by starting state

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    nstates : int
        total number of discrete states
    lag : int
        lag time

    """
    n = number_of_states(dtrajs)
    res = []
    for i in range(n):
        res.append([])
    for dtraj in dtrajs:
        states, seqs = _split_sequences_singletraj(dtraj, n, lag)
        for i in range(len(states)):
            res[states[i]].append(seqs[i])
    return res

def _indicator_multitraj(ss, i, j):
    """ Returns conditional sequence for transition i -> j given all conditional sequences """
    iseqs = ss[i]
    res = []
    for iseq in iseqs:
        x = np.zeros(len(iseq))
        I = np.where(iseq == j)
        x[I] = 1.0
        res.append(x)
    return res

def _transition_indexes(dtrajs, lag):
    """ for each state, returns a list of target states to which a transition is observed at lag tau """
    C = count_matrix_mult(dtrajs, lag, sliding=True, sparse=True)
    res = []
    for i in range(C.shape[0]):
        I,J = C[i].nonzero()
        res.append(J)
    return res

def statistical_inefficiencies(dtrajs, lag, C=None, truncate_acf=True, mact=2.0):
    """ Computes statistical inefficiencies of sliding-window transition counts at given lag

    Consider a discrete trajectory :math`{ x_t }` with :math:`x_t \in {1, ..., n}`. For each starting state :math:`i`,
    we collect the target sequence

    .. mathh:
        Y^(i) = {x_{t+\tau} | x_{t}=i}

    which contains the time-ordered target states at times :math:`t+\tau` whenever we started in state :math:`i`
    at time :math:`t`. Then we define the indicator sequence:

    .. math:
        a^{(i,j)}_t (\tau) = 1(Y^(i)_t = j)

    The statistical inefficiency for transition counts :math:`c_{ij}(tau)` is computed as the statistical inefficiency
    of the sequence :math:`a^{(i,j)}_t (\tau)`.

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    lag : int
        lag time
    C : scipy sparse matrix (n, n) or None
        sliding window count matrix, if already available
    truncate_acf : bool, optional, default=True
        When the normalized autocorrelation function passes through 0, it is truncated in order to avoid integrating
        random noise

    Returns
    -------
    I : scipy sparse matrix (n, n)
        Statistical inefficiency matrix with a sparsity pattern identical to the sliding-window count matrix at the
        same lag time. Will contain a statistical inefficiency :math:`I_{ij} \in (0,1]` whenever there is a count
        :math:`c_{ij} > 0`. When there is no transition count (:math:`c_{ij} = 0`), the statistical inefficiency is 0.

    See also
    --------
    msmtools.util.statistics.statistical_inefficiency
        used to compute the statistical inefficiency for conditional trajectories

    """
    # count matrix
    if C is None:
        C = count_matrix_mult(dtrajs, lag, sliding=True, sparse=True)
    # split sequences
    splitseq = _split_sequences_multitraj(dtrajs, lag)
    # compute inefficiencies
    res = C.copy()  # copy count matrix and use its sparsity structure
    I,J = C.nonzero()
    for k in range(len(I)):
        i = I[k]
        j = J[k]
        X = _indicator_multitraj(splitseq, i, j)
        res[i, j] = statistical_inefficiency(X, truncate_acf=truncate_acf, mact=mact)

    return res

def effective_count_matrix(dtrajs, lag, average='row', truncate_acf=True, mact=1.0):
    """ Computes the statistically effective transition count matrix

    Given a list of discrete trajectories, compute the effective number of statistically uncorrelated transition
    counts at the given lag time. First computes the full sliding-window counts :math:`c_{ij}(tau)`. Then uses
    :func:`statistical_inefficiencies` to compute statistical inefficiencies :math:`I_{ij}(tau)`. The number of
    effective counts in a row is then computed as

    .. math:
        c_i^{\mathrm{eff}}(tau) = \sum_j I_{ij}(tau) c_{ij}(tau)

    and the effective transition counts are obtained by scaling the rows accordingly:

    .. math:
        c_{ij}^{\mathrm{eff}}(tau) = \frac{c_i^{\mathrm{eff}}(tau)}{c_i(tau)} c_{ij}(tau)

    This procedure is not yet published, but a manuscript is in preparation [1]_.

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    lag : int
        lag time
    average : str, default='row'
        Use either of 'row', 'all', 'none', with the following consequences:
        'none': the statistical inefficiency is applied separately to each
            transition count (not recommended)
        'row': the statistical inefficiency is averaged (weighted) by row
            (recommended).
        'all': the statistical inefficiency is averaged (weighted) over all
            transition counts (not recommended).
    truncate_acf : bool, optional, default=True
        Mode of estimating the autocorrelation time of transition counts.
        True: When the normalized autocorrelation function passes through 0,
        it is truncated in order to avoid integrating random noise. This
        tends to lead to a slight underestimate of the autocorrelation time
    mact : float, default=2.0
        multiplier for the autocorrelation time. We tend to underestimate the
        autocorrelation time (and thus overestimate effective counts)
        because the autocorrelation function is truncated when it passes
        through 0 in order to avoid numerical instabilities.
        This is a purely heuristic factor trying to compensate this effect.
        This parameter might be removed in the future when a more robust
        estimation method of the autocorrelation time is used.

    See also
    --------
    statistical_inefficiencies
        is used for computing the statistical inefficiences of sliding window
        transition counts

    References
    ----------
    .. [1] Noe, F. and H. Wu: in preparation (2015)

    """
    # observed C
    C = count_matrix_mult(dtrajs, lag, sliding=True, sparse=True)
    # statistical inefficiencies
    si = statistical_inefficiencies(dtrajs, lag, C=C, truncate_acf=truncate_acf, mact=mact)
    # effective element-wise counts
    Ceff = C.multiply(si)
    # averaging
    if average.lower() == 'row':
        # reduction factor by row
        factor = np.array(Ceff.sum(axis=1) / np.maximum(1.0, C.sum(axis=1)))
        # row-based effective counts
        Ceff = scipy.sparse.csr_matrix(C.multiply(factor))
    elif average.lower() == 'all':
        # reduction factor by all
        factor = Ceff.sum() / C.sum()
        Ceff = scipy.sparse.csr_matrix(C.multiply(factor))
    # else: by element, we're done.

    return Ceff
