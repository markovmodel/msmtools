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

r"""Transition matrix sampling for revrsible stochastic matrices.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: Frank Noe <frank DOT noe AT fu-berlin DOT de>

"""
from __future__ import absolute_import

import numpy as np
import ctypes
cimport numpy as np

from msmtools.analysis import statdist, is_connected
# from msmtools.estimation import is_connected

cdef extern from "sample_rev.h":
    void _update(double* C, double* sumC, double* X, int n, int n_step)

    void _update_sparse(double* C, double* sumC, double* X, double* sumX,
                        int* I, int* J, int n, int n_idx, int n_step)

    double _update_step(double v0, double v1, double v2, 
                        double c0, double c1, double c2, int random_walk_stepsize)

cdef extern from "_rnglib.h":
    void initialize()
    void set_initial_seed(int g1, int g2)

cdef class VSampler:

    def __init__(self):
        """Seed the generator upon init"""
        initialize()
        set_initial_seed(np.random.randint(1, 2147483563),
                         np.random.randint(1, 2147483399))
       
    # def update(self, C, sumC, X, nstep):
    #     n = C.shape[0]
    #     pC    = <double*> np.PyArray_DATA(C)
    #     psumC = <double*> np.PyArray_DATA(sumC)
    #     pX    = <double*> np.PyArray_DATA(X)
    #     # call
    #     _update(pC, psumC, pX, n, nstep)

    def update_sparse(self, C, sumC, X, I, J, nstep):
        n = C.shape[0]
        n_idx = len(I)
        sumX = np.zeros( (n), dtype=ctypes.c_double, order='C' )
        sumX[:] = X.sum(axis=1)

        cdef np.ndarray[int, ndim=1, mode="c"] cI
        cI = np.array( I, dtype=ctypes.c_int, order='C' )
        cdef np.ndarray[int, ndim=1, mode="c"] cJ
        cJ = np.array( J, dtype=ctypes.c_int, order='C' )

        pC    = <double*> np.PyArray_DATA(C)
        psumC = <double*> np.PyArray_DATA(sumC)
        pX    = <double*> np.PyArray_DATA(X)
        psumX = <double*> np.PyArray_DATA(sumX)
        pI    = <int*>    np.PyArray_DATA(cI)
        pJ    = <int*>    np.PyArray_DATA(cJ)
        # call
        _update_sparse(pC, psumC, pX, psumX, pI, pJ, n, n_idx, nstep)


class SamplerRev:
    
    def __init__(self, C, P0=None):
        self.C = 1.0*C

        """Set up initial state of the chain"""
        if P0 is None:
            A = C + C.T
            V0 = A/A.sum()
        else:
            pi0 = statdist(P0)
            V0 = pi0[:,np.newaxis] * P0            
        
        self.V = V0   
        # self.v = self.V.sum(axis=1)
        self.c = self.C.sum(axis=1)
        
        """Check for valid input"""
        self.check_input()

        """Get nonzero indices"""
        self.I, self.J = np.where( (self.C + self.C.T)>0.0 )

        """Init Vsampler"""
        self.vsampler = VSampler()

    def check_input(self):
        if not np.all(self.C>=0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.V>=0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.V, self.V.T):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern"""
        iC, jC = np.where( (self.C+self.C.T)>0 )
        iV, jV = np.where( (self.V+self.V.T)>0 )
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')        

    def update(self, N=1):
        self.vsampler.update_sparse(self.C, self.c, self.V, self.I, self.J, N)

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        P = self.V/self.V.sum(axis=1)[:, np.newaxis]
        if return_statdist:
            nu = 1.0*self.V.sum(axis=1)
            pi = nu/nu.sum()
            return P, pi
        else:
            return P    







