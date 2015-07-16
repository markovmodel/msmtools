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

r"""Transition matrix sampling for revrsible stochastic matrices with fixed stationary vector.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import

# cython: profile=False, boundscheck=False, wraparound=False
cimport cython
import numpy as np
cimport numpy as np

from . mle_trev_given_pi import mle_trev_given_pi
from msmtools.analysis import is_connected

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "sample_revpi.h":
    
    double sample_quad(double xkl, double xkk, double xll,
                       double ckl, double clk, double ckk, double cll,
                       double bk, double bl)

    double sample_quad_rw(double xkl, double xkk, double xll,
                          double ckl, double clk, double ckk, double cll,
                          double bk, double bl)

    void update_sparse(double *X, double *C, double *b, size_t n,
                       size_t * I, size_t * J, size_t n_idx)


cdef extern from "_rnglib.h":
    void initialize()
    void set_initial_seed(int g1, int g2)

cdef class XSampler:
    
    def __init__(self):
        """Seed the generator upon init"""
        initialize()
        set_initial_seed(np.random.randint(1, 2147483563),
                         np.random.randint(1, 2147483399))       

    def update_revpi(self, np.ndarray[DTYPE_t, ndim=2] X , np.ndarray[DTYPE_t, ndim=2] C,
                     np.ndarray[DTYPE_t, ndim=1] b):
        cdef size_t k, l, M
        cdef double xkl, xkl_new
        M = C.shape[0]
        for k in range(M):
            for l in range(k):
                if (C[k, l] + C[l, k]) > 0.0:
                    xkl = 1.0*X[k, l]
                    xkl_new = sample_quad(X[k, l], X[k, k], X[l, l],
                                          C[k, l], C[l, k], C[k, k], C[l, l],
                                          b[k], b[l])
                    X[k, l] = xkl_new
                    X[k, k] += (xkl - xkl_new)
                    X[l, k] = xkl_new
                    X[l, l] += (xkl - xkl_new)
                    
                    xkl = 1.0*X[k, l]
                    xkl_new = sample_quad_rw(X[k, l], X[k, k], X[l, l],
                                             C[k, l], C[l, k], C[k, k], C[l, l],
                                             b[k], b[l])
                    X[k, l] = xkl_new
                    X[k, k] += (xkl - xkl_new)
                    X[l, k] = xkl_new
                    X[l, l] += (xkl - xkl_new)

    def update_revpi_sparse(self, np.ndarray[DTYPE_t, ndim=2] X ,
                            np.ndarray[DTYPE_t, ndim=2] C,
                            np.ndarray[DTYPE_t, ndim=1] b,
                            np.ndarray[np.int_t, ndim=1] I,
                            np.ndarray[np.int_t, ndim=1] J):
        cdef size_t n, n_idx
        n = C.shape[0]
        n_idx = I.shape[0]
        update_sparse(<double*> X.data, <double*> C.data, <double*> b.data,
                       n, <size_t*> I.data, <size_t*> J.data, n_idx)
    

class SamplerRevPi:

    def __init__(self, C, pi, P0=None, P_mle=None, eps=0.1):
        # set_counter(0, 0, 0, 0)       
        self.C = 1.0*C
        self.pi = pi

        if P_mle is None:
            # P_mle = tmatrix(C, reversible=True, mu=pi, eps=1e-6)
            P_mle = mle_trev_given_pi(C, pi)

        if P0 is None:
            cdiag = np.diag(C)
            """Entries with cii=0"""
            ind = (cdiag == 0)
            """Add counts, s.t. cii+bii>0 for all i"""
            bdiag = np.zeros_like(cdiag)
            bdiag[ind] = 1.0
            B = np.diag(bdiag)
            # P0 = tmatrix(C+B, reversible=True, mu=pi, eps=1e-6)
            P0 = mle_trev_given_pi(C+B, pi)

        """Diagonal prior parameters"""
        b = np.zeros(C.shape[0])

        cii = C[np.diag_indices(C.shape[0])]
        
        """Zero diagonal entries of C"""
        ind1 = np.isclose(cii, 0.0)
        b[ind1] = eps
        
        """Non-zero diagonal entries of P0"""
        pii0 = P_mle[np.diag_indices(P_mle.shape[0])]
        ind2 = (pii0 > 0.0)

        """Find elements pii0>0 and cii=0"""
        ind3 = np.logical_and(ind1, ind2)
        b[ind3] = 1.0

        self.b = b        

        """Initial state of the chain"""
        self.X = pi[:, np.newaxis] * P0       
        self.X /= self.X.sum()

        """Check for valid input"""
        self.check_input()

        """Set up index arrays"""
        self.I, self.J = np.where((self.C + self.C.T)>0.0)

        """Init the xsampler"""
        self.xsampler = XSampler()

    def check_input(self):
        if not np.all(self.C>=0):
            raise ValueError("Count matrix contains negative elements")
        if not is_connected(self.C):
            raise ValueError("Count matrix is not connected")
        if not np.all(self.X>=0.0):
            raise ValueError("P0 contains negative entries")
        if not np.allclose(self.X, self.X.T):
            raise ValueError("P0 is not reversible")
        """Check sparsity pattern - ignore diagonal"""
        C_sym = self.C + self.C.T
        X_sym = self.X + self.X.T    
        ind = np.diag_indices(C_sym.shape[0])                
        C_sym[ind] = 0.0
        X_sym[ind] = 0.0
        iC, jC = np.where( C_sym > 0.0 )
        iV, jV = np.where( X_sym > 0.0 )
        if not np.array_equal(iC, iV):
            raise ValueError('Sparsity patterns of C and X are different.')
        if not np.array_equal(jC, jV):
            raise ValueError('Sparsity patterns of C and X are different.')
    
    def update(self, N=1):
        for i in range(N):
            """Do 6 updates to decorrelate"""
            for j in range(6):
                self.xsampler.update_revpi(self.X, self.C, self.b)

    # def update(self, N=1):
    #     self.xsampler.update_revpi_sparse(self.X, self.C, self.b, self.I, self.J)

    def sample(self, N=1, return_statdist=False):
        self.update(N=N)
        P = self.X/self.pi[:, np.newaxis]
        if return_statdist:
            return P, self.pi
        else:
            return P
