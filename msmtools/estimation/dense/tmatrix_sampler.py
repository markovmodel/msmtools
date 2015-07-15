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

r"""Transition matrix sampling module. Provides a common class for sampling of 

i) non-reversible transition matrices
ii) reverisble transition matrices
iii) reversible transition matrices with fixed stationary vector

from given data

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: Frank Noe <frank DOT noe AT fu-berlin DOT de>

"""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from . sampler_nrev import SamplerNonRev
from . sampler_rev import SamplerRev
from . sampler_revpi import SamplerRevPi

class TransitionMatrixSampler:
    
    def __init__(self, C, reversible=False, mu=None, P0=None, nsteps=1, prior='sparse'):
        
        if not prior == 'sparse':
            raise ValueError("Only Sparse prior is currenty implemented")        

        self.C = C

        """Number of Gibbs sampling sweeps per returned sample"""
        self.nsteps = nsteps

        if reversible:
            if mu is None:
                self.sampler = SamplerRev(C, P0=P0)
            else:
                self.sampler = SamplerRevPi(C, mu, P0=P0)
        else:
            if mu is None:
                self.sampler = SamplerNonRev(C-1.0)
            else:
                msg = """Non reversible sampling with fixed stationary vector not implemented"""
                raise ValueError(msg)

    def sample(self, nsamples=1, return_statdist=False):
        nsteps = self.nsteps
        if nsamples==1:
            return self.sampler.sample(N=nsteps, return_statdist=return_statdist)
        else:
            N = self.C.shape[0]
            P_samples = np.zeros((nsamples, N, N))
            if return_statdist:
                pi_samples = np.zeros((nsamples, N))
                for i in range(nsamples):
                    P_samples[i, :, :], pi_samples[i, :] = self.sampler.sample(N=nsteps,
                                                                          return_statdist=True)
                return P_samples, pi_samples
            else:
                for i in range(nsamples):
                    P_samples[i, :, :] = self.sampler.sample(N=nsteps, return_statdist=False)
                return P_samples
            
                
            
    
    

