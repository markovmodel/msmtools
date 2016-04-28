
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

import math
import numpy as np

from . sampler_nrev import SamplerNonRev
from . sampler_rev import SamplerRev
from . sampler_revpi import SamplerRevPi


class TransitionMatrixSampler(object):

    def __init__(self, C, reversible=False, mu=None, P0=None, nsteps=1, prior='sparse'):

        if not prior == 'sparse':
            raise ValueError("Only Sparse prior is currenty implemented")

        self.C = C

        # distinguish the sampling cases and initialize accordingly
        if reversible:
            if mu is None:
                if nsteps is None:
                    # use sqrt(n) as a rough guess for the decorrelation time
                    nsteps = math.sqrt(np.shape(C)[0])
                self.sampler = SamplerRev(C, P0=P0)
            else:
                if nsteps is None:
                    nsteps = 6  # because we have observed autocorrelation times of about 3.
                self.sampler = SamplerRevPi(C, mu, P0=P0)
        else:
            if mu is None:
                nsteps = 1  # just force to 1, because this is independent sampling
                self.sampler = SamplerNonRev(C-1.0)
            else:
                msg = """Non reversible sampling with fixed stationary vector not implemented"""
                raise ValueError(msg)

        # remember number of steps to decorrelate between samples
        self.nsteps = nsteps

    def sample(self, nsamples=1, return_statdist=False, call_back=None, n_jobs=None):
        if nsamples == 1:
            return self.sampler.sample(N=self.nsteps, return_statdist=return_statdist)
        else:
            N = self.C.shape[0]

            # init pool
            import joblib
            if n_jobs is None:
                import multiprocessing
                n_jobs = multiprocessing.cpu_count()

            pool = joblib.Parallel(n_jobs=n_jobs)

            if n_jobs > 1 and call_back is not None:
                assert callable(call_back), "no valid call back function given! Was '%s'" % call_back
                import joblib.parallel
                joblib.parallel.CallBack = call_back
            elif n_jobs == 1 and call_back is not None:
                assert callable(call_back), "no valid call back function given! Was '%s'" % call_back
                def _print(msg, msg_args):
                    # NOTE: this is a ugly hack, because if we only use one job,
                    # we do not get the joblib callback interface, as a workaround
                    # we use the Parallel._print function, which is called with
                    # msg_args = (done_jobs, total_jobs)
                    if len(msg_args) == 2:
                        call_back()
                pool._print = _print
                # NOTE: verbose has to be set, otherwise our print hack does not work.
                pool.verbose = 50

            task_iter = (joblib.delayed(_sample_worker)(sampler=self.sampler,
                                                        N=self.nsteps,
                                                        return_statdist=return_statdist)
                         for _ in range(nsamples))

            res = pool(task_iter)

            if return_statdist:
                P_samples = np.array([e[0] for e in res])
                pi_samples = np.array([e[1] for e in res])
                return P_samples, pi_samples
            else:
                P_samples = np.array(res)
                return P_samples

# wrapper needed for joblib
def _sample_worker(sampler, N, return_statdist):
    return sampler.sample(N=N, return_statdist=return_statdist)
