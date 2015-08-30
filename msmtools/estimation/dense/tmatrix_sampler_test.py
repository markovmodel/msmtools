r"""Unit tests for the covariance module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
from __future__ import absolute_import
from __future__ import division

import unittest
import numpy as np

from msmtools.estimation import tmatrix
from msmtools.estimation.dense.tmatrix_sampler import TransitionMatrixSampler


class TestSamplerNonReversible(unittest.TestCase):

    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(42)

        self.C = 1.0*np.array([[7048, 6, 2], [6, 2, 3], [2, 3, 2933]])

        """Mean in the asymptotic limit, N_samples -> \infty"""
        alpha = self.C
        alpha0 = self.C.sum(axis=1)

        self.mean = alpha/alpha0[:, np.newaxis]
        self.var = alpha * (alpha0[:,np.newaxis] - alpha)/\
            (alpha0**2 * (alpha0 + 1.0))[:,np.newaxis]

        self.N = 1000

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_mean(self):
        """Create sampler object"""
        sampler = TransitionMatrixSampler(self.C, reversible=False)

        """Compute sample mean"""
        mean = np.zeros_like(self.C)
        for i in range(self.N):
            mean += sampler.sample()
        mean *= 1.0/self.N

        """Check if sample mean and true mean fall into the 2\sigma intervall"""
        self.assertTrue(np.all( np.abs(mean - self.mean) <= 2.0*np.sqrt(self.var/self.N) ))


class TestSamplerReversible(unittest.TestCase):

    def setUp(self):
        self.C = 1.0*np.array([[7048, 6, 0], [6, 2, 3], [0, 3, 2933]])
        self.P_mle = tmatrix(self.C, reversible=True)
        self.N = 1000

    def tearDown(self):
        pass

    def test_mean(self):
        """Create sampler object"""
        sampler = TransitionMatrixSampler(self.C, reversible=True)    

        sample = np.zeros((self.N, 3, 3))
        for i in range(self.N):
            sample[i, :, :] = sampler.sample()
        mean = np.mean(sample, axis=0)
        std = np.std(sample, axis=0)            

        """Check if sample mean and MLE agree within the sample standard deviation"""
        self.assertTrue(np.all( np.abs(mean - self.P_mle) <= std) )

class TestSamplerReversiblePi(unittest.TestCase):
    
    def setUp(self):
        self.C = 1.0*np.array([[7048, 6, 0], [6, 2, 3], [0, 3, 2933]])
        self.pi = np.array([ 0.70532947,  0.00109989,  0.29357064])
        self.P_mle = tmatrix(self.C, reversible=True, mu=self.pi)

        self.N = 1000

    def tearDown(self):
        pass

    def test_mean(self):
        """Create sampler object"""
        sampler = TransitionMatrixSampler(self.C, reversible=True, mu=self.pi, nsteps=10)    

        sample = np.zeros((self.N, 3, 3))
        for i in range(self.N):
            sample[i, :, :] = sampler.sample()
        mean = np.mean(sample, axis=0)
        std = np.std(sample, axis=0)        

        """Check if sample mean and MLE agree within the sample standard deviation"""
        self.assertTrue(np.all( np.abs(mean - self.P_mle) <= std) )

if __name__=="__main__":
    unittest.main()


