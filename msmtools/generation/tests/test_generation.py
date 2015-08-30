
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
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

'''
@author: noe, trendelkampschroer
'''
from __future__ import absolute_import
from __future__ import division
import unittest
import numpy as np
import msmtools.generation as msmgen
import msmtools.estimation as msmest
import msmtools.analysis as msmana

class Test(unittest.TestCase):

    def setUp(self):
        """Safe random state"""
        self.state = np.random.get_state()
        """Set seed to enforce deterministic behavior"""
        np.random.seed(42)

    def tearDown(self):
        """Reset state"""
        np.random.set_state(self.state)

    def test_trajectory(self):
        P = np.array([[0.9,0.1],
                      [0.1,0.9]])
        N = 1000
        traj = msmgen.generate_traj(P, N, start=0)

        # test shapes and sizes
        assert traj.size == N
        assert traj.min() >= 0
        assert traj.max() <= 1

        # test statistics of transition matrix
        C = msmest.count_matrix(traj,1)
        Pest = msmest.transition_matrix(C)
        assert np.max(np.abs(Pest - P)) < 0.025


    def test_trajectories(self):
        P = np.array([[0.9,0.1],
                      [0.1,0.9]])

        # test number of trajectories
        M = 10
        N = 10
        trajs = msmgen.generate_trajs(P, M, N, start=0)
        assert len(trajs) == M

        # test statistics of starting state
        trajs = msmgen.generate_trajs(P, 1000, 1)
        ss = np.concatenate(trajs).astype(int)
        pi = msmana.stationary_distribution(P)
        piest = msmest.count_states(ss) / 1000.0
        assert np.max(np.abs(pi - piest)) < 0.025

        # test stopping state = starting state
        M = 10
        trajs = msmgen.generate_trajs(P, M, N, start=0, stop=0)
        for traj in trajs:
            assert traj.size == 1

        # test if we always stop at stopping state
        M = 100
        stop = 1
        trajs = msmgen.generate_trajs(P, M, N, start=0, stop=stop)
        for traj in trajs:
            assert traj.size == N or traj[-1] == stop
            assert stop not in traj[:-1]

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()