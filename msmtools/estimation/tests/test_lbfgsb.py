import unittest

import numpy as np
from numpy import array, asarray, float64, int32, zeros
import ratematrix
from msmtools.util.lbfgsb import fmin_l_bfgs_b

def func(x):
    f = 0.25 * (x[0] - 1) ** 2
    for i in range(1, x.shape[0]):
        f += (x[i] - x[i-1] ** 2) ** 2
    f *= 4
    return f

def grad(x):
    g = zeros(x.shape, float64)
    t1 = x[1] - x[0] ** 2
    g[0] = 2 * (x[0] - 1) - 16 * x[0] * t1
    for i in range(1, g.shape[0] - 1):
        t2 = t1
        t1 = x[i + 1] - x[i] ** 2
        g[i] = 8 * t2 - 16*x[i] * t1
    g[-1] = 8 * t1
    return g

def func_and_grad(x):
    return func(x), grad(x)


class TestLBFGSB(unittest.TestCase):
    def setUp(self):
        self.factr = 1e7
        self.pgtol = 1e-5

        n = 25
        self.m = 10

        self.bounds = [(None, None)] * n
        for i in range(0, n, 2):
            self.bounds[i] = (1.0, 100)
        for i in range(1, n, 2):
            self.bounds[i] = (-100, 100)

        self.x0 = zeros((n,), float64)
        self.x0[:] = 3

    def test_fmin(self):
        x, f, d = fmin_l_bfgs_b(func_and_grad, self.x0, approx_grad=0,
                                m=self.m, factr=self.factr, pgtol=self.pgtol)

        self.assertTrue(d['warnflag']==0)

    def test_maxls(self):
        x, f, d = fmin_l_bfgs_b(func_and_grad, self.x0, approx_grad=0,
                                m=self.m, factr=self.factr, pgtol=self.pgtol,
                                maxls=1)

        self.assertTrue(d['warnflag']!=0)
        self.assertTrue(d['task']==b'ABNORMAL_TERMINATION_IN_LNSRCH')

if __name__ == '__main__':
    unittest.main()
