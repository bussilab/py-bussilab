import unittest

from bussilab.maxent import maxent

class TestMaxent(unittest.TestCase):
    def test_maxent(self):
        import numpy as np
        traj = [[2], [4]]

        m = maxent(traj, [2.5])
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-0.28768, -1.38630])**2), 0.0)
        self.assertAlmostEqual((m.averages[0]-2.5)**2, 0.0)
        self.assertAlmostEqual((m.lambdas[0]-0.549306)**2, 0.0)

        m = maxent(traj, [2.5], logW=[0, -5])
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-0.28768, -1.38630])**2), 0.0)
        self.assertAlmostEqual((m.averages[0]-2.5)**2, 0.0)
        self.assertAlmostEqual((m.lambdas[0]- -1.95069378)**2, 0.0)

        m = maxent(traj, [2.5], logW=[0, -5], l2=1)
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-0.01697795, -4.08431654])**2), 0.0)
        self.assertAlmostEqual((m.averages[0]-2.03367)**2, 0.0)
        self.assertAlmostEqual((m.lambdas[0]- -0.46633)**2, 0.0)

        def reg(x):
            return 0.5*0.5*np.sum(x**2), 0.5*x
        m = maxent(traj, [2.5], logW=[0, -5], l2=[0.5], regularization=reg)
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-0.01697795, -4.08431654])**2), 0.0)
        self.assertAlmostEqual((m.averages[0]-2.03367)**2, 0.0)
        self.assertAlmostEqual((m.lambdas[0]- -0.46633)**2, 0.0)

    def test_maxent2d(self):
        import numpy as np
        traj = [[0, 0], [1, 0], [0, 1]]

        m = maxent(traj, (0.3, 0.4))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.20396686, -1.20398582, -0.91628543])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.3, 0.4])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[1.89607144e-05, -2.87681431e-01])**2), 0.0)

        m = maxent(traj, ((-np.inf, 0.3), (0.4, np.inf)))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.20396686, -1.20398582, -0.91628543])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.3, 0.4])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[1.89607144e-05, -2.87681431e-01])**2), 0.0)

        m = maxent(traj, ((-np.inf, 0.3), 0.4))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.20396686, -1.20398582, -0.91628543])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.3, 0.4])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[1.89607144e-05, -2.87681431e-01])**2), 0.0)

        m = maxent(traj, ((-np.inf, 0.3), 0.4), logW=(0, 1, 2))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.20396686, -1.20398582, -0.91628543])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.3, 0.4])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[0.99998443, 1.71233637])**2), 0.0)

        m = maxent(traj, ((0.3, np.inf), (-np.inf, 0.4)))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.0986122, -1.0986122, -1.0986122])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.33333333, 0.333333333])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[0.0, 0.0])**2), 0.0)

        m = maxent(traj, ((0.3, 1.0), (0.0, 0.4)))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.0986122, -1.0986122, -1.0986122])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.33333333, 0.333333333])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[0.0, 0.0])**2), 0.0)

        m = maxent(traj, ((0.3, 1.0), (-np.inf, 0.4)))
        self.assertTrue(m.success)
        self.assertAlmostEqual(np.sum((m.logW_ME-[-1.0986122, -1.0986122, -1.0986122])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.averages-[0.33333333, 0.333333333])**2), 0.0)
        self.assertAlmostEqual(np.sum((m.lambdas-[0.0, 0.0])**2), 0.0)

try:
    import cudamat
    _has_cudamat=True
except ModuleNotFoundError:
    _has_cudamat=False

if _has_cudamat:
    class TestCuda(unittest.TestCase):
        def test(self):
            import numpy as np
            traj = [[0, 0], [1, 0], [0, 1]]
            m = maxent(traj, ((0.3, 1.0), (-np.inf, 0.4)),cuda=True)
            self.assertTrue(m.success)
            self.assertAlmostEqual(np.sum((m.logW_ME-[-1.0986122, -1.0986122, -1.0986122])**2), 0.0)
            self.assertAlmostEqual(np.sum((m.averages-[0.33333333, 0.333333333])**2), 0.0)
            self.assertAlmostEqual(np.sum((m.lambdas-[0.0, 0.0])**2), 0.0)

            cu_traj=cudamat.CUDAMatrix(np.array(traj))
            m = maxent(cu_traj, ((0.3, 1.0), (-np.inf, 0.4)),cuda=True)
            self.assertTrue(m.success)
            self.assertAlmostEqual(np.sum((m.logW_ME-[-1.0986122, -1.0986122, -1.0986122])**2), 0.0)
            self.assertAlmostEqual(np.sum((m.averages-[0.33333333, 0.333333333])**2), 0.0)
            self.assertAlmostEqual(np.sum((m.lambdas-[0.0, 0.0])**2), 0.0)

if __name__ == "__main__":
    unittest.main()
