import unittest
from contextlib import redirect_stderr
from io import StringIO

from bussilab.wham import wham
from bussilab.coretools import TestCase

class TestWham(TestCase):
    def test_wham(self):
        import numpy as np
        a = np.exp(wham(np.array([[1, 10, 7], [2, 9, 6], [3, 8, 5]])).logW)
        self.assertAlmostEqual(np.sum((a-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        b = np.exp(wham(np.array([[1, 10], [2, 9], [3, 8]]), traj_weight=(0.1, 0.2)).logW)
        self.assertAlmostEqual(np.sum((b-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        c = np.exp(wham(np.array([[3, 5],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4]])).logW)
        self.assertAlmostEqual(np.sum((c-np.array([
            0.06421006, 0.09357899, 0.09357899, 0.09357899, 0.09357899,
            0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899
        ]))**2), 0.0)
        d = np.exp(wham(np.array([[3, 5], [4, 4]]), frame_weight=(1, 10)).logW)
        self.assertAlmostEqual(np.sum((d-np.array([[0.06421006, 0.93578994]]))**2), 0.0)
        e = np.exp(wham(np.array([[1, 5, 14], [2, 4.5, 12], [3, 4, 10]]),T=[1.0,0.5,2.0]).logW)
        self.assertAlmostEqual(np.sum((e-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)

    def test_wham1s(self):
        import numpy as np
        # check the substituting version
        a = np.exp(wham(np.array([[1, 10, 7], [2, 9, 6], [3, 8, 5]]),method="substitute").logW)
        self.assertAlmostEqual(np.sum((a-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        b = np.exp(wham(np.array([[1, 10], [2, 9], [3, 8]]), traj_weight=(1, 2),method="substitute").logW)
        self.assertAlmostEqual(np.sum((b-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        c = np.exp(wham(np.array([[3, 5],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4],
                                  [4, 4]])).logW)
        self.assertAlmostEqual(np.sum((c-np.array([
            0.06421006, 0.09357899, 0.09357899, 0.09357899, 0.09357899,
            0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899
        ]))**2), 0.0)
        d = np.exp(wham(np.array([[3, 5], [4, 4]]), frame_weight=(1, 10),method="substitute").logW)
        self.assertAlmostEqual(np.sum((d-np.array([[0.06421006, 0.93578994]]))**2), 0.0)
        e = np.exp(wham(np.array([[1, 5, 14], [2, 4.5, 12], [3, 4, 10]]),T=[1.0,0.5,2.0],method="substitute").logW)
        self.assertAlmostEqual(np.sum((e-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)

    def test_wham2(self):
        # test passing lists instead of np arrays
        import numpy as np
        a = np.exp(wham([[1, 10, 7], [2, 9, 6], [3, 8, 5]]).logW)
        self.assertAlmostEqual(np.sum((a-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        b = np.exp(wham([[1, 10], [2, 9], [3, 8]], traj_weight=(1, 2)).logW)
        self.assertAlmostEqual(np.sum((b-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        c = np.exp(wham([[3, 5], 
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4]]).logW)
        self.assertAlmostEqual(np.sum((c-np.array([
            0.06421006, 0.09357899, 0.09357899, 0.09357899, 0.09357899,
            0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899
        ]))**2), 0.0)
        d = np.exp(wham(np.array([[3, 5], [4, 4]]), frame_weight=(1, 10)).logW)
        self.assertAlmostEqual(np.sum((d-np.array([[0.06421006, 0.93578994]]))**2), 0.0)

    def test_wham3(self):
        from bussilab import cli
        from bussilab.coretools import cd
        import os
        with cd(os.path.dirname(os.path.abspath(__file__))):
            try:
                os.remove("wham1.out")
            except FileNotFoundError:
                pass
            cli.cli("wham -b wham1.in -o wham1.out")
            self.assertEqualFile("wham1.out")
            os.remove("wham1.out")

    def test_wham4(self):
        import numpy as np
        # check if the algorithm is numerically robust if one of the biases if heavily offset
        for diffZ in (0.0,1e1,1e2,1e3,1e4,1e5,1e6):
            w=wham(2.0*np.array([[1,1-diffZ],[2,2-diffZ]]),T=2.0)
            self.assertAlmostEqual(w.logZ[1]-w.logZ[0],diffZ)
            self.assertAlmostEqual(w.logW[1]-w.logW[0],1.0)

    def test_wham4s(self):
        import numpy as np
        # check if the algorithm is numerically robust if one of the biases if heavily offset, with substitution
        for diffZ in (0.0,1e1,1e2,1e3,1e4,1e5,1e6):
            w=wham(2.0*np.array([[1,1-diffZ],[2,2-diffZ]]),method="substitute",T=2.0)
            self.assertAlmostEqual(w.logZ[1]-w.logZ[0],diffZ)
            self.assertAlmostEqual(w.logW[1]-w.logW[0],1.0)

    def test_wham5(self):
        import numpy as np
        # check if the algorithm is numerically robust if one frame has weight much smaller than another
        for diffW in (0.0,1e1,1e2,1e3,1e4,1e5,1e6):
            w=wham(2.0*np.array([[0.0,0.0],[diffW,diffW]]),T=2.0)
            self.assertAlmostEqual(w.logW[1]-w.logW[0],diffW)
            self.assertAlmostEqual(w.logZ[1]-w.logZ[0],0.0)

    def test_wham6(self):
        import numpy as np
        large=1e+3
        bias=np.array([
            [0,large],
            [0,0],
            [large,0]
        ]
        )
        bias1=np.array(bias)
        bias2=bias-bias[:,0][:,np.newaxis]
        w1=wham(bias1)
        w2=wham(bias2)
        self.assertAlmostEqual(w1.logZ[0],w1.logZ[1])
        self.assertAlmostEqual(w2.logZ[0],w2.logZ[1])

    def test_wham_restart(self):
        import numpy as np
        w0=wham(((0,10),(-4,11)),T=0.2)
        self.assertGreaterEqual(w0.nfev,8)
        w1=wham(((0,10),(-4,11)),T=0.2,logW=w0.logW)
        self.assertLessEqual(w1.nfev,2)
        w2=wham(((0,10),(-4,11)),T=0.2,logZ=w0.logZ)
        self.assertLessEqual(w2.nfev,2)

    def test_normalize_modes(self):
        import numpy as np
        bias = [[0.0], [1.0]]
        unnormalized = np.array([0.0, 1.0])
        normalized = unnormalized - np.logaddexp(0.0, 1.0)
        for method in ("minimize", "substitute"):
            with self.subTest(method=method):
                np.testing.assert_allclose(
                    wham(bias, method=method, normalize=False).logW, unnormalized)
                np.testing.assert_allclose(
                    wham(bias, method=method, normalize=True).logW, normalized)
                np.testing.assert_allclose(
                    wham(bias, method=method, normalize="log").logW, normalized)

    def test_minimize_options(self):
        import numpy as np
        bias = [[1, 10, 7], [2, 9, 6], [3, 8, 5]]
        expected = wham(bias).logW
        for options in ({"jac": True}, {"method": "BFGS"}):
            with self.subTest(options=options):
                np.testing.assert_allclose(
                    wham(bias, minimize_opt=options).logW, expected)
        with self.assertRaisesRegex(ValueError, "minimize_opt.*jac"):
            wham(bias, minimize_opt={"jac": False})

    def test_invalid_options(self):
        bias = [[0.0], [1.0]]
        with self.assertRaisesRegex(ValueError, "method should be"):
            wham(bias, method="unknown")
        with self.assertRaisesRegex(ValueError, "normalize should be"):
            wham(bias, normalize="unknown")

    def test_verbose(self):
        bias = [[0.0], [1.0]]
        for method in ("minimize", "substitute"):
            with self.subTest(method=method):
                output = StringIO()
                with redirect_stderr(output):
                    wham(bias, method=method, verbose=True)
                self.assertIn("WHAM: start", output.getvalue())
                self.assertIn("WHAM: end", output.getvalue())
                if method == "substitute":
                    self.assertIn("WHAM: iteration", output.getvalue())

if __name__ == "__main__":
    unittest.main()
