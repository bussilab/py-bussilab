import unittest

from bussilab.wham import wham
from bussilab.coretools import TestCase

class TestWham(TestCase):
    def test_wham(self):
        import numpy as np
        a = np.exp(wham(np.array([[1, 10, 7], [2, 9, 6], [3, 8, 5]])).logW)
        self.assertAlmostEqual(np.sum((a-np.array([0.41728571, 0.39684866, 0.18586563]))**2), 0.0)
        b = np.exp(wham(np.array([[1, 10], [2, 9], [3, 8]]), traj_weight=(1, 2)).logW)
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

    def test_wham2(self):
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

if __name__ == "__main__":
    unittest.main()
