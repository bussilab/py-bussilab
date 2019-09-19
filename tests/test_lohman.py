import unittest
import numpy as np
from bussilab.lohman import lohman
from bussilab.coretools import TestCase

class TestLohman(TestCase):
    def test_lohman(self):
        t = np.linspace(0, 4, 5)
        a = lohman(t, ku=1, kd=0.1)
        aref = np.array((0.0, 0.60648083, 0.80836077, 0.87556076, 0.89792969))
        b = lohman(t, ku=2, kd=0.1)
        bref = np.array((0.0, 0.83575578, 0.93809945, 0.95063209, 0.95216679))
        c = lohman(t, ku=1, kd=0.2)
        cref = np.array((0.0, 0.58233816, 0.75773504, 0.81056356, 0.82647521))
        d = lohman(t, ku=1, kd=0.1, n=2)
        dref = np.array((0.0, 0.24873614, 0.53341313, 0.69537387, 0.77165485))
        e = lohman(t, ku=1, kd=0.1, boundaries=(0.3, 0.7))
        eref = np.array((0.3, 0.54259233, 0.62334431, 0.6502243, 0.65917188))
        self.assertAlmostEqual(np.sum((a-aref)**2), 0.0)
        self.assertAlmostEqual(np.sum((b-bref)**2), 0.0)
        self.assertAlmostEqual(np.sum((c-cref)**2), 0.0)
        self.assertAlmostEqual(np.sum((d-dref)**2), 0.0)
        self.assertAlmostEqual(np.sum((e-eref)**2), 0.0)

if __name__ == "__main__":
    unittest.main()
