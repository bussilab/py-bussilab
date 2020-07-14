import unittest

from bussilab import potts
from bussilab.coretools import TestCase

class TestPotts(TestCase):
    def run_potts(self,fullmatrix):
        import numpy as np
        m=potts.Model(3,fullmatrix=fullmatrix)
        h=np.array((-1.0,0.0,1.0))
        J=np.array(((0.0,0.3,-0.2),
                    (0.3,0.0,-0.6),
                    (-0.2,-0.6,0.0)))
        averages=m.compute(h,J)[1]
        infer=m.infer(averages)
        h_,J_=infer.h,infer.J
        print(averages)
        print(infer)
        self.assertTrue(infer.success)
        self.assertAlmostEqual(np.sum((h-h_)**2),0.0)
        self.assertAlmostEqual(np.sum((J-J_)**2),0.0)

# Temporarily removed:
    def test_potts1(self):
        self.run_potts(True)

# Temporarily removed:
    def test_potts2(self):
        self.run_potts(False)

if __name__ == "__main__":
    unittest.main()
