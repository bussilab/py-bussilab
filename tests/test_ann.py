import unittest

import numpy as np

from bussilab.ann import ANN

def derivatives(ann,n=100,prefactor=1e-10,vector=True):
    p=[]
    x=np.random.normal(size=ann.narg)
    start=np.random.normal(size=ann.npar)
    ann.setpar(start)
    gp=ann.getpar()
    assert np.all(np.equal(gp,start))
    if vector:
        v0,deriv=ann.derparVec(x.reshape((1,-1)))
    else:
        v0,deriv=ann.derpar(x)
    for i in range(n):
        test=prefactor*np.random.normal(size=ann.npar)
        ann.setpar(start+test)
        v1=ann.apply(x)
        p.append((v1-v0,np.matmul(deriv,test)))
    p=np.array(p)/prefactor
    return np.average((p[:,0]-p[:,1])**2)

class TestANN(unittest.TestCase):
    def test_ann1(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10])),1e-10)
    def test_ann2(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10])),1e-10)
    def test_ann3(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10,10])),1e-8)
    def test_ann4(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,8,6,4,2])),1e-8)

    def test_ann1r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10],activation='relu')),1e-10)
    def test_ann2r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10],activation='relu')),1e-10)
    def test_ann3r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10,10],activation='relu')),1e-8)
    def test_ann4r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,8,6,4,2],activation='relu')),1e-8)

    def test_small(self):
        ann=ANN([2,1])
        ann.setpar(np.array([1.0]*ann.npar))
        d=ann.deriv(np.array([1.0]*2))
        self.assertAlmostEqual(d[0],4.048587351573742)
        self.assertAlmostEqual(d[1][0][0,0],0.9525741268224333)
        self.assertAlmostEqual(d[1][1][0],3.048587351573742)
        self.assertAlmostEqual(d[2][0][0],0.9525741268224333)
        self.assertAlmostEqual(d[2][1][0],1.0)

if __name__ == "__main__":
    unittest.main()
