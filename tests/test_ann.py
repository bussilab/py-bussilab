import unittest

import os
import numpy as np

from bussilab.ann import ANN
from bussilab.coretools import TestCase
from bussilab.coretools import cd

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

class TestANN(TestCase):
    def test_ann1(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10],cuda=False)),1e-9)
    def test_ann2(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10],cuda=False)),1e-9)
    def test_ann3(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10,10],cuda=False)),1e-8)
    def test_ann4(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,8,6,4,2],cuda=False)),1e-8)

    def test_ann1r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10],cuda=False,activation='relu')),1e-9)
    def test_ann2r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10],cuda=False,activation='relu')),1e-9)
    def test_ann3r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,10,10],cuda=False,activation='relu')),1e-8)
    def test_ann4r(self):
        np.random.seed(1977)
        self.assertLess(derivatives(ANN([10,8,6,4,2],cuda=False,activation='relu')),1e-8)

    def test_small(self):
        ann=ANN([2,1],cuda=False)
        ann.setpar(np.array([1.0]*ann.npar))
        d=ann.deriv(np.array([1.0]*2))
        self.assertAlmostEqual(d[0],4.048587351573742)
        self.assertAlmostEqual(d[1][0][0,0],0.9525741268224333)
        self.assertAlmostEqual(d[1][1][0],3.048587351573742)
        self.assertAlmostEqual(d[2][0][0],0.9525741268224333)
        self.assertAlmostEqual(d[2][1][0],1.0)

    def test_small2(self):
        ann=ANN([3,2],activation="relu",cuda=False)
        ann.setpar(0.1*np.array(range(ann.npar)))

        a=ann.apply((1,2,3))
        self.assertIsInstance(a,float)
        self.assertAlmostEqual(a,4.61)
        aa=ann.apply([(1,2,3),(4,5,6)])
        self.assertIsInstance(aa,np.ndarray)
        self.assertEqual(aa.shape,(2,))
        self.assertAlmostEqual(aa[0],4.61)
        self.assertAlmostEqual(aa[1],7.58)

        d=ann.deriv((1,2,3))
        self.assertIsInstance(d,tuple)
        self.assertEqual(len(d),3)
        self.assertIsInstance(d[0],float)
        self.assertAlmostEqual(d[0],4.61)
        self.assertIsInstance(d[1],list)
        self.assertEqual(len(d[1]),2)
        self.assertEqual(d[1][0].shape,(3,2))
        self.assertAlmostEqual(d[1][0][0,0],0.6)
        self.assertAlmostEqual(d[1][0][0,1],0.7)
        self.assertAlmostEqual(d[1][0][1,0],1.2)
        self.assertAlmostEqual(d[1][0][1,1],1.4)
        self.assertAlmostEqual(d[1][0][2,0],1.8)
        self.assertAlmostEqual(d[1][0][2,1],2.1)
        self.assertEqual(d[1][1].shape,(2,1))
        self.assertAlmostEqual(d[1][1][0,0],2.4)
        self.assertAlmostEqual(d[1][1][1,0],3.1)
        self.assertIsInstance(d[2],list)
        self.assertEqual(len(d[2]),2)
        self.assertEqual(d[2][0].shape,(2,))
        self.assertAlmostEqual(d[2][0][0],0.6)
        self.assertAlmostEqual(d[2][0][1],0.7)
        self.assertEqual(d[2][1].shape,(1,))
        self.assertAlmostEqual(d[2][1][0],1.0)
        self.assertAlmostEqual(d[2][0][1],0.7)

    def backprop(self,layers=None,activation="softplus"):
        if layers is None:
            layers=[10,8,6,4,2]
        np.random.seed(1977)
        ann=ANN(layers,activation=activation,random_weights=True)
        traj=np.random.normal(size=(1000,ann.narg))
        d=ann.derpar(traj)
        reference=np.matmul(d[0],d[1])
        state=ann.forward(traj)
        der=ann.backward_par(state.f,state)
        self.assertAlmostEqual(np.sum((reference-der)**2),0.0)

    def test_backprop1(self):
        self.backprop([10,10,10,10],"softplus")
    def test_backprop2(self):
        self.backprop([10,8,6,4,2],"softplus")
    def test_backprop1r(self):
        self.backprop([10,10,10,10],"relu")
    def test_backprop2r(self):
        self.backprop([10,8,6,4,2],"relu")

    def test_plumed(self):
        ann=ANN([3,2],cuda=False)
        ann.setpar(np.array(range(ann.npar)))
        with cd(os.path.dirname(os.path.abspath(__file__))):
            ann.dumpPlumed("ann_plumed.dat","ann")
            ann.dumpPlumed("ann_plumed_combine.dat","combine","pref","x1,x2,x3")
            self.assertEqualFile("ann_plumed.dat")
            self.assertEqualFile("ann_plumed_combine.dat")
            os.remove("ann_plumed.dat")
            os.remove("ann_plumed_combine.dat")

try:
    import cudamat
    _has_cudamat=True
except ModuleNotFoundError:
    _has_cudamat=False

if _has_cudamat:
    class TestCuda(unittest.TestCase):
        def _test_layers(self,layers,activation='softplus'):
            np.random.seed(1977)
            x=np.random.normal(size=layers[0])
            a=ANN(layers,cuda=False,activation=activation,random_weights=True)
            d=a.derpar(x)
            dc=ANN(layers,cuda=True,activation=activation,random_weights=True).setpar(a.getpar()).derpar(x)
            self.assertAlmostEqual(d[0],dc[0],places=5)
            for i in range(len(d[1])):
                self.assertAlmostEqual(d[1][i],dc[1][i],places=5)

        def _test_layers_vec(self,layers,activation='softplus'):
            np.random.seed(1977)
            x=np.random.normal(size=(10,layers[0]))
            a=ANN(layers,cuda=False,activation=activation,random_weights=True)
            d=a.derpar(x)
            dc=ANN(layers,cuda=True,activation=activation,random_weights=True).setpar(a.getpar()).derpar(x)
            for i in range(len(d[0])):
                self.assertAlmostEqual(d[0][i],dc[0][i],places=5)
            for i in range(len(d[1])):
                for j in range(len(d[1][i])):
                    self.assertAlmostEqual(d[1][i,j],dc[1][i,j],places=5)

        def test_ann1(self):
            self._test_layers([10])
        def test_ann2(self):
            self._test_layers([10,10])
        def test_ann3(self):
            self._test_layers([10,10,10])
        def test_ann4(self):
            self._test_layers([10,8,6,4,2])

        def test_ann1r(self):
            self._test_layers([10],activation='relu')  # might fail due to relu discontinuity
        def test_ann2r(self):
            self._test_layers([10,10],activation='relu')  # might fail due to relu discontinuity
        def test_ann3r(self):
            self._test_layers([10,10,10],activation='relu')  # might fail due to relu discontinuity
        def test_ann4r(self):
            self._test_layers([10,8,6,4],activation='relu')  # might fail due to relu discontinuity

        def test_ann1_vec(self):
            self._test_layers_vec([10])
        def test_ann2_vec(self):
            self._test_layers_vec([10,10])
        def test_ann3_vec(self):
            self._test_layers_vec([10,10,10])
        def test_ann4_vec(self):
            self._test_layers_vec([10,8,6,4,2])

        def test_ann1r_vec(self):
            self._test_layers_vec([10],activation='relu')  # might fail due to relu discontinuity
        def test_ann2r_vec(self):
            self._test_layers_vec([10,10],activation='relu')  # might fail due to relu discontinuity
        def test_ann3r_vec(self):
            self._test_layers_vec([10,10,10],activation='relu')  # might fail due to relu discontinuity
        def test_ann4r_vec(self):
            self._test_layers_vec([10,8,6,4],activation='relu')  # might fail due to relu discontinuity

        def backprop(self,layers=None,activation="softplus"):
            if layers is None:
                layers=[10,8,6,4,2]
            np.random.seed(1977)
            ann=ANN(layers,activation=activation,random_weights=True,cuda=True)
            traj=np.random.normal(size=(1000,ann.narg))
            d=ann.derpar(traj)
            reference=np.matmul(d[0],d[1])
            state=ann.forward(traj)
            der=ann.backward_par(state.f,state)
            self.assertAlmostEqual(np.sum((reference-der)**2),0.0,places=5)

        def test_backprop1(self):
            self.backprop([10,10,10,10],"softplus")
        def test_backprop2(self):
            self.backprop([10,8,6,4,2],"softplus")
        def test_backprop1r(self):
            self.backprop([10,10,10,10],"relu")
        def test_backprop2r(self):
            self.backprop([10,8,6,4,2],"relu")


if __name__ == "__main__":
    unittest.main()
