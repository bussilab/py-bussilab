import unittest

import numpy as np
import scipy.spatial.distance as distance
from bussilab.coretools import TestCase


class TestClustering(TestCase):
    def test_maxclique(self):
        from bussilab.clustering import max_clique
        data=(np.array(range(50))**2).reshape(-1,1)
        dist=distance.squareform(distance.pdist(data))
        adj=np.int_(dist<1000)
        cl=max_clique(adj)
        self.assertEqual(cl.method,"max_clique")
        self.assertEqual(cl.weights,[32, 13, 5])
        ref=[range(32), range(32,45), range(45,50)]
        # compare sets, since order is irrelevant
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=np.hstack((np.ones(32),3*np.ones(13),8*np.ones(5)))
        cl=max_clique(adj,weights)
        self.assertEqual(cl.weights,[61, 32, 18])
        ref=[range(38,50), range(0,32), range(32,38)]
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_daura(self):
        from bussilab.clustering import daura
        data=(np.array(range(50))**2).reshape(-1,1)
        dist=distance.squareform(distance.pdist(data))
        adj=np.int_(dist<500)
        cl=daura(adj)
        self.assertEqual(adj.shape,dist.shape)  # check that adj has not been resized
        self.assertEqual(cl.method,"daura")
        self.assertEqual(cl.weights,[32, 13, 5])
        ref=[range(32), range(32,45), range(45,50)]
        # compare sets, since order is irrelevant
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        data=np.array([0,1,10,10,10]).reshape(-1,1)
        dist=distance.squareform(distance.pdist(data))
        adj=np.int_(dist<3)
        cl=daura(adj)
        self.assertEqual(cl.method,"daura")
        self.assertEqual(cl.weights,[3,2])
        ref=[[2,3,4],[0,1]]
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        data=np.array([0,1,10]).reshape(-1,1)
        weights=np.array([1,1,3])
        dist=distance.squareform(distance.pdist(data))
        adj=np.int_(dist<3)
        cl=daura(adj,weights)
        self.assertEqual(cl.method,"daura")
        self.assertEqual(cl.weights,[3,2])
        ref=[[2],[0,1]]
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        data=np.array([10,0,1]).reshape(-1,1)
        weights=np.array([3,1,1])
        dist=distance.squareform(distance.pdist(data))
        adj=np.int_(dist<3)
        cl=daura(adj,weights)
        self.assertEqual(adj.shape,dist.shape)  # check that adj has not been resized
        self.assertEqual(weights.shape,(len(dist),))  # check that adj has not been resized
        self.assertEqual(cl.method,"daura")
        self.assertEqual(cl.weights,[3,2])
        ref=[[0],[1,2]]
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

try:
    import networkit
    _has_networkit=True
except ModuleNotFoundError:
    _has_networkit=False

if _has_networkit:
    class TestClusteringNetworkit(TestCase):
        def test_maxclique(self):
            from bussilab.clustering import max_clique
            data=(np.array(range(50))**2).reshape(-1,1)
            dist=distance.squareform(distance.pdist(data))
            adj=np.int_(dist<1000)
            cl=max_clique(adj,use_networkit=True)
            self.assertEqual(cl.method,"max_clique")
            print("nk",cl)
            self.assertEqual(cl.weights,[32, 13, 3, 2])
            ref=[range(32), range(34,47), range(47,50),(32,33)]
            # compare sets, since order is irrelevant
            for i in range(len(cl.clusters)):
                self.assertEqual(set(ref[i]),set(cl.clusters[i]))

            weights=np.hstack((np.ones(32),3*np.ones(13),8*np.ones(5)))
            cl=max_clique(adj,weights,use_networkit=True)
            self.assertEqual(cl.weights,[61, 32, 18])
            ref=[range(38,50), range(0,32), range(32,38)]
            for i in range(len(cl.clusters)):
                self.assertEqual(set(ref[i]),set(cl.clusters[i]))


if __name__ == "__main__":
    unittest.main()

