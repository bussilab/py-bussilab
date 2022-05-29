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
        ref=[[31, 30, 28, 29, 27, 25, 26, 23, 24, 22, 20, 21, 18, 19, 16, 17, 13, 14, 15, 10, 11, 12, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4], [38, 39, 40, 41, 42, 43, 44, 37, 35, 36, 34, 33, 32], [45, 48, 49, 46, 47]]
        # compare sets, since order is irrelevant
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=np.array(range(50))+1
        cl=max_clique(adj,weights)
        self.assertEqual(cl.weights,[546, 546, 99, 69, 15])
        ref=[[45, 38, 39, 40, 41, 42, 43, 44, 37, 35, 36, 47, 46], [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 10, 11, 12, 5, 6, 7, 8, 9, 32], [48, 49], [33, 34], [0, 1, 2, 3, 4]]
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
        ref=[[31, 30, 28, 29, 27, 25, 26, 23, 24, 22, 20, 21, 18, 19, 16, 17, 13, 14, 15, 10, 11, 12, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4], [38, 39, 40, 41, 42, 43, 44, 37, 35, 36, 34, 33, 32], [45, 48, 49, 46, 47]]
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
            self.assertEqual(cl.weights,[32, 13, 3, 2])
            ref=[[31, 30, 28, 29, 27, 25, 26, 23, 24, 22, 20, 21, 18, 19, 16, 17, 13, 14, 15, 10, 11, 12, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4], [38, 39, 40, 41, 42,
            43, 44, 37, 35, 36, 34, 45, 46], [48, 49, 47], [32,33]]
            # compare sets, since order is irrelevant
            for i in range(len(cl.clusters)):
                self.assertEqual(set(ref[i]),set(cl.clusters[i]))

            weights=np.array(range(50))+1
            cl=max_clique(adj,weights,use_networkit=True)
            self.assertEqual(cl.weights,[546, 546, 99, 69, 15])
            ref=[[32, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [47, 35, 45, 36, 43, 46, 37, 42, 38, 41, 39, 40, 44], [48, 49], [33, 34], [0, 4, 1, 2, 3]]
            for i in range(len(cl.clusters)):
                self.assertEqual(set(ref[i]),set(cl.clusters[i]))


if __name__ == "__main__":
    unittest.main()

