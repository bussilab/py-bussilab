import unittest

import numpy as np
import scipy.spatial.distance as distance
from scipy import sparse
from bussilab.coretools import TestCase

def dataset1():
    return np.array([[-0.27434351,  0.06897978,  0.691931  ,  0.03051945, -1.11740313],
                     [ 0.09421232,  0.18027066, -0.06561068,  0.43135584, -0.43940911],
                     [-0.58521568,  0.03401204, -0.63839148, -0.23572534,  0.7680671 ],
                     [ 0.76687884,  0.62421027,  0.16370233, -0.899945  ,  0.68917803],
                     [-1.03296177, -1.64284264, -1.14054955, -0.46923336,  0.47111502],
                     [ 0.66999821, -0.48762024, -0.10909411,  0.50337187,  0.32333498],
                     [-1.5843071 , -0.77184176, -1.27124076,  0.46066538, -0.48326112],
                     [-0.34907915, -0.78747764,  1.0505906 ,  0.16638586,  1.37716574],
                     [ 0.83328731,  1.01717983,  0.10425336, -0.71342138, -1.49533519],
                     [ 0.04637906,  2.22325827, -0.33923953, -0.72341812, -0.10958039],
                     [ 1.02688336, -0.06777232, -0.54296664,  1.16736651,  1.25059246],
                     [-0.49533634, -0.26167575,  1.31880093,  1.70218493,  0.43341073],
                     [ 0.76752508, -0.27790873, -1.31025055, -2.67449964,  0.11026565],
                     [-0.46181482, -0.55080617, -1.11513591,  0.53394519, -0.22060322],
                     [-0.17277899,  0.0679858 ,  0.04163297,  0.05483895, -1.32811042],
                     [ 0.55942438, -0.45916286, -2.24796214, -0.32409745,  0.37803287],
                     [ 1.87984435, -0.52573726, -0.15719837, -0.78858451, -1.08018692],
                     [-0.25623155,  0.85477056, -1.0210906 , -1.17935635, -0.08414587],
                     [-1.19320835,  0.38967952,  1.70916784,  1.06780696, -2.24571906],
                     [ 0.44823174,  1.59230506,  0.15871062, -0.23835377,  0.59421649],
                     [-1.06998977, -0.91881022,  0.05298366, -0.97827771, -1.78408201],
                     [-0.81556567, -1.50470659, -1.38748846,  0.06731462, -1.01353324],
                     [-1.68093824, -1.00525244,  0.03008928, -0.03406712,  1.42217833],
                     [-0.81782272, -1.66081026, -0.39005011, -1.05360066,  0.74259682],
                     [ 1.29354339, -0.15176496,  1.28761518, -0.61495318,  0.18550871],
                     [ 1.55105343, -1.25720131, -0.27978451,  0.29854249,  0.17464664],
                     [-0.20532629, -0.5092539 ,  0.5594995 , -0.26360377, -0.48954807],
                     [-0.05945412, -1.66385354, -1.84581852,  0.07426876,  0.23254214],
                     [ 1.24005529,  1.2027705 ,  0.48165738,  0.7507277 ,  0.55896436],
                     [-0.50173855,  0.39622259, -1.57123979, -1.85249664,  1.4424275 ]])

def dataset1_weights():
    return np.array([1.00937788, 0.99180958, 1.00778711, 0.98892454, 1.00444056,
                     0.99750995, 1.01300244, 0.99761303, 0.98371752, 0.98464331,
                     0.99546134, 1.01373022, 0.99337099, 1.01127106, 0.99460744,
                     0.99053065, 1.01448288, 1.00005337, 1.0102084 , 0.9901373 ,
                     0.99089065, 0.99662465, 0.99441189, 0.99437434, 1.00473962,
                     1.00137577, 0.98325373, 1.01462485, 1.01370019, 0.99486246])

def dataset2():
    return np.array([[-0.59339408,  1.45733255],
                     [-0.42087971,  0.20388607],
                     [ 0.89720348, -0.53101914],
                     [-0.85465884,  0.80789103],
                     [ 0.32262436, -0.72368994]])

def dataset2_weights():
    return np.array([1.    , 1.0025, 1.005 , 1.0075, 1.01  ])

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

    def test_maxclique_zero_weights(self):
        from bussilab.clustering import max_clique
        cl=max_clique(np.eye(2,dtype=bool),weights=[0,0])
        self.assertEqual([set(cluster) for cluster in cl.clusters],[{0},{1}])
        self.assertEqual(cl.weights,[0,0])

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

    def test_daura_many_small_clusters(self):
        from bussilab.clustering import daura
        adj=np.eye(120,dtype=bool)
        weights=np.arange(1,121,dtype=np.int8)
        cl=daura(adj,weights)
        self.assertEqual([cluster.tolist() for cluster in cl.clusters],
                         [[i] for i in range(119,-1,-1)])
        self.assertEqual(cl.weights,list(range(120,0,-1)))
        self.assertTrue(np.array_equal(adj,np.eye(120,dtype=bool)))
        self.assertTrue(np.array_equal(weights,np.arange(1,121,dtype=np.int8)))

    def test_daura_large_cluster(self):
        from bussilab.clustering import daura
        adj=np.eye(302,dtype=bool)
        adj[:300,:300]=True
        cl=daura(adj,np.ones(302,dtype=np.int8))
        self.assertEqual(cl.weights,[300,1,1])
        self.assertEqual([cluster.tolist() for cluster in cl.clusters],
                         [list(range(300)),[300],[301]])

    def test_daura_without_self_edges(self):
        from bussilab.clustering import daura
        cl=daura(np.zeros((2,2),dtype=bool))
        self.assertEqual([cluster.tolist() for cluster in cl.clusters],[[0],[1]])
        self.assertEqual(cl.weights,[1,1])
        cl=daura(np.zeros((2,2),dtype=bool),weights=np.array([3.0,2.0]))
        self.assertEqual([cluster.tolist() for cluster in cl.clusters],[[0],[1]])
        self.assertEqual(cl.weights,[3.0,2.0])
        cl=daura(np.array([[0,1],[1,0]],dtype=bool))
        self.assertEqual([cluster.tolist() for cluster in cl.clusters],[[0,1]])
        self.assertEqual(cl.weights,[2])

    def test_daura_large_weight_does_not_hide_remaining_cluster(self):
        from bussilab.clustering import daura
        adj=np.eye(5,dtype=bool)
        for i,j in ((0,1),(0,2),(2,3)):
            adj[i,j]=adj[j,i]=True
        cl=daura(adj,np.array([1e16,1,100,100,0.5]),max_clusters=2)
        self.assertEqual([set(cluster) for cluster in cl.clusters],
                         [{0,2,3},{1}])
        self.assertEqual(cl.weights,[1e16+200,1])

    def test_daura_sparse_matches_dense(self):
        from bussilab.clustering import daura
        rng=np.random.default_rng(542)
        upper=np.triu(rng.random((80,80))<0.08,1)
        adj=upper|upper.T  # no diagonal entries
        weights=rng.uniform(0.5,2.0,len(adj))
        for frame_weights in (None,weights,np.arange(80,dtype=np.int8)):
            expected=daura(adj,frame_weights,max_clusters=12)
            for sparse_adj in (sparse.csr_matrix(adj),
                               sparse.csc_matrix(adj),sparse.coo_matrix(adj)):
                actual=daura(sparse_adj,frame_weights,max_clusters=12)
                self.assertEqual([c.tolist() for c in actual.clusters],
                                 [c.tolist() for c in expected.clusters])
                np.testing.assert_allclose(actual.weights,expected.weights)
                np.testing.assert_array_equal(sparse_adj.toarray(),adj)

    def test_daura_sparse_edge_cases(self):
        from bussilab.clustering import daura
        star=np.zeros((4,4),dtype=bool)
        star[1,[0,2,3]]=True
        star[[0,2,3],1]=True
        self.assertEqual(daura(sparse.csr_matrix(star),max_clusters=1)
                         .clusters[0].tolist(),[0,1,2,3])

        adj=np.eye(5,dtype=bool)
        for i,j in ((0,1),(0,2),(2,3)):
            adj[i,j]=adj[j,i]=True
        weighted=daura(sparse.csr_matrix(adj),
                       np.array([1e16,1,100,100,0.5]),max_clusters=2)
        self.assertEqual([set(cluster) for cluster in weighted.clusters],
                         [{0,2,3},{1}])
        self.assertEqual(weighted.weights,[1e16+200,1])

        # Explicitly stored zeros are not edges; the diagonal is implicit.
        with_zeros=sparse.csr_matrix((np.array([0,1,1]),
                                     (np.array([0,0,1]),np.array([0,1,0]))),
                                    shape=(3,3))
        result=daura(with_zeros)
        self.assertEqual([c.tolist() for c in result.clusters],[[0,1],[2]])
        self.assertEqual(with_zeros.nnz,3)

        unsigned=daura(sparse.csr_matrix((2,2)),
                       weights=np.array([2**63,1],dtype=np.uint64))
        self.assertEqual([c.tolist() for c in unsigned.clusters],[[0],[1]])
        self.assertEqual(unsigned.weights,[2**63,1])

        many=sparse.eye(10000,format="csr",dtype=bool)
        result=daura(many,max_clusters=3)
        self.assertEqual([c.tolist() for c in result.clusters],[[0],[1],[2]])
        self.assertEqual(daura(sparse.csr_matrix((0,0))).clusters,[])
        with self.assertRaises(ValueError):
            daura(sparse.csr_matrix((2,3)))
        with self.assertRaises(ValueError):
            daura(sparse.eye(2,format="csr"),weights=[1])

    def test_daura_neighbor_iterables(self):
        from bussilab.clustering import daura
        rows=[[2,1,2],[0],[3,0],[2],[]]
        adjacency=sparse.csr_matrix([
            [0,1,1,0,0], [1,0,0,0,0], [1,0,0,1,0],
            [0,0,1,0,0], [0,0,0,0,0]])
        expected=daura(adjacency)
        actual=daura(rows)
        self.assertEqual([c.tolist() for c in actual.clusters],
                         [c.tolist() for c in expected.clusters])
        self.assertEqual(actual.weights,expected.weights)
        self.assertEqual(rows,[[2,1,2],[0],[3,0],[2],[]])

        # Both the outer and inner iterables may be consumed only once.
        generated=(iter(row) for row in rows)
        weighted=daura(generated,weights=[2,1,3,4,5],max_clusters=2)
        reference=daura(adjacency,weights=[2,1,3,4,5],max_clusters=2)
        self.assertEqual([c.tolist() for c in weighted.clusters],
                         [c.tolist() for c in reference.clusters])
        self.assertEqual(weighted.weights,reference.weights)

        dense=[[1,0],[0,1]]
        self.assertEqual([c.tolist() for c in daura(dense).clusters],[[0],[1]])
        self.assertEqual([c.tolist() for c in daura(iter(dense)).clusters],[[0,1]])

    def test_daura_invalid_neighbor_iterables(self):
        from bussilab.clustering import daura
        for rows in ([[2],[]], [[-1],[]], [[1.0],[]], [[True],[]],
                     [[1],None]):
            with self.assertRaises(ValueError):
                daura(rows)

    def test_zero_max_clusters(self):
        from bussilab.clustering import daura, max_clique, qt
        for result in (max_clique(np.eye(1,dtype=bool),max_clusters=0),
                       daura(np.eye(1,dtype=bool),max_clusters=0),
                       qt(np.zeros((1,1)),1,max_clusters=0)):
            self.assertEqual(result.clusters,[])
            self.assertEqual(result.weights,[])

    def test_empty_clustering_input(self):
        from bussilab.clustering import daura, max_clique, qt
        for result in (max_clique(np.empty((0,0),dtype=bool)),
                       daura(np.empty((0,0),dtype=bool)),
                       qt(np.empty((0,0)),1)):
            self.assertEqual(result.clusters,[])
            self.assertEqual(result.weights,[])

    def test_invalid_clustering_inputs(self):
        from bussilab.clustering import daura, max_clique, qt
        for method, matrix in ((max_clique,np.eye(2,dtype=bool)),
                               (daura,np.eye(2,dtype=bool)),
                               (lambda matrix,**kwargs: qt(matrix,1,**kwargs),np.zeros((2,2)))):
            with self.assertRaises(ValueError):
                method(matrix,weights=[1])
            with self.assertRaises(ValueError):
                method(matrix,weights=[1,np.nan])
            with self.assertRaises(ValueError):
                method(matrix,weights=[1,np.inf])
            with self.assertRaises(ValueError):
                method(matrix,weights=[1,-1])
            with self.assertRaises(ValueError):
                method(matrix,max_clusters=-1)
        with self.assertRaises(ValueError):
            daura(np.ones((2,3)))
        with self.assertRaises(ValueError):
            qt(np.ones((2,3)),1)
        with self.assertRaises(ValueError):
            qt(np.zeros((2,2)),np.nan)

    def test_qt(self):
        from bussilab.clustering import qt
        data=(np.array(range(50))**2).reshape(-1,1)
        dist=distance.squareform(distance.pdist(data))
        cl=qt(dist,1000)
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,[32, 13, 5])
        ref=[range(32), range(32,45), range(45,50)]
        # compare sets, since order is irrelevant
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=np.hstack((np.ones(32),3*np.ones(13),8*np.ones(5)))
        cl=qt(dist,1000,weights)
        self.assertEqual(cl.weights,[61, 32, 18])
        ref=[range(38,50), range(0,32), range(32,38)]
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt_large_cluster_read_only_distances(self):
        from bussilab.clustering import qt
        dist=np.full((262,262),10.0,dtype=np.float32)
        dist[:260,:260]=0.1
        np.fill_diagonal(dist,50.0)
        dist.flags.writeable=False
        cl=qt(dist,0.5,np.ones(262,dtype=np.int8),max_clusters=2)
        self.assertEqual(cl.weights,[260,1])
        self.assertEqual(set(cl.clusters[0]),set(range(260)))
        self.assertEqual(set(cl.clusters[1]),{260})
        self.assertEqual(dist[0,0],50.0)

    def test_qt_large_weight_does_not_hide_remaining_cluster(self):
        from bussilab.clustering import qt
        dist=np.array([[0,0.5,0.5,2],
                       [0.5,0,2,2],
                       [0.5,2,0,2],
                       [2,2,2,0]])
        weights=np.array([1e16,1,2,0.5])
        cl=qt(dist,1,weights,max_clusters=2)
        self.assertEqual([set(cluster) for cluster in cl.clusters],
                         [{0,2},{1}])
        self.assertEqual(cl.weights,[1e16+2,1])

    def test_qt_zero_weights(self):
        from bussilab.clustering import qt
        cl=qt(np.zeros((2,2)),1,weights=np.zeros(2))
        self.assertEqual([set(cluster) for cluster in cl.clusters],[{0,1}])
        self.assertEqual(cl.weights,[0])

    def test_qt_cutoff_is_inclusive(self):
        from bussilab.clustering import qt
        dist=np.array([[0.0,1.0],[1.0,0.0]])
        cl=qt(dist,1.0)
        self.assertEqual([set(cluster) for cluster in cl.clusters],[{0,1}])
        self.assertEqual(cl.weights,[2])
        cl=qt(np.zeros((2,2)),0.0)
        self.assertEqual([set(cluster) for cluster in cl.clusters],[{0,1}])

    def test_max_clique2(self):
        from bussilab.clustering import max_clique
        dist=distance.squareform(distance.pdist(dataset1()))
        # skip this since it is not reproducible on python 3.6/3.7
        #cl=max_clique(dist<3)
        #ref=[[1, 26, 2, 9, 0, 3, 8, 17, 19, 28, 14], [5, 13, 4, 23, 21, 27, 25, 15], [11, 7, 24, 10], [12, 16], [20, 18], [6, 22], [29]]
        #refw=[11, 8, 4, 2, 2, 2, 1]
        #self.assertEqual(cl.method,"max_clique")
        #self.assertEqual(cl.weights,refw)
        #for i in range(len(cl.clusters)):
        #    self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=max_clique(dist<3,weights)
        ref=[[1, 15, 5, 13, 2, 23, 4, 21, 27, 17, 6], [26, 0, 14, 28, 3, 25, 16, 24], [11, 7, 10], [29, 9, 19], [20, 18], [22], [12], [8]]
        refw=[11.0220285541, 8.01046204991, 3.006804594578, 2.969643064484, 2.001099056174, 0.994411893694, 0.993370993929, 0.983717515293]
        self.assertEqual(cl.method,"max_clique")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_max_clique3(self):
        from bussilab.clustering import max_clique
        dist=distance.squareform(distance.pdist(dataset1()))
        cl=max_clique(dist<3,min_size=4)
        ref=[[1, 26, 2, 9, 0, 3, 8, 17, 19, 28, 14], [5, 13, 4, 23, 21, 27, 25, 15], [11, 7, 24, 10]]
        refw=[11, 8, 4]
        self.assertEqual(cl.method,"max_clique")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=max_clique(dist<3,weights,min_size=2.96)
        ref=[[1, 15, 5, 13, 2, 23, 4, 21, 27, 17, 6], [26, 0, 14, 28, 3, 25, 16, 24], [11, 7, 10], [29, 9, 19]]
        refw=[11.0220285541, 8.01046204991, 3.006804594578, 2.969643064484]
        self.assertEqual(cl.method,"max_clique")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_max_clique4(self):
        from bussilab.clustering import max_clique
        dist=distance.squareform(distance.pdist(dataset1()))
        cl=max_clique(dist<3,max_clusters=2)
        ref=[[1, 26, 2, 9, 0, 3, 8, 17, 19, 28, 14], [5, 13, 4, 23, 21, 27, 25, 15]]
        refw=[11, 8]
        self.assertEqual(cl.method,"max_clique")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=max_clique(dist<3,weights,max_clusters=2)
        ref=[[1, 15, 5, 13, 2, 23, 4, 21, 27, 17, 6], [26, 0, 14, 28, 3, 25, 16, 24]]
        refw=[11.0220285541, 8.01046204991]
        self.assertEqual(cl.method,"max_clique")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt2(self):
        from bussilab.clustering import qt
        dist=distance.squareform(distance.pdist(dataset1()))
        dist[0,0]=1e-3 # this is to check that qt() ignores the diagonal terms
        cl=qt(dist,3)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [3, 19, 28, 24, 8, 0, 26, 14], [11, 7, 10], [16, 25], [29, 12], [18, 20], [9], [22]]
        refw=[11, 8, 3, 2, 2, 2, 1, 1]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=qt(dist,3,weights)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [16, 25, 24, 26, 3, 14, 0, 28], [11, 7, 10], [19, 9, 8], [18, 20], [29, 12], [22]]
        refw=[11.02202855414899, 8.010462049910068, 3.0068045945788784, 2.958498123817103, 2.0010990561741044, 1.9882334498903202, 0.99441189]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt3(self):
        from bussilab.clustering import qt
        dist=distance.squareform(distance.pdist(dataset1()))
        cl=qt(dist,3,min_size=4)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [3, 19, 28, 24, 8, 0, 26, 14]]
        refw=[11, 8]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=qt(dist,3,weights,min_size=2.96)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [16, 25, 24, 26, 3, 14, 0, 28], [11, 7, 10]]
        refw=[11.02202855414899, 8.010462049910068, 3.0068045945788784]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt4(self):
        from bussilab.clustering import qt
        dist=distance.squareform(distance.pdist(dataset1()))
        cl=qt(dist,3,max_clusters=2)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [3, 19, 28, 24, 8, 0, 26, 14]]
        refw=[11, 8]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        weights=dataset1_weights()
        cl=qt(dist,3,weights,max_clusters=2)
        ref=[[13, 6, 21, 4, 27, 23, 2, 5, 15, 1, 17], [16, 25, 24, 26, 3, 14, 0, 28]]
        refw=[11.0220285541, 8.01046204991]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))
    def test_qt5(self):
        from bussilab.clustering import qt
        # test with identical points
        dist=distance.squareform(distance.pdist([[1],[1],[1],[2],[2],[3]]))
        cl=qt(dist,0.5)
        ref=[[0,1,2],[3,4],[5],[6]]
        refw=[3,2,1]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

        dist=distance.squareform(distance.pdist([[1],[1],[1],[2],[2],[3],[4],[5]]))
        weights=np.array([1,1,1,2,2,5,0.5,0.1])
        cl=qt(dist,0.5,weights)
        ref=[[5],[3,4],[0,1,2],[6],[7]]
        refw=[5,4,3,0.5,0.1]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt6(self):
        from bussilab.clustering import qt
        dist=distance.squareform(distance.pdist(dataset2()))
        weights=dataset2_weights()
        cl=qt(dist,1.0,weights)
        ref=[[2, 4],[1, 3],[0]]
        refw=[2.015, 2.01, 1.0]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt7(self):
        from bussilab.clustering import qt
        data=np.array([[1],[2.1],[10],[11]])
        dist=distance.squareform(distance.pdist(data))
        cl=qt(dist,2.0)
        ref=[[2, 3],[0, 1]]
        refw=[2,2]
        self.assertEqual(cl.method,"qt")
        self.assertEqual(cl.weights,refw)
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt8(self):
        from bussilab.clustering import qt
        # this is to test that [2,3] gets priority wrt [0,1] since it is more compact
        dist=distance.squareform(distance.pdist(np.array([[0.0],[0.5],[1.0],[1.2],])))
        weights=np.array([1,1,1,1])
        cl=qt(dist,0.6,weights)
        ref=[[2,3],[0,1]]
        refw=[2,2]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
        for i in range(len(cl.clusters)):
            self.assertEqual(set(ref[i]),set(cl.clusters[i]))

    def test_qt9(self):
        from bussilab.clustering import qt
        # this is to test that [1] grows with [2] rather than with [0] since [2] has
        # more weight than [0].
        # notice that [2] grows with [3], which is closer
        # however, [1,2] has weight 2.15 and is preferred over [2,3], which has weight 2.1
        dist=distance.squareform(distance.pdist(np.array([[0.0],[1.0],[2.0],[2.5]])))
        weights=np.array([1,1.05,1.1,1])
        cl=qt(dist,1.2,weights)
        ref=[[1,2],[0],[3]]
        refw=[2.15,1,1]
        self.assertEqual(cl.method,"qt")
        self.assertAlmostEqual(np.sum((cl.weights-np.array(refw))**2),0.0)
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

