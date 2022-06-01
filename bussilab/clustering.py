"""
Module with some clustering tools
"""

from typing import Optional

import networkx
import numpy as np

from .coretools import Result

class ClusteringResult(Result):
    """Result of a `bussilab.clustering` calculation."""
    def __init__(self,
                *,
                method: str,
                clusters: list,
                weights: Optional[list]
                ):
        self.method = method
        """`str` containing the name of the method used."""
        self.clusters = clusters
        """`list of lists` containing the members of each cluster."""
        self.weights = weights
        """`list` containing the weights of the clusters."""

def max_clique(adj,weights=None,*,min_size=0,max_clusters=None,use_networkit=False):
    """Clustering algorithm used in [Reisser et al, NAR (2020)](https://doi.org/10.1093/nar/gkz1184).

       Parameters
       ----------

       adj : array_like, square matrix

           adj[i,j] contains 1 (or True) if frames i and j are adjacent, 0 (or False) otherwise.

       weights : array_like, optional

           weights[i] contains the weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters.

       use_networkit : bool, optional

           if True, use a networkit implementation that seems to be faster.
           It requires python package networkit to be installed in advance!

       Example
       -------

       ```
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.max_clique(dist<0.7)
       ```
    """
    # weights: optional weights
    # if adj is a graph, it will be copied
    cliques=[]
    ww=[]
    if use_networkit:
        import networkit # pylint: disable=import-error
        graph=networkit.nxadapter.nx2nk(networkx.Graph(adj))
        graph.removeSelfLoops()
        while graph.numberOfNodes()>0:
            if weights is None:
                cl=networkit.clique.MaximalCliques(graph,maximumOnly=True)
                cl.run()
                maxi=cl.getCliques()[0]
                maxw=len(maxi)
            else:
                cl=networkit.clique.MaximalCliques(graph)
                cl.run()
                maxw=0.0
                for i in cl.getCliques():
                    w=np.sum(weights[i])
                    if w > maxw:
                        maxi=i
                        maxw=w
            if maxw<min_size:
                break
            cliques.append(maxi)
            ww.append(maxw)
            if max_clusters is not None:
                if len(cliques)>=max_clusters:
                    break
            for i in maxi:
                graph.removeNode(i)
    else:
        graph=networkx.Graph(adj)
        while graph.number_of_nodes()>0:
            maxw=0.0
            for i in networkx.algorithms.clique.find_cliques(graph):
                if weights is not None:
                    w=np.sum(weights[i])
                else:
                    w=len(i)
                if w > maxw:
                    maxi=i
                    maxw=w
            if maxw<min_size:
                break
            cliques.append(maxi)
            ww.append(maxw)
            if max_clusters is not None:
                if len(cliques)>=max_clusters:
                    break
            graph.remove_nodes_from(maxi)
    return ClusteringResult(method="max_clique",clusters=cliques, weights=ww)

def daura(adj,weights=None,*,min_size=0,max_clusters=None):
    """Clustering algorithm introduced in Daura et al, Angew. Chemie (1999).

       Parameters
       ----------

       adj : array_like, square matrix

           adj[i,j] contains 1 (or True) if frames i and j are adjacent, 0 (or False) otherwise.

       weights : array_like, optional

           weights[i] contains the weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters.

       Example
       -------

       ```
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.daura(dist<0.7)
       ```
    """
    adj=adj.copy()  # take a copy (adj is modified while clustering)
    indexes=np.arange(len(adj))
    clusters=[]
    ww=[]
    if weights is not None:
        weights = weights.copy()  # take a copy (weights is modified while clustering)
    while len(indexes)>0:
        if weights is not None:
            d=np.sum(adj*weights,axis=0)
        else:
            d=np.sum(adj,axis=0)
        n=np.argmax(d)
        if d[n]<min_size:
            break
        ww.append(d[n])
        ii=np.where(adj[n]>0)[0]
        clusters.append(indexes[ii])
        if max_clusters:
            if len(clusters) >= max_clusters:
                break
        adj=np.delete(adj,ii,axis=0)
        adj=np.delete(adj,ii,axis=1)
        if weights is not None:
            weights=np.delete(weights,ii)
        indexes=np.delete(indexes,ii)
    return ClusteringResult(method="daura",clusters=clusters, weights=ww)

def qt(distances,cutoff,weights=None,*,min_size=0,max_clusters=None):
    """Quality threshold clustering.

       The method is explained in the [original paper](https://doi.org/10.1101/gr.9.11.1106).
       The implementation has been adapted from [this one](https://github.com/rglez/QT),
       which is also released under a GPL licence.
       Thus, if you use this algorithm please cite [this article](https://doi.org/10.1021/acs.jcim.9b00558),
       which also discusses the imporant differences between this algorithm and the Daura et al algorithm
       in the context of analysing molecular dynamics simulations.

       The implementation included here, at variance with the original one, allows passing weights
       and can be used with arbitrary metrics.

       Parameters
       ----------

       distances : array_like, square matrix

           distances[i,j] contains the distance between i and j frame.

       cutoff : number

           maximum distance for two frames to be included in the same cluster

       weights : array_like, optional

           weights[i] contains the weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters.

       WARNING: irrespectively of min_size, the current implementation does not report
       clusters with a single member

       Example
       -------

       ```
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.qt(dist,0.7)
       ```
    """
    clusters=[]
    ww=[]
    import numpy.ma as ma
    if weights is not None:
        weights=weights.copy()
    matrix=distances.copy()
    np.fill_diagonal(matrix,0)
    N=len(matrix)
    matrix[matrix > cutoff] = np.inf
    matrix[matrix == 0] = np.inf
    if weights is None:
        degrees = (matrix < np.inf).sum(axis=0)
    else:
        degrees = np.sum((matrix < np.inf) * weights,axis=0)

    # =============================================================================
    # QT algotithm
    # =============================================================================

    clusters_arr = np.ndarray(N, dtype=np.int64)
    clusters_arr.fill(-1)

    ncluster = 0
    while True:
        # This while executes for every cluster in trajectory ---------------------
        len_precluster = 0
        while True:
            # This while executes for every potential cluster analyzed ------------
            biggest_node = degrees.argmax()
            precluster = []
            precluster.append(biggest_node)
            candidates = np.where(matrix[biggest_node] < np.inf)[0]
            next_ = biggest_node
            distances = matrix[next_][candidates]
            while True:
                # This while executes for every node of a potential cluster -------
                next_ = candidates[distances.argmin()]
                precluster.append(next_)
                post_distances = matrix[next_][candidates]
                mask = post_distances > distances
                distances[mask] = post_distances[mask]
                if (distances == np.inf).all():
                    break
            degrees[biggest_node] = 0
            # This section saves the maximum cluster found so far -----------------
            if weights is None:
                new_len_precluster=len(precluster)
            else:
                new_len_precluster=np.sum(weights[precluster])

            if new_len_precluster > len_precluster:
                len_precluster = new_len_precluster
                max_precluster = precluster
                degrees = ma.masked_less(degrees, len_precluster)
            if not degrees.max():
                break
        # General break if min_size is reached -------------------------------------
        if len_precluster < min_size:
            break

        # ---- Store cluster frames -----------------------------------------------
        clusters_arr[max_precluster] = ncluster
        ncluster += 1

        clusters.append(max_precluster)
        ww.append(len_precluster)

        if max_clusters is not None:
            if ncluster >= max_clusters:
                break


        # ---- Update matrix & degrees (discard found clusters) -------------------
        matrix[max_precluster, :] = np.inf
        matrix[:, max_precluster] = np.inf

        if weights is None:
            degrees = (matrix < np.inf).sum(axis=0)
        else:
            degrees = np.sum((matrix < np.inf) * weights,axis=0)
        if (degrees == 0).all():
            break
    return ClusteringResult(method="qt",clusters=clusters, weights=ww)
