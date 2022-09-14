"""
Module with some clustering tools
"""

from typing import Optional

import networkx
import numpy as np

from .coretools import Result
from .coretools import import_numba_jit

numba_jit=import_numba_jit()

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

       WARNING: important fix in v0.0.39 for version with weights

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
            d=np.sum(adj*weights[:,np.newaxis],axis=0)
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

@numba_jit
def _qt_outer(distances,next_,cutoff):
    candidates=np.empty(len(distances),dtype='int')
    n=0
    for i in range(next_):
        if distances[next_,i]<cutoff:
           candidates[n]=i
           n+=1
    for i in range(next_+1,len(distances)):
        if distances[next_,i]<cutoff:
           candidates[n]=i
           n+=1
    return candidates[:n]

@numba_jit
def _qt_inner(distances,dist_from_cluster,candidates,cutoff,weights):
    if len(dist_from_cluster)<1:
        return (-1,np.inf)
    next_i=0
    minval=dist_from_cluster[0]
    weight_minval=weights[candidates[0]]
    N=len(dist_from_cluster)
    for i in range(1,N):
        val=dist_from_cluster[i]
        if val<minval or (val==minval and weights[candidates[i]]>weight_minval):
            next_i=i
            minval=val
            weight_minval=weights[candidates[i]]
    if minval>cutoff:
        return (-1,minval)
    next_=candidates[next_i]
    for i in range(N):
        val=distances[next_][candidates[i]]
        if dist_from_cluster[i]<val:
            dist_from_cluster[i]=val
    dist_from_cluster[next_i]=np.inf
    return (next_,minval)

def qt(distances,cutoff,weights=None,*,min_size=0,max_clusters=None):
    """Quality threshold clustering.

       The method is explained in the [original paper](https://doi.org/10.1101/gr.9.11.1106).
       The implementation has been adapted from [this one](https://github.com/rglez/QT),
       which is also released under a GPL licence.
       Thus, if you use this algorithm please cite [this article](https://doi.org/10.1021/acs.jcim.9b00558),
       which also discusses the important differences between this algorithm and the Daura et al algorithm
       in the context of analysing molecular dynamics simulations.
       Additionally, mention which exact version of the bussilab package you used.

       The implementation included here, at variance with the original one, allows passing weights
       and can be used with arbitrary metrics. In addition, it also reports clusters of size 1
       (unless one passes max_clusters>1). The code is optimized when compared with the original one,
       and speed can be further increased by passing `np.array(distances,dtype='float32')` to distances.

       WARNING: important fix in v0.0.39 for version with weights

       As of version v0.0.40, clusters with the same number of members are prioritized based on their
       diameter (smaller diameter gets the priority). This is expected to make non-weighted calculations
       more reproducible, but might change some results when compared with previous versions.
       In addition, when growing a single candidate cluster, it two points are at the same
       distance from the growing cluster the one with higher weight is chosen. With these
       priorities, any choice where either (a) weights are float or (b) distances are float
       should lead to deterministing clustering irrespective of roundoff errors, assuming
       floats are never identical.


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

       Example
       -------

       ```
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.qt(dist,0.7)

       clustering.qt(np.array(dist,dtype='float32')) # should be slightly faster
       ```
    """
    clusters=[]
    ww=[]
    N=len(distances)
    if weights is None:
        weights=np.ones(N,dtype="int")
    else:
        weights=weights.copy()
    distances=distances.copy()
    np.fill_diagonal(distances,0.0)
    indexes=np.arange(N)

    ncluster=0
    while len(weights)>0:
        degrees=np.sum((distances < cutoff)*weights[:,np.newaxis],axis=0)
        sorted_indexes=np.argsort(-degrees)

        cluster_size=0
        diameter=0.0
        cluster=[]
        i_degrees=0
        for i_degrees in range(len(sorted_indexes)):

            if degrees[sorted_indexes[i_degrees]] < cluster_size: # optimization
                break

            next_=sorted_indexes[i_degrees]
            precluster = [next_]
            new_cluster_diameter=0.0
            candidates=_qt_outer(distances,next_,cutoff)
            dist_from_cluster=distances[next_][candidates]
            while True:
                (next_,minval)=_qt_inner(distances,dist_from_cluster,candidates,cutoff,weights)
                if(next_<0):
                    break
                precluster.append(next_)
                if minval > new_cluster_diameter:
                    new_cluster_diameter = minval
            new_cluster_size=np.sum(weights[precluster])
            # pick largest cluster (sum of weights)
            # if same size, pick the most compact one (smaller diameter)
            if new_cluster_size > cluster_size or (new_cluster_size == cluster_size and new_cluster_diameter < diameter):
                cluster_size = new_cluster_size
                cluster = precluster
                diameter = new_cluster_diameter

        if cluster_size < min_size:
            break

        ncluster += 1

        clusters.append(indexes[cluster])
        ww.append(cluster_size)

        if max_clusters is not None:
            if ncluster >= max_clusters:
                break

        # see https://stackoverflow.com/questions/63563151/how-to-remove-columns-and-rows-at-certain-indexes-in-numpy-at-the-same-time
        idx = list(set(range(len(distances))).difference(cluster))
        distances=distances[np.ix_(idx,idx)]
        weights=weights[idx]
        indexes=indexes[idx]
    return ClusteringResult(method="qt",clusters=clusters, weights=ww)
