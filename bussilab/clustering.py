"""Cluster frames from a full adjacency or distance matrix.

`max_clique()` and `daura()` use a Boolean adjacency matrix. The former
repeatedly extracts a maximum-weight clique; the latter repeatedly selects
the frame with the most neighbors and removes its neighborhood. `qt()` uses
a distance matrix and builds clusters whose pairwise distances do not exceed
a cutoff. All three methods can use nonnegative frame weights and return a
`ClusteringResult` with cluster members and their total weights.

The input matrix requires memory proportional to the square of the number of
frames. `max_clique()` can also take exponential time on difficult graphs.

## Changelog

- **v0.0.52**
  - Made `daura()` and `qt()` reuse the full input matrix instead of repeatedly
    copying smaller matrices; updated cluster degrees incrementally and
    reduced temporary arrays. The input matrices are left unchanged.
  - Fixed handling of absent adjacency self-edges, zero frame weights,
    floating-point degree cancellation, and `max_clusters=0`. `qt()` now
    includes distances exactly equal to the cutoff.
  - Against commit `4dec1e4`, `daura()` was about 17x faster without weights
    and 21x faster with weights on a 1,000-frame singleton matrix. `qt()` was
    about 1.5x faster without weights and 1.4x faster with weights on 1,200
    random 2D points at cutoff 0.12 (96 and 98 clusters, respectively). These
    are medians of three warm runs on the same machine, using float64
    distances and, in the weighted cases, float64 weights.
- **v0.0.40**
  - In `qt()`, prioritized smaller-diameter clusters when cluster sizes tie
    and higher-weight frames when candidates are equally distant.
- **v0.0.39**
  - Fixed weighted clustering in `daura()` and `qt()`.

"""

from typing import Optional

import networkx
import numpy as np

from .coretools import Result
from .coretools import import_numba_jit

numba_jit=import_numba_jit()

class ClusteringResult(Result):
    """Clusters and their sizes, in the order selected by a clustering method."""
    def __init__(self,
                *,
                method: str,
                clusters: list,
                weights: Optional[list]
                ):
        self.method = method
        """Name of the method: `max_clique`, `daura`, or `qt`."""
        self.clusters = clusters
        """List of clusters, each containing original frame indices."""
        self.weights = weights
        """Total frame weight of each cluster, or its size if unweighted."""

def _validate_weights(weights,size):
    weights=np.asarray(weights)
    if weights.shape!=(size,):
        raise ValueError("weights must contain one value per frame")
    if not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError("weights must be finite and nonnegative")
    return weights

def max_clique(adj,weights=None,*,min_size=0,max_clusters=None,use_networkit=False):
    """Clustering algorithm used in [Reisser et al, NAR (2020)](https://doi.org/10.1093/nar/gkz1184).

       Parameters
       ----------

       adj : array_like, square matrix

           adj[i,j] contains 1 (or True) if frames i and j are adjacent, 0 (or False) otherwise.

       weights : array_like, optional

           weights[i] contains the finite, nonnegative weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters; zero returns an empty result.

       use_networkit : bool, optional

           if True, use a networkit implementation that seems to be faster.
           It requires python package networkit to be installed in advance!

       Example
       -------

       ```
       from bussilab import clustering
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.max_clique(dist<0.7)
       ```
    """
    if max_clusters is not None:
        if max_clusters<0:
            raise ValueError("max_clusters must be nonnegative")
        if max_clusters==0:
            return ClusteringResult(method="max_clique",clusters=[],weights=[])
    if weights is not None:
        weights=_validate_weights(weights,len(adj))
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
                maxw=-np.inf
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
            maxw=-np.inf
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
    """Cluster by repeatedly removing the largest adjacency neighborhood.

       This is the algorithm introduced in Daura et al, Angew. Chemie (1999).
       The input is a symmetric adjacency matrix; self-edges are optional.

       Parameters
       ----------

       adj : array_like, square matrix

           adj[i,j] contains 1 (or True) if frames i and j are adjacent, 0 (or False) otherwise.

       weights : array_like, optional

           weights[i] contains the finite, nonnegative weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters; zero returns an empty result.

       Example
       -------

       ```
       from bussilab import clustering
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.daura(dist<0.7)
       ```
    """
    if max_clusters is not None:
        if max_clusters<0:
            raise ValueError("max_clusters must be nonnegative")
        if max_clusters==0:
            return ClusteringResult(method="daura",clusters=[],weights=[])
    adj=np.asarray(adj)
    if adj.ndim!=2 or adj.shape[0]!=adj.shape[1]:
        raise ValueError("adj must be a square matrix")
    missing_diagonal=np.diag(adj)<=0
    active=np.ones(len(adj),dtype=bool)
    clusters=[]
    ww=[]
    if weights is not None:
        weights=_validate_weights(weights,len(adj))
        # Avoid materializing the weighted adjacency matrix.
        # Match np.sum's accumulator type (notably for bool and small integers).
        product_dtype=np.result_type(adj.dtype,weights.dtype)
        degree_dtype=np.sum(np.empty(0,dtype=product_dtype)).dtype
        if np.issubdtype(degree_dtype,np.floating):
            d=np.empty(len(adj),dtype=degree_dtype)
        else:
            d=np.einsum('ij,i->j',adj,weights,dtype=degree_dtype)
            d+=weights*missing_diagonal
    else:
        d=np.sum(adj,axis=0)
        d+=missing_diagonal
    batch_size=256
    floating_degrees=np.issubdtype(d.dtype,np.floating)
    if floating_degrees:
        def recompute(columns,indexes):
            for start in range(0,len(columns),batch_size):
                selected=columns[start:start+batch_size]
                block=adj[np.ix_(indexes,selected)]
                missing=missing_diagonal[selected]
                if np.any(missing):
                    block[np.searchsorted(indexes,selected[missing]),
                          np.flatnonzero(missing)]=1
                if weights is not None:
                    d[selected]=np.sum(block*weights[indexes,np.newaxis],axis=0)
                else:
                    d[selected]=np.sum(block,axis=0)
        all_indexes=np.arange(len(adj))
        recompute(all_indexes,all_indexes)
    while np.any(active):
        indexes=np.flatnonzero(active)
        n=indexes[np.argmax(d[indexes])]
        if d[n]<min_size:
            break
        ww.append(d[n])
        ii=indexes[(adj[n,indexes]>0)|(indexes==n)]
        clusters.append(ii)
        if max_clusters is not None:
            if len(clusters) >= max_clusters:
                break
        active[ii]=False
        if floating_degrees:
            touched=np.zeros(len(adj),dtype=bool)
            for i in ii:
                touched|=adj[i]!=0
            remaining=np.flatnonzero(active)
            recompute(remaining[touched[remaining]],remaining)
        else:
            # Bound the temporary created by indexing rows of the dense matrix.
            if weights is not None:
                for start in range(0,len(ii),batch_size):
                    batch=ii[start:start+batch_size]
                    d-=np.einsum('ij,i->j',adj[batch],weights[batch],dtype=degree_dtype)
            else:
                for start in range(0,len(ii),batch_size):
                    batch=ii[start:start+batch_size]
                    d-=np.sum(adj[batch],axis=0)
    return ClusteringResult(method="daura",clusters=clusters, weights=ww)

@numba_jit
def _qt_outer(distances,indexes,next_,cutoff):
    candidates=np.empty(len(indexes),dtype='int')
    n=0
    for i in indexes:
        if i!=next_ and distances[next_,i]<=cutoff:
            candidates[n]=i
            n+=1
    return candidates[:n]

@numba_jit
def _qt_inner(distances,dist_from_cluster,candidates,count,cutoff,weights):
    if count<1:
        return (-1,np.inf,count)
    next_i=0
    minval=dist_from_cluster[0]
    weight_minval=weights[candidates[0]]
    for i in range(1,count):
        val=dist_from_cluster[i]
        if val<minval or (val==minval and weights[candidates[i]]>weight_minval):
            next_i=i
            minval=val
            weight_minval=weights[candidates[i]]
    if minval>cutoff:
        return (-1,minval,count)
    next_=candidates[next_i]
    remaining=0
    for i in range(count):
        if i==next_i:
            continue
        val=distances[next_,candidates[i]]
        if dist_from_cluster[i]<val:
            dist_from_cluster[i]=val
        if dist_from_cluster[i]<=cutoff:
            candidates[remaining]=candidates[i]
            dist_from_cluster[remaining]=dist_from_cluster[i]
            remaining+=1
    return (next_,minval,remaining)

def qt(distances,cutoff,weights=None,*,min_size=0,max_clusters=None):
    """Quality threshold clustering.

       The method is explained in the [original paper](https://doi.org/10.1101/gr.9.11.1106).
       The implementation has been adapted from [this one](https://github.com/rglez/QT),
       which is also released under a GPL licence.
       Thus, if you use this algorithm please cite [this article](https://doi.org/10.1021/acs.jcim.9b00558),
       which also discusses the important differences between this algorithm and the Daura et al algorithm
       in the context of analysing molecular dynamics simulations.
       Additionally, mention which exact version of the bussilab package you used.

       This implementation supports weights and arbitrary distance metrics.
       Singleton clusters are included unless excluded by `min_size` or
       `max_clusters`. When candidate clusters have the same total weight,
       the one with smaller diameter is selected. When growing a candidate
       cluster, equally distant frames are considered by descending weight.
       A float32 distance matrix uses less memory than float64, although
       rounding can change results near the cutoff or a tie.

       Parameters
       ----------

       distances : array_like, square matrix

           distances[i,j] contains the distance between frames i and j.
           The matrix should be symmetric; its diagonal is ignored.

       cutoff : number

           Maximum pairwise distance within a cluster (inclusive).

       weights : array_like, optional

           weights[i] contains the finite, nonnegative weight of the i-th frame.

       min_size : number

           Minimum cluster size. Clusters smaller than this size are not reported.
           When using weights, the cluster size is defined as the sum of the weights of
           the members of the cluster.

       max_clusters : int

           Maximum number of clusters; zero returns an empty result.

       Example
       -------

       ```
       from bussilab import clustering
       import numpy as np
       import scipy.spatial.distance as distance
       dist=distance.squareform(distance.pdist(trajectory))
       clustering.qt(dist,0.7)

       clustering.qt(np.array(dist,dtype='float32'),0.7)
       ```
    """
    if max_clusters is not None:
        if max_clusters<0:
            raise ValueError("max_clusters must be nonnegative")
        if max_clusters==0:
            return ClusteringResult(method="qt",clusters=[],weights=[])
    clusters=[]
    ww=[]
    distances=np.asarray(distances)
    if distances.ndim!=2 or distances.shape[0]!=distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    if np.isnan(cutoff):
        raise ValueError("cutoff must not be NaN")
    N=len(distances)
    if weights is None:
        weights=np.ones(N,dtype="int")
    else:
        weights=_validate_weights(weights,N)
    indexes=np.arange(N)
    degree_dtype=np.sum(np.empty(0,dtype=weights.dtype)).dtype
    degrees=np.zeros(N,dtype=degree_dtype)
    batch_size=256
    floating_weights=np.issubdtype(weights.dtype,np.floating)
    if floating_weights:
        def recompute(columns,indexes):
            for start in range(0,len(columns),batch_size):
                selected=columns[start:start+batch_size]
                adjacent=distances[np.ix_(indexes,selected)]<=cutoff
                adjacent[np.searchsorted(indexes,selected),np.arange(len(selected))]=0<=cutoff
                degrees[selected]=np.sum(adjacent*weights[indexes,np.newaxis],axis=0)
        recompute(indexes,indexes)
    else:
        # Integer degrees can be updated exactly as points are removed.
        for start in range(0,N,batch_size):
            stop=min(start+batch_size,N)
            adjacent=distances[start:stop]<=cutoff
            np.fill_diagonal(adjacent[:,start:stop],0<=cutoff)
            degrees+=np.einsum('ij,i->j',adjacent,weights[start:stop],dtype=degree_dtype)

    ncluster=0
    while len(indexes)>0:
        sorted_indexes=np.argsort(-degrees[indexes])

        cluster_size=-np.inf
        diameter=0.0
        cluster=[]
        i_degrees=0
        for i_degrees in range(len(sorted_indexes)):

            if degrees[indexes[sorted_indexes[i_degrees]]] < cluster_size: # optimization
                break

            next_=indexes[sorted_indexes[i_degrees]]
            precluster = [next_]
            new_cluster_diameter=0.0
            candidates=_qt_outer(distances,indexes,next_,cutoff)
            dist_from_cluster=distances[next_][candidates]
            candidate_count=len(candidates)
            while True:
                (next_,minval,candidate_count)=_qt_inner(
                    distances,dist_from_cluster,candidates,candidate_count,cutoff,weights)
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

        clusters.append(np.array(cluster))
        ww.append(cluster_size)

        if max_clusters is not None:
            if ncluster >= max_clusters:
                break

        removed=np.array(cluster)
        remaining=indexes[~np.isin(indexes,removed)]
        if floating_weights:
            # Recompute only degrees touched by removed points. Subtraction
            # could lose small weights when contributions differ greatly.
            touched=np.zeros(N,dtype=bool)
            for i in removed:
                touched|=distances[i]<=cutoff
            recompute(remaining[touched[remaining]],remaining)
        else:
            # Removed rows no longer contribute to integer degrees.
            for start in range(0,len(removed),batch_size):
                batch=removed[start:start+batch_size]
                adjacent=distances[batch]<=cutoff
                degrees-=np.einsum('ij,i->j',adjacent,weights[batch],dtype=degree_dtype)
        indexes=remaining
    return ClusteringResult(method="qt",clusters=clusters, weights=ww)
