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

def max_clique(adj,weights=None):
    """Same algorithm as in Reisser, et al, NAR (2020)."""
    # weights: optional weights
    # if adj is a graph, it will be copied
    graph=networkx.Graph(adj)
    cliques=[]
    ww=[]
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
        cliques.append(maxi)
        ww.append(maxw)
        graph.remove_nodes_from(maxi)
    return ClusteringResult(method="max_clique",clusters=cliques, weights=ww)

def daura(adj,weights=None):
    """Same algorithm as in Daura et al, Angew. Chemie (1999)."""
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
        ww.append(d[n])
        ii=np.where(adj[n]>0)[0]
        clusters.append(indexes[ii])
        adj=np.delete(adj,ii,axis=0)
        adj=np.delete(adj,ii,axis=1)
        if weights is not None:
            weights=np.delete(weights,ii)
        indexes=np.delete(indexes,ii)
    return ClusteringResult(method="daura",clusters=clusters, weights=ww)
