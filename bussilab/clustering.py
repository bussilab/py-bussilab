"""
Module with some clustering tools
"""

from typing import Optional

import networkx
import numpy as np

from .coretools import Result

class ClusteringResult(Result):
    def __init__(self,
                *,
                method: str,
                clusters: list,
                weights: Optional[list]
                ):
        self.method = method
        self.clusters = clusters
        self.weights = weights

def max_clique(adj,weights=None):
    """Same algorithm as in Reisser, et al, NAR (2020)."""
    # weights: optional weights
    if isinstance(adj, networkx.Graph):
        graph=adj
    else:
        graph=networkx.Graph(adj)
    cliques=[]
    ww=[]
    while graph.number_of_nodes()>0:
        maxw=0.0
        for i in networkx.algorithms.clique.find_cliques(graph):
            if weights is not None:
                w=np.sum(weights[np.array(i)])
            else:
                w=len(i)
            if w > maxw:
                maxi=i
                maxw=w
        cliques.append(maxi)
        ww.append(maxw)
        graph.remove_nodes_from(maxi)
    return ClusteringResult(method="max_clique",clusters=cliques, weights=ww)
