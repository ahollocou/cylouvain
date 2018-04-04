# -*- coding: utf-8 -*-
"""Cython implementation of the Louvain algorithm"""

# Author: Alexandre Hollocou <alexandre@hollocou.fr>
# License: BSD 3 clause

from ._louvain import CythonLouvain
from ._louvain import modularity as cython_modularity

from scipy import sparse
import numpy as np
import networkx as nx


def best_partition(graph, resolution=1.):
    """
    Given a graph, compute a partition of the nodes
    using the Louvain heuristic to maximize the modularity function.

    Parameters
    ----------
    graph: networkx.Graph, scipy.csr_matrix or np.ndarray
        The input graph or its adjacency matrix (sparse or dense).

    resolution: double, optional, default: 1.0
        The resolution parameter that controls the size of the communities.
        This parameter corresponds to the time introduced in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    partition: dictionary
        The output partition, with communities numbered from 0 to the number of communities.
        The keys of the dictionary correspond to the nodes and the values to the communities.

    References
    ----------

    - Fast unfolding of communities in large networks, 2008
      Blondel, Vincent D and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne
      Journal of statistical mechanics: theory and experiment, 2008(10), P10008.

    Notes
    -----
    Uses a Cython version of the Louvain algorithm.

    """
    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
    else:
        raise TypeError(
            "The argument should be a NetworkX graph, a NumPy array or a SciPy Compressed Sparse Row matrix.")
    louvain = CythonLouvain(adj_matrix)

    dendrogram = louvain.generate_dendrogram(resolution)

    partition = range(len(dendrogram[-1]))
    for i in range(1, len(dendrogram) + 1):
        new_partition = dict()
        for j in range(len(dendrogram[-i])):
            new_partition[j] = partition[dendrogram[-i][j]]
        partition = new_partition

    if type(graph) == nx.classes.graph.Graph:
        return {node: partition[i] for (i, node) in enumerate(graph.nodes())}
    else:
        return partition


def modularity(partition, graph, resolution=1.):
    """
    Compute the modularity of a node partition of a graph.

    Parameters
    ----------
    partition: dict
       The partition of the nodes.
       The keys of the dictionary correspond to the nodes and the values to the communities.

    graph: networkx.Graph, scipy.csr_matrix or np.ndarray
        The graph or its adjacency matrix (sparse or dense).

    resolution: double, optional, default: 1.0
        The resolution parameter in the modularity function.
        This parameter corresponds to the time introduced in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    modularity : float
       The modularity.

    References
    ----------

    - Finding and evaluating community structure in networks, 2004
      Newman, M. E. and Girvan, M.
      Physical review E, 69(2), 026113.

    Notes
    -----
    Uses optimized Cython code to compute modularity.

    """
    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
        partition = {i: partition[node] for (i, node) in enumerate(graph.nodes())}
    else:
        raise TypeError(
            "The argument should be a NetworkX graph, a NumPy array or a SciPy Compressed Sparse Row matrix.")
    return cython_modularity(partition, adj_matrix, resolution)
