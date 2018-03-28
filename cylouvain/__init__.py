from ._louvain import CythonLouvain
from ._louvain import modularity as cython_modularity

from scipy import sparse
import numpy as np
import networkx as nx


def best_partition(graph, resolution=1.):
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
    if type(graph) == sparse.csr_matrix:
        adj_matrix = graph
    elif type(graph) == np.ndarray:
        adj_matrix = sparse.csr_matrix(graph)
    elif type(graph) == nx.classes.graph.Graph:
        adj_matrix = nx.adj_matrix(graph)
    else:
        raise TypeError(
            "The argument should be a NetworkX graph, a NumPy array or a SciPy Compressed Sparse Row matrix.")
    return cython_modularity(partition, adj_matrix, resolution)
