# Fast implementation of the Louvain algorithm.
# Author: Alexandre Hollocou <alexandre@hollocou.fr>
# License: 3-clause BSD

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map

cimport numpy as np
import numpy as np

cimport cython

float_type = np.float
ctypedef np.float_t float_type_t

int_type = np.float
ctypedef np.int_t int_type_t


cdef class CythonLouvain(object):

    cdef int_type_t __PASS_MAX
    cdef float_type_t __MIN

    # Current graph
    cdef int_type_t n_nodes
    cdef vector[vector[int_type_t]] graph_neighbors
    cdef vector[vector[float_type_t]] graph_weights

    # Status
    cdef vector[vector[int_type_t]] communities
    cdef int_type_t *node2com
    cdef float_type_t *internals
    cdef float_type_t *loops
    cdef float_type_t *degrees
    cdef float_type_t *gdegrees
    cdef float_type_t total_weight

    # List of partitions
    cdef vector[vector[int_type_t]] partition_list

    @cython.boundscheck(False)
    def __init__(self, adj_matrix):

        cdef size_t i, j

        # Initialize graph
        self.n_nodes = adj_matrix.shape[0]
        self.graph_neighbors.resize(self.n_nodes)
        self.graph_weights.resize(self.n_nodes)

        # Copy graph
        cdef np.int32_t[:] indptr = adj_matrix.indptr
        cdef np.int32_t[:] indices = adj_matrix.indices
        cdef float_type_t[:] data = np.array(adj_matrix.data, dtype=float_type)
        for i in range(indptr.shape[0] - 1):
            for j in range(indptr[i], indptr[i + 1]):
                self.graph_neighbors[i].push_back(indices[j])
                self.graph_weights[i].push_back(data[j])

        # Stopping criteria
        self.__PASS_MAX = -1
        self.__MIN = 0.0000001

    cdef void init_status(self):

        # Initialize internal state variables
        free(self.node2com)
        free(self.internals)
        free(self.loops)
        free(self.degrees)
        free(self.gdegrees)
        self.node2com = <int_type_t*> malloc(self.n_nodes * sizeof(int_type_t))
        self.internals = <float_type_t*> malloc(self.n_nodes * sizeof(float_type_t))
        self.loops = <float_type_t*> malloc(self.n_nodes * sizeof(float_type_t))
        self.degrees = <float_type_t*> malloc(self.n_nodes * sizeof(float_type_t))
        self.gdegrees = <float_type_t*> malloc(self.n_nodes * sizeof(float_type_t))
        self.total_weight = 0.

        # Fill internal state variables
        cdef size_t node, i
        cdef int_type_t neighbor
        cdef float_type_t neighbor_weight
        for node in range(self.n_nodes):
            self.node2com[node] = node
            self.internals[node] = 0.
            self.loops[node] = 0.
            self.degrees[node] = 0.
            self.gdegrees[node] = 0.
            for i in range(self.graph_neighbors[node].size()):
                neighbor = self.graph_neighbors[node][i]
                neighbor_weight = self.graph_weights[node][i]
                if neighbor == node:
                    self.internals[node] += neighbor_weight
                    self.loops[node] += neighbor_weight
                    self.degrees[node] += 2. * neighbor_weight
                    self.gdegrees[node] += 2. * neighbor_weight
                    self.total_weight += 2. * neighbor_weight
                else:
                    self.degrees[node] += neighbor_weight
                    self.gdegrees[node] += neighbor_weight
                    self.total_weight += neighbor_weight
        self.total_weight = self.total_weight / 2.
        return

    @cython.boundscheck(False)
    cdef void remove(self, int_type_t node, int_type_t com, float_type_t weight):
        self.node2com[node] = -1
        self.degrees[com] = self.degrees[com] - self.gdegrees[node]
        self.internals[com] = self.internals[com] - weight - self.loops[node]

    @cython.boundscheck(False)
    cdef void insert(self, int_type_t node, int_type_t com, float_type_t weight):
        self.node2com[node] = com
        self.degrees[com] = self.degrees[com] + self.gdegrees[node]
        self.internals[com] = self.internals[com] + weight + self.loops[node]

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef float_type_t modularity(self, float_type_t resolution):
        cdef float_type_t result = 0.
        for com in range(self.n_nodes):
            result += resolution * self.internals[com] / self.total_weight
            result -= ((self.degrees[com] / (2. * self.total_weight)) ** 2)
        return result

    cdef map[int_type_t, float_type_t] neighcom(self, int_type_t node):
        cdef map[int_type_t, float_type_t] neighbor_weight
        for i in range(self.graph_neighbors[node].size()):
            if self.graph_neighbors[node][i] != node:
                neighborcom = self.node2com[self.graph_neighbors[node][i]]
                neighbor_weight[neighborcom] = neighbor_weight[neighborcom] + self.graph_weights[node][i]
        return neighbor_weight

    @cython.cdivision(True)
    cdef void one_level(self, float_type_t resolution):
        modified = True
        cdef int_type_t nb_pass_done = 0
        cdef float_type_t cur_mod = self.modularity(resolution)
        cdef float_type_t new_mod = cur_mod

        cdef int_type_t com
        cdef float_type_t weight
        cdef int_type_t node_com
        cdef int_type_t best_com

        cdef float_type_t increase
        cdef float_type_t best_increase

        cdef map[int_type_t, float_type_t] neighbor_weight

        while modified and nb_pass_done != self.__PASS_MAX:
            cur_mod = new_mod
            modified = False
            nb_pass_done += 1

            for node in range(self.n_nodes):
                node_com = self.node2com[node]
                neighbor_weight = self.neighcom(node)
                self.remove(node, node_com, neighbor_weight[node_com])
                best_com = node_com
                best_increase = 0.
                for com_weight in neighbor_weight:
                    com = com_weight.first
                    weight = com_weight.second
                    if weight > 0:
                        increase = resolution * weight - \
                                   self.degrees[com] * self.gdegrees[node] / (self.total_weight * 2.)
                        if increase > best_increase:
                            best_increase = increase
                            best_com = com
                self.insert(node, best_com, neighbor_weight[best_com])
                if best_com != node_com:
                    modified = True
            new_mod = self.modularity(resolution)
            if new_mod - cur_mod < self.__MIN:
                break

    cdef void renumber(self):
        cdef vector[int_type_t] com_n_nodes
        com_n_nodes.resize(self.n_nodes)
        for node in range(self.n_nodes):
            com_n_nodes[self.node2com[node]] += 1

        cdef vector[int_type_t] com_new_index
        com_new_index.resize(self.n_nodes)
        cdef int_type_t final_index = 0
        for com in range(self.n_nodes):
            if com_n_nodes[com] > 0:
                com_new_index[com] = final_index
                final_index += 1

        cdef vector[vector[int_type_t]] new_communities
        new_communities.resize(final_index)
        cdef int_type_t *new_node2com = <int_type_t*> malloc(self.n_nodes * sizeof(int_type_t))

        for node in range(self.n_nodes):
            new_communities[com_new_index[self.node2com[node]]].push_back(node)
            new_node2com[node] = com_new_index[self.node2com[node]]

        self.communities = new_communities
        self.node2com = new_node2com

    cdef void induced_graph(self):

        cdef int_type_t new_n_nodes = self.communities.size()
        cdef vector[vector[int_type_t]] new_graph_neighbors
        new_graph_neighbors.resize(new_n_nodes)
        cdef vector[vector[float_type_t]] new_graph_weights
        new_graph_weights.resize(new_n_nodes)

        cdef int_type_t neighbor
        cdef int_type_t neighbor_com
        cdef float_type_t neighbor_weight

        cdef map[int_type_t, float_type_t] to_insert

        for com in range(new_n_nodes):
            to_insert.clear()
            for node in self.communities[com]:
                for i in range(self.graph_neighbors[node].size()):
                    neighbor = self.graph_neighbors[node][i]
                    neighbor_com = self.node2com[neighbor]
                    neighbor_weight = self.graph_weights[node][i]
                    if neighbor == node:
                        to_insert[neighbor_com] += 2 * neighbor_weight
                    else:
                        to_insert[neighbor_com] += neighbor_weight
            for com_weight in to_insert:
                new_graph_neighbors[com].push_back(com_weight.first)
                if com_weight.first == com:
                    new_graph_weights[com].push_back(com_weight.second / 2.)
                else:
                    new_graph_weights[com].push_back(com_weight.second)

        self.n_nodes = new_n_nodes
        self.graph_neighbors = new_graph_neighbors
        self.graph_weights = new_graph_weights

    cdef vector[int_type_t] get_partition(self):
        cdef vector[int_type_t] partition
        for i in range(self.n_nodes):
            partition.push_back(self.node2com[i])
        return partition

    def generate_dendrogram(self, float_type_t resolution):
        cdef float_type_t mod
        cdef float_type_t new_mod

        self.init_status()
        self.one_level(resolution)
        new_mod = self.modularity(resolution)
        self.renumber()
        self.partition_list.push_back(self.get_partition())
        mod = new_mod
        self.induced_graph()
        self.init_status()

        while True:
            self.one_level(resolution)
            new_mod = self.modularity(resolution)
            if new_mod - mod < self.__MIN:
                break
            self.renumber()
            self.partition_list.push_back(self.get_partition())
            mod = new_mod
            self.induced_graph()
            self.init_status()

        return self.partition_list


@cython.boundscheck(False)
@cython.cdivision(True)
def modularity(partition, adj_matrix, resolution):
    cdef size_t node, i, neighbor, com
    cdef float_type_t edge_weight

    # Copy graph
    cdef np.int32_t n_nodes = adj_matrix.shape[0]
    cdef np.int32_t[:] indptr = adj_matrix.indptr
    cdef np.int32_t[:] indices = adj_matrix.indices
    cdef float_type_t[:] data = np.array(adj_matrix.data, dtype=float_type)

    # Copy partition
    cdef vector[np.int32_t] part
    for node in range(n_nodes):
        part.push_back(partition[node])

    # Compute degrees
    cdef float_type_t links = 0.
    cdef vector[float_type_t] degrees
    cdef float_type_t degree
    for node in range(n_nodes):
        degree = 0.
        for i in range(indptr[node], indptr[node + 1]):
            if indices[i] == node:
                degree += 2 * data[i]
                links += 2 * data[i]
            else:
                degree += data[i]
                links += data[i]
        degrees.push_back(degree)
    links /= 2

    cdef map[int_type_t, float_type_t] inc
    cdef map[int_type_t, float_type_t] deg

    for node in range(n_nodes):
        com = part[node]
        deg[com] = deg[com] + degrees[node]
        for i in range(indptr[node], indptr[node + 1]):
            neighbor = indices[i]
            edge_weight = data[i]
            if part[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc[com] + float(edge_weight)
                else:
                    inc[com] = inc[com] + float(edge_weight) / 2.

    cdef float_type_t res = 0.
    for it in deg:
        com = it.first
        res += resolution * (inc[com] / links) - \
               (deg[com] / (2. * links)) ** 2
    return res
