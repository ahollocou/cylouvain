import networkx as nx
import cylouvain

graph = nx.Graph()
graph.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
graph.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd'), ('c', 'e'), ('d', 'e')])

partition = cylouvain.best_partition(graph)

print(partition)
