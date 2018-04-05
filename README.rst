cylouvain: Cython Louvain
=========================

cylouvain is a Python module that provides a fast implementation
of the classic Louvain algorithm for node clustering in graph.

This module uses Cython in order to obtain C-like performance with
code mostly writen in Python.

Installation
------------

Install the latest version of cylouvain using ``pip`` ::

    $ pip install cylouvain

Dependencies
------------

cylouvain requires:

- Python (>= 2.7 or >= 3.4)
- NumPy
- SciPy
- NetworkX

Simple example
--------------

Build a simple graph with NetworkX::

    >>> import networkx as nx
    >>> graph = nx.Graph()
    >>> graph.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
    >>> graph.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'c'),
                              ('c', 'd'), ('c', 'e'), ('d', 'e')])

Compute a partition of the nodes using cylouvain::

    >>> import cylouvain
    >>> partition = cylouvain.best_partition(graph)
    >>> print(partition)
    {'a': 0, 'b': 0, 'c': 0, 'd': 1, 'e': 1}

Compute the corresponding modularity::

    >>> modularity = cylouvain.modularity(partition, graph)
    >>> print("Modularity: %0.3f\n" % modularity)
    Modularity: 0.111

References
----------

The Louvain algorithm is an heuristic to find a node partition that maximizes the modularity function.
It is described in::

    Fast unfolding of communities in large networks
    Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre
    Journal of Statistical Mechanics: Theory and Experiment 2008 (10), P10008 (12pp)

The modularity function was first introduced in::

    Finding and evaluating community structure in networks
    Newman, Mark EJ and Girvan, Michelle
    Physical review E, 2004, vol. 69, no 2, p. 026113.

License
-------

Released under the 3-Clause BSD license (see `COPYING`)::

   Copyright (C) 2018 Alexandre Hollocou <alexandre@hollocou.fr>
