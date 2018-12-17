"""
Create eigen-decomposition of network, saves it.
"""

from network.gene_network import GeneNetworkPCA

GeneNetworkPCA('../network/adjacency_sparse.npz', '../network/node_index.csv').compute_eigdec()
