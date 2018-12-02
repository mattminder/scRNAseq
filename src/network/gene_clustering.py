# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:29:52 2018

@author: silus
"""

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.cluster import spectral_clustering
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Load adjacency matrix and plot for good measure
adjacency_sparse = sparse.load_npz("adjacency_sparse.npz")

gene_graph = nx.from_scipy_sparse_matrix(adjacency_sparse)
n_nodes = gene_graph.order()
# Eliminate disconnected nodes
isolated_nodes = nx.isolates(gene_graph)
print("{0:.3f} % of nodes are disconnected"
      .format(len(list(isolated_nodes))/n_nodes*100))
gene_graph.remove_nodes_from(isolated_nodes)

# Keep largest component only
print("There are {0} connected components".format(nx.number_connected_components(gene_graph)))
gene_graph_lc = max(nx.connected_component_subgraphs(gene_graph), key=len)
print("The largest component contains {0:.3f} % of all genes".format(gene_graph_lc.order()/n_nodes*100))

# Get adjacency to do further processing
adjacency_lc = nx.to_scipy_sparse_matrix(gene_graph_lc)

# Widen kernel width to get a better similarity measure
# Equivalent to having a larger standart deviation of distances 
adjacency_lc = adjacency_lc.power(1/4)

# Sklearn magic to do spectral embedding (Laplacian eigenmap)
# ÄNDERE FÜR ANGERI CLUSTER AHZAU
k = 40
print("Spectral clustering: Trying to find {0} clusters".format(k))
x_labels = spectral_clustering(adjacency_lc, n_clusters=k, assign_labels='kmeans')

labels, counts = np.unique(x_labels, return_counts=True)
plt.bar(labels, counts)
plt.xlabel('Cluster')
plt.ylabel('Number of genes')

protein_gene_index = pd.read_csv('protein_genes_index.csv')
cluster_assignments = pd.DataFrame(data=x_labels, index=np.array(gene_graph_lc.nodes()), columns=['cluster'])
protein_gene_cluster = protein_gene_index.join(cluster_assignments, on='node_idx')
protein_gene_cluster = protein_gene_cluster.dropna(subset=['cluster'])
protein_gene_cluster[['cluster']] = protein_gene_cluster[['cluster']].astype(int)
# Save cluster assignments to csv
protein_gene_cluster.to_csv('protein_gene_cluster_assign.csv', encoding='utf-8')