# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:29:52 2018

@author: silus
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import spectral_embedding
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Load adjacency matrix and plot for good measure
adjacency = np.load("adjacency.npy")
plt.spy(adjacency)

# Eliminate disconnected nodes
k_i = np.sum(adjacency, axis=1)
n_nodes = adjacency.shape[0]
disc_nodes = k_i == 0
n_disc_nodes = np.sum(disc_nodes)
print("{0:.3f} % of nodes are disconnected".format(n_disc_nodes/n_nodes*100))

# Reduced adjacency (without disconnected noted)
adjacency_red = adjacency[~disc_nodes, :]
adjacency_red = adjacency_red[:,~disc_nodes] # Python expert level 1000

# Use networkx to identify largest component 
gene_graph = nx.from_numpy_array(adjacency_red)
print("There are {0} connected components".format(nx.number_connected_components(gene_graph)))
# Adjacency of largest component only
largest_cc = np.array(list(max(nx.connected_components(gene_graph), key=len)))
adjacency_lc = adjacency_red[largest_cc,:]
adjacency_lc = adjacency_lc[:,largest_cc]
print("The largest component contains {0:.3f} % of all genes".format(adjacency_lc.shape[0]/n_nodes*100))

# Sklearn magic to do spectral embedding (Laplacian eigenmap)
dims = 3
x_embed = spectral_embedding(adjacency_lc, n_components=dims, norm_laplacian=True, drop_first=True)

# Plot embedded data
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_embed[:,0], x_embed[:,1], x_embed[:,2], color='b')
ax.set_title('Normalized Laplacian eigenmap (d = 3)')