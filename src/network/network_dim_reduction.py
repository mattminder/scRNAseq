# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:44:52 2018

@author: silus
"""

import numpy as np
import networkx as nx
import pandas as pd
import pygsp as pg
from helpers.gsp_helpers import graph_sampling, graph_filtering
from helpers.data_import import train_data_in_network
from scipy import sparse
import matplotlib.pyplot as plt

# Two dimensionality reduction techniques based on GSP:
# 1. Graph sampling based on vertices where signal energy is the most concentrated
# 2. Graph frequency sampling: Low/high pass filtering

adjacency_largest_component = sparse.load_npz('adjacency_largest_component.npz')
n_nodes = adjacency_largest_component.shape[0]
node_features = train_data_in_network();
node_features = np.unique(node_features, axis=0)
# read_csv preserves ordering of features as in the csv (increasing feature idx).
# Need to translate this into node order
node_order = np.argsort(node_features[:,1])
nodes_ordered = node_features[node_order,0]
train_data = pd.read_csv('../../data/train_data.csv',usecols=node_features[:,1])

# Map train data onto a graph signal (One value per node)
graph_signals = np.zeros((train_data.shape[0], n_nodes))
graph_signals[:,nodes_ordered] = train_data.values
# Create PyGSP graph
#graph = pg.graphs.Graph(adjacency_largest_component)
# Compute Fourier basis
#graph.compute_fourier_basis(recompute=False)

# Create signal
#### TODO ######



