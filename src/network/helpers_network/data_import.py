# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:24:54 2018

@author: silus
"""
import pandas as pd
import numpy as np
from helpers_network.gsp_helpers import graph_sampling, gbf
from scipy import sparse

def train_data_in_network():
    """Find features in train data and return those that can also be found in
    the network along with the corresponding node index"""
    gene_idx = pd.read_csv('protein_genes_index.csv')
    gene_idx['gene_name'] = gene_idx['gene_name'].apply(lambda x: x.upper())
    
    # Nodes in largest component
    node_idx_lc = np.load('nodes_largest_component.npy')
    node2nodelc = pd.DataFrame(data={'node_lc_idx': np.arange(len(node_idx_lc)), 
                               'node_idx' : node_idx_lc})
    # Lookup table node idx to node idx in largest component (adjacency indices)
    node2nodelc = node2nodelc.set_index('node_idx')
    gene_idx = gene_idx.join(node2nodelc, on='node_idx')
    
    # Get array of all the features in train data
    data = pd.read_csv('../../data/ind_train_data.csv', nrows=1)
    train_data_features = data.T
    train_data_features.reset_index(inplace=True)
    train_data_features = train_data_features.rename(columns={'index':'gene', 0:'feature_idx'})
    train_data_features['feature_idx'] = np.arange(len(train_data_features['feature_idx']))
    train_data_features = train_data_features.set_index('gene')
    
    node2feature = gene_idx.join(train_data_features, on='gene_name')
    # Drop nodes that are not in the feature index or in the largest component
    node2feature = node2feature.dropna(subset=['feature_idx', 'node_lc_idx'])
    print("Out of {0} genes in the train data, {1} could be found in the largest comp. of the gene network"
          .format(len(train_data_features), len(node2feature)))
    node_feature_idx = node2feature[['node_lc_idx','feature_idx']].values.astype(int)
    # First column: Node index of the gene
    # Second column: The feature index of the gene
    return node_feature_idx

def transform_data_with_network(dataset, n_nodes_lc, U, e, K):

    node_features = train_data_in_network();
    node_features = np.unique(node_features, axis=0)
    
    # read_csv preserves ordering of features as in the csv (increasing feature idx).
    # Need to translate this into node order
    node_order = np.argsort(node_features[:,1])
    nodes_ordered = node_features[node_order,0]
    train_data = pd.read_csv('../../data/'+ dataset + '.csv',usecols=node_features[:,1])
    
    # Map train data onto a graph signal (One value per node)
    # If a node (gene) of the largest. comp was not present in the train data,
    # node (gene) gets assigned zero 
    graph_signals = np.zeros((train_data.shape[0], n_nodes_lc))
    graph_signals[:,nodes_ordered] = train_data.values
    
    # 1st method: Graph sampling
    nodes_to_keep_GS_LF = graph_sampling(U, e, K, interval=(0,int(n_nodes_lc/2)))
    nodes_to_keep_GS_HF = graph_sampling(U, e, K, interval=(int(n_nodes_lc/2), n_nodes_lc))
    # These are the genes, whose signal is the most meaningful based on GS
    x_GS_LF = graph_signals[:, nodes_to_keep_GS_LF]
    x_GS_HF = graph_signals[:, nodes_to_keep_GS_HF]
    
    # 2nd method: Graph-based filtering (GBF)
    x_GBF_LF = (gbf(U, graph_signals.T, K, 'low-pass')).T
    x_GBF_HF = (gbf(U, graph_signals.T, K, 'high-pass')).T
    
    return x_GS_LF, x_GS_HF, x_GBF_LF, x_GBF_HF