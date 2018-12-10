# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:24:54 2018

@author: silus
"""
import pandas as pd
import numpy as np

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
    data = pd.read_csv('../../data/train_data.csv', nrows=1)
    train_data_features = data.T
    train_data_features.reset_index(inplace=True)
    train_data_features = train_data_features.rename(columns={'index':'gene', 0:'feature_idx'})
    train_data_features['feature_idx'] = np.arange(len(train_data_features['feature_idx']))
    train_data_features = train_data_features.set_index('gene')
    
    node2feature = gene_idx.join(train_data_features, on='gene_name')
    node2feature = node2feature.dropna(subset=['feature_idx', 'node_lc_idx'])
    print("Out of {0} genes in the train data, {1} could be found in the largest comp. of the gene network"
          .format(len(train_data_features), len(node2feature)))
    node_feature_idx = node2feature[['node_lc_idx','feature_idx']].values.astype(int)
    # First column: Node index of the gene
    # Second column: The feature index of the gene
    return node_feature_idx