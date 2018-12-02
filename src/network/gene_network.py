# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:16:12 2018

@author: silus
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import matplotlib.pyplot as plt

# Import protein interaction network
protein_data = pd.read_csv('../../data/protein_links.txt', sep=' ')
# Format protein names (keep only ID)
protein_data[['protein1','protein2']] = protein_data[['protein1','protein2']].apply(lambda x: x.str.slice(13,24))
# Convert ID to int
protein_data[['protein1','protein2']] = protein_data[['protein1','protein2']].astype(int)

# Import gene names, ids and transcribed protein ids
gene_data = pd.read_csv('../../data/gene_name_transcript_protein_id.txt', sep=',')

# Lookup table of genes that express a given protein
prot2gene = gene_data.drop(columns=['Gene stable ID','Transcript stable ID'])
prot2gene = prot2gene.rename(columns={'Gene name':'gene_name','Protein stable ID':'protein_id'})
prot2gene[['protein_id']] = prot2gene[['protein_id']].apply(lambda x: x.str.slice(7,18))
# Drop genes that don't code any proteins
prot2gene = prot2gene.dropna(subset=['protein_id'])
prot2gene[['protein_id']] = prot2gene[['protein_id']].astype(int)
prot2gene = prot2gene.set_index('protein_id')
print("Ensembl gene database: {0} genes coding for {0} proteins found".format(prot2gene.shape[0]))

# Proteins in protein interaction network
prot_in_network = protein_data[['protein1']].drop_duplicates(keep='first').rename(columns={'protein1':'protein_id'})
# Filter genes: Keep only those that code for a protein in the network
genes_in_network = prot_in_network.set_index('protein_id').join(prot2gene)
print("{0} proteins in the network couldn't be associated to a gene".format(genes_in_network['gene_name'].isnull().sum()))
genes_in_network = genes_in_network.dropna(subset=['gene_name'])
print("{0} genes coding for {1} proteins found in protein interaction network"
      .format(genes_in_network[['gene_name']].drop_duplicates(keep='first').shape[0],prot_in_network.shape[0]))

# Every gene gets a node index
nodes_idx = genes_in_network[['gene_name']].drop_duplicates(keep='first')
nodes_idx.insert(0,'node_idx', np.arange(len(nodes_idx)))
# Associate proteins to node index by using the gene coding for them
prot2node = genes_in_network.join(nodes_idx.set_index('gene_name'), on='gene_name')
prot2node.to_csv('protein_genes_index.csv', encoding='utf-8')
prot2node = prot2node.drop(columns=['gene_name'])

# Make edgelist: Replace protein IDs with associated node Ids
edges = protein_data.join(prot2node, on='protein1')
edges = edges.join(prot2node, on='protein2', rsuffix='_2')
edges = edges.drop(columns=['protein1', 'protein2'])
# Delete edges between proteins without any associated gene
edges = edges.dropna(subset=['node_idx','node_idx_2'])
edges = edges.astype(int)

# Need heat kernel for laplacian eigenmap. Convert the combined scores to heat kernel weight

# Convert scores into distance: The higher the combined score, the lower the distance
edges_weighted = edges.copy()
edges_weighted[['combined_score']]= edges[['combined_score']].apply(lambda x: 1/x)
# Heat kernel
sigma = edges_weighted['combined_score'].std()
edges_weighted[['combined_score']] = edges_weighted[['combined_score']].apply(lambda x: np.exp(-x**2/(2*sigma**2)))
edges_weighted = edges_weighted.rename(columns={'combined_score':'weight'})

# Adjacency matrix
n_nodes = len(nodes_idx)
adjacency = np.zeros((n_nodes, n_nodes), dtype=float)

for idx, row in edges_weighted.iterrows():
    i, j = int(row.node_idx), int(row.node_idx_2)
    adjacency[i, j] = row.weight
    adjacency[j, i] = row.weight

save_npz("adjacency_sparse.npz", csr_matrix(adjacency))