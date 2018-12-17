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
protein_data[['protein1','protein2']] = protein_data[['protein1', 'protein2']].apply(lambda x: x.str.slice(13,24))
# Convert ID to int
protein_data[['protein1','protein2']] = protein_data[['protein1', 'protein2']].astype(int)

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

# Edgelist: Replace protein IDs with gene names
edges = protein_data.join(prot2gene, on='protein1')
edges = edges.join(prot2gene, on='protein2', rsuffix='_2')

# Delete edges between unknown proteins
edges = edges.dropna(subset=['gene_name','gene_name_2'])
print(edges.shape)
# Delete self-loops
edges = edges[edges.gene_name != edges.gene_name_2]
print(edges.shape)
edges = edges.drop(columns=['protein1','protein2'])

np.testing.assert_allclose(len(edges[['gene_name']].drop_duplicates(keep='first')),
                           len(edges[['gene_name_2']].drop_duplicates(keep='first')))

nodes_idx = edges[['gene_name']].drop_duplicates(keep='first')
nodes_idx.insert(0,'node_idx', np.arange(len(nodes_idx)))
nodes_idx = nodes_idx.set_index('gene_name')
nodes_idx.to_csv('node_index.csv',encoding='utf-8')

edges = edges.join(nodes_idx, on='gene_name')
edges = edges.join(nodes_idx, on='gene_name_2',rsuffix='_2')
edges = edges.drop(columns=['gene_name', 'gene_name_2'])

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