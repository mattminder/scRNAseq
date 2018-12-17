# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:13:44 2018

@author: silus
"""

import numpy as np
import pandas as pd
from scipy import sparse

class GeneNetworkPCA:
    """ Data transformation based on a gene network constructed from a protein interaction
    network.
    W_path:                 Path to adjacency matrix saved in scipy sparse format
    fourier_basis path:     Path to eigenvalues and eigenvectors saved in numpy 
                            format. (Tuple, path to eigenvalues is first element)
    node_index:             Panda dataframe with gene names and corresponding node
                            index
    n_components:           Number of components of the transformed data 
    method:                 'gs' for graph sampling, 'gbf' for graph-based filtering
                            'filter' is a rectangle filter without dim. reduction
    attenuation:            Frequency region that is considered for the transformation
                            Options are 'low-pass','high-pass' or 'band-pass'
    cutoffs:                Custom cut-off indices for band-pass between 0 and n_nodes.
                            Is automatically set when choosing low or high pass attenuation
    lap_type:               Only of importance when no fourier_basis_path is given.
                            Specifies the type of laplacian that is used to compute
                            the Fourier basis"""
                            
    def __init__(self, W_path, node_index_path, n_components=2570, fourier_basis_path=None,  
                 method='gs', attenuation='low-pass', cutoffs=None, lap_type='combinatorial'):
        
        # Set adjacency matrix and pca method
        self.method = method
        self.W = sparse.load_npz(W_path)
        
        if len(self.W.shape) != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError('W has incorrect shape {}'.format(self.W.shape))
        
        # Load eigendecomposition if it exists
        if fourier_basis_path:
            self._U = np.load(fourier_basis_path[1])
            self._e = np.load(fourier_basis_path[0])
        
        # Setting Laplacian type in case of Fourier basis computation
        if lap_type not in ['combinatorial', 'normalized']:
            print("Unknown Laplacian type. Setting to combinatorial")
            self.lap_type = 'combinatorial'
        else:
            self.lap_type = lap_type
            
        node_index = pd.read_csv(node_index_path)
        
        if 'gene_name' and 'node_idx' not in node_index.columns:
            raise ValueError('Node index has incorrect format. Need to have a column "gene_name" and "node_idx"')
        node_index['gene_name'] = node_index['gene_name'].apply(lambda x: x.upper())
        self.nodes = node_index
        
        self.n_nodes = len(node_index)
        self.n_components = n_components
        self.method = method
        self.attenuation = attenuation
        
        if attenuation == 'low-pass':
            self.cutoffs = (0, int(self.n_nodes/2))
        elif attenuation == 'high-pass':
            self.cutoffs = (int(self.n_nodes/2), self.n_nodes)
        elif attenuation not in ['low-pass','high-pass'] and not cutoffs:
            raise ValueError
            ("Unknown filter type. Choose either 'low-pass' or 'high-pass'. Alternatively, 'band-pass' and specify cutoffs")
        else:
            self.cutoffs = cutoffs
    
    def _check_fourier_properties(self, name):
        if not hasattr(self, '_' + name):
            print("{0} not available, computing Fourier basis...".format(name))
            self.compute_fourier_basis()
        return getattr(self, '_' + name)
    
    def _compute_laplacian(self):
        if self.lap_type == 'combinatorial':
                D = sparse.diags(np.ravel(self.W.sum(1)), 0)
                L = (D - self.W).tocsc()

        elif self.lap_type == 'normalized':
            dw = np.asarray(self.W.sum(axis=1)).squeeze()
            d = np.power(dw, -0.5)
            D = sparse.diags(np.ravel(d), 0).tocsc()
            L = sparse.identity(self.n_nodes) - D * self.W * D
            
        return L
    
    def compute_fourier_basis(self, recompute=True):
        
        if hasattr(self, '_e') and hasattr(self, '_U') and not recompute:
            return
        
        L = self._compute_laplacian()
        self._e, self._U = np.linalg.eigh(L.toarray())
        # Columns are eigenvectors. Sorted in ascending eigenvalue order.

        # Smallest eigenvalue should be zero: correct numerical errors.
        # Eigensolver might sometimes return small negative values, which
        # filter's implementations may not anticipate. Better for plotting too.
        assert -1e-12 < self._e[0] < 1e-12
        self._e[0] = 0

        if self.lap_type == 'normalized':
            # Spectrum bounded by [0, 2].
            assert self._e[-1] <= 2
        
        filename_vec = 'eigvectors'+'_'+ self.lap_type + '.npy'
        filename_val = 'eigvalues'+'_'+ self.lap_type + '.npy'
        
        print('Saving Fourier basis in CWD as {0} and {1}'.format(filename_val, filename_vec))
        np.save(filename_val, self._e)
        np.save(filename_vec, self._U)
      
    def _subset_in_network(self, column_labels):
        """Identifies the features of the input data that are present in the network
        column_labels:  pandas DataFrame with the labels of each column (eg. gene name)
        
        Returns:
        node2ft:        pandas Datframe with one row containing the index of a column
                        of x and the index of the node that corresponds to this column"""
        fl = column_labels.copy()
        fl = fl.T
        fl.reset_index(inplace=True)
        fl = fl.rename(columns={'index':'gene', 0:'feature_idx'})
        fl.drop([0], inplace=True)
        fl['feature_idx'] = np.arange(len(fl['feature_idx']))
        fl = fl.set_index('gene')
        node2ft = self.nodes.join(fl, on='gene_name')
        node2ft = node2ft.dropna(subset=['feature_idx', 'node_idx'])
        print("Out of {0} genes in the data, {1} could be found in the gene network"
              .format(len(fl), len(node2ft)))
        node2ft = node2ft[['node_idx','feature_idx']].values.astype(int)
        return node2ft
    
    def _graph_sampling(self, x):
        """Graph sampling algorithm: Choose K eigenvalues with the highest
        Graph weighted coherence (signal energy) in the frequency band
        specified in interval.
        x:  Signal on graph (numpy array with length n_nodes)"""
    
        eigenvecs = self._check_fourier_properties('U')
        
        # Computed weighted coherence in given interval
        F_basis_sampled = eigenvecs[:,self.cutoffs[0]:self.cutoffs[1]]
        graph_weighted_coherence_nodes = np.sum(np.power(F_basis_sampled, 2), axis=1)
        
        # Choose K nodes with the highest weighted coherence
        keep_nodes = np.argsort(-graph_weighted_coherence_nodes)[0:self.n_components]
        
        return x[:,keep_nodes]
    
    def _gbf(self, x):
        """Graph-based filtering: All eigenvectors with a eigenvalue below the
        cut-off frequency f_c span a subspace. The original data is then projected
        onto this subspace. This is somewhat similar to low or highpass filtering.
        x:  Signal on graph (numpy array with length n_nodes)"""
        
        eigenvecs_all = self._check_fourier_properties('U')
        eigenvecs = eigenvecs_all[:,self.cutoffs[0]:self.cutoffs[1]]
        
        if (self.attenuation == 'low-pass'):
            subspace_base = eigenvecs[:,0:self.n_components]
        else:
            subspace_base = eigenvecs[:,-self.n_components:]
        # Project on subspace
        return x @ subspace_base
    
    def _simple_filter(self, x):
        """ Simple rectangle filter (keeps frequencies bounded by elements in
        cutoffs attribute)
        Does not reduce dimensions, just a filter)"""
        eigenvecs = self._check_fourier_properties('U')
        x_hat = x @ eigenvecs 
        x_hat_filtered = np.zeros(len(x_hat))
        x_hat_filtered[self.cutoffs[0]:self.cutoffs[1]] = x_hat[self.cutoffs[0]:self.cutoffs[1]]
        
        return x_hat_filtered @ eigenvecs.T
        
    
    def _fit(self, x, column_labels_path):
        """ Converts input signal x to a signal on the network nodes by mapping each 
        entry of x to a node using the column_labels and then applies the desired
        transformation. Nodes that represnt genes that are not present in 
        the input get 0"""
        column_labels = pd.read_csv(column_labels_path)
        
        if not (isinstance(column_labels, pd.DataFrame)):
            raise TypeError("Column labels must be pandas DataFrame")
        
        signal = np.zeros((x.shape[0], self.n_nodes))
        node2feature = self._subset_in_network(column_labels)
        
        # Assign the value of a feature to the signal at the index of the representing node
        signal[:,node2feature[:,0]] = x[:, node2feature[:,1]]
        
        if self.method == 'gs':
            x_tf = self._graph_sampling(signal)
        elif self.method == 'gbf':
            x_tf = self._gbf(signal)
        elif self.method == 'filter':
            x_tf = self._simple_filter(signal)
            x_tf = x_tf[:,node2feature[:,0]]
        else:
            x_tf = signal
            
        self.X_transformed = x_tf
        
        return x_tf
    
    def fit(self, x, column_labels_path):
        self._fit(x, column_labels_path)
        return self
        
    def fit_transform(self, x, column_labels_path):
        x_transform = self._fit(x, column_labels_path)
        return x_transform
        
    
        
    