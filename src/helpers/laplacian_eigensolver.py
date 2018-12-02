# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 09:56:36 2018

@author: silus
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.manifold.spectral_embedding_ import _set_diag
from scipy import sparse
from scipy.sparse.linalg import eigsh

def get_laplacian_eig(adjacency, dims, normed=True, random_state=None):
    random_state = check_random_state(random_state)
    laplacian, dd = sparse.csgraph.laplacian(adjacency, normed=normed, return_diag=True)
    laplacian = _set_diag(laplacian, 1, True)
    laplacian *= -1
    v0 = random_state.uniform(-1, 1, laplacian.shape[0])
    lambdas, diffusion_map = eigsh(laplacian, k=dims, sigma=1.0, which='LM', tol=0.0, v0=v0)
    
    embedding = diffusion_map.T[dims::-1] * dd
    embedding = _deterministic_vector_sign_flip(embedding)
    
    return lambdas, embedding[:dims].T
    