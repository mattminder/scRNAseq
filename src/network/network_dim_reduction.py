# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:44:52 2018

@author: silus
"""

import numpy as np
from helpers_network.data_import import transform_data_with_network
import matplotlib.pyplot as plt


# Two dimensionality reduction techniques based on GSP:
# 1. Graph sampling based on vertices where signal energy is the most concentrated
# 2. Graph-based Filtering (GBF) according to Rui et al. 

n_nodes_lc = 20289
K = 2570

e = np.load('eigenvalues_normalized.npy')
U = np.load('eigenvectors_normalized.npy')

x_GS_LF_ind_train, x_GS_HF_ind_train, x_GBF_LF_ind_train, x_GBF_HF_ind_train = transform_data_with_network('ind_train_data', n_nodes_lc, U, e, K)
np.save('../../data/ind_train_data_GS_LF.npy', x_GS_LF_ind_train)
np.save('../../data/ind_train_data_GBF_HF.npy', x_GBF_HF_ind_train)
np.save('../../data/ind_train_data_GS_HF.npy', x_GS_HF_ind_train)
np.save('../../data/ind_train_data_GBF_LF.npy', x_GBF_LF_ind_train)

x_GS_LF_test, x_GS_HF_test, x_GBF_LF_test, x_GBF_HF_test = transform_data_with_network('test_data', n_nodes_lc, U, e, K)
np.save('../../data/test_data_GS_LF.npy', x_GS_LF_test)
np.save('../../data/test_data_GBF_HF.npy', x_GBF_HF_test)
np.save('../../data/test_data_GS_HF.npy', x_GS_HF_test)
np.save('../../data/test_data_GBF_LF.npy', x_GBF_LF_test)






