# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 01:15:44 2018

@author: silus
"""

import numpy as np

def graph_sampling(eigenvecs, eigenvals, K, interval=None):
    """Graph sampling algorithm: Choose K eigenvalues with the highest
    Graph weighted coherence (signal energy) in the frequency band
    specified in interval."""
    
    if interval:
        index = np.arange(len(eigenvals))
        idx_low = min(index[eigenvals > interval[0]])
        idx_high = min(index[eigenvals > interval[1]])
    else:
        idx_low = 0
        idx_high = len(eigenvals)
    
    # Computed weighted coherence in given interval
    F_basis_sampled = eigenvecs[:,idx_low:idx_high]
    graph_weighted_coherence = np.sum(np.power(F_basis_sampled, 2), axis=1)
    
    # Choose K nodes with the highest weighted coherence
    keep_nodes = np.argsort(-graph_weighted_coherence)[0:K]
    
    return keep_nodes

def lowpass_kernel(e, t):
    def lpfilter(a, t):
        return 1/(1+t*a)
    lp_filter = np.vectorize(lpfilter)
    return lp_filter(e, t)

def highpass_kernel(e, t):
    def hpfilter(a, t):
        return t*a/(t*a+1)
    lp_filter = np.vectorize(hpfilter)
    return lp_filter(e, t)

def graph_filtering(U, e, x, t, kernel):
    filters = {'low_pass': lowpass_kernel, 'high_pass': highpass_kernel}
    
    # Get GFT of signal
    x_hat = U.T @ x
    
    # Filter
    x_hat_f = np.multiply(x_hat, filters[kernel](e, t))
    
    # Filtered, node domain with iGFT
    return U @ x_hat_f

    
        