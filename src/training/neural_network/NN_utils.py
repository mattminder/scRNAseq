# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:42:38 2018

@author: Zora
"""

import csv
import numpy as np

def read_csv_to_array(name):
    reader = csv.reader(open(name, "rt"), delimiter=",")
    x = list(reader)
    #result = np.array(x).astype("float")
    result = np.array(x)
    return result


def read_labels(name, source=False):
    array = read_csv_to_array(name)
    labels = array[1:,1].astype("int")
    if source:
        sources = array[1:,0]
        return labels, sources
    return labels
    

def read_data(name, gene=False, source=False):
    array = read_csv_to_array(name)
    data = array[:,:].astype("float")
    if source:
        sources = array[1:,0]
    if gene:
        genes = array[0,1:]
        if source:
            return data, genes, sources
        else:
            return data, genes
    if source:
        return data, sources
    return data