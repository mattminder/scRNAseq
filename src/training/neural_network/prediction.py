# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:58:04 2018

@author: zoswald
"""

# Import the necessary packages
import numpy as np
import torch
from torch.autograd import Variable
from neural_net import NeuralNet

def prediction(data_np_array, result_file_np=None, result_file_csv=None):
    X_test = np.load(data_np_array)
    pred_test = predict(X_test)
    # Save predictions
    if result_file_np is not None:
        np.save(result_file_np, pred_test)
    if result_file_csv is not None:
        np.savetxt(result_file_csv, pred_test, delimiter=",")
    return pred_test
    
def predict(X):
    data_pts = Variable(torch.from_numpy(X).type(torch.FloatTensor))
    Net = NeuralNet(n_input_channels=X.shape[1], n_output=2)
    Net.load_state_dict(torch.load('neuralNet.pt'))
    output = Net.predict(data_pts)
    #_, prediction = torch.max(output.data, 1)
    #pred_y = prediction.data.cpu().numpy().squeeze()
    pred_y = output.data.numpy()
    return pred_y
