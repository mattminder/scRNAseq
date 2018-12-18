# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:24:33 2018

@author: Zora
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as utils
from helpers.neural_net import NeuralNet


def nn_train(x_train, y_train, classif_folder, lr=7e-2, reg=2e-10, momentum=0.95, epochs=10):
    """
    Trains the neural network and stores the solution in the classif_folder
    :param x_train: data to be trained on
    :param y_train: response (labels)
    :param classif_folder: folder in which all classifiers are saved
    :param lr: (optional) learning rate at the beginning of the training
    :param reg: (optional) L2 regularization weight
    :param momentum: (optional) momentum of the optimization
    :param epochs: (optional) number of training epochs
    :return: Nothing
    """
    
    X_train, y_train = torch.from_numpy(x_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor)

    traindataset = utils.TensorDataset(X_train, y_train)
    trainloader = utils.DataLoader(traindataset, batch_size=100, shuffle=True)

    # Initialization of neural net
    net = NeuralNet(n_input_channels=X_train.shape[1])
    criterion = torch.nn.CrossEntropyLoss()

    # Training and validation
    for e in range(epochs):
        print(e)
        learn_rate = lr*0.01**(e/epochs) # Learning rate decay, starting from lr given
        optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate, weight_decay=reg, momentum=momentum)
        for data, labels in iter(trainloader):
            # transform inputs and outputs into Variable
            inputs, targets = Variable(data), Variable(labels)

            # set gradient to zero
            optimizer.zero_grad()

            # forward pass
            out = net.forward(inputs)

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

    # Save end state
    location = classif_folder + 'neuralNet.pt'            
    torch.save(net.state_dict(), location)
    
    
def nn_predict(X, classif_path, result_file_np=None, result_file_csv=None):
    """
    Predicts the class for new data with a trained neural network
    :param X: data to predict class on
    :param classif_path: path to trained neural net (should have .pt extension)
    :param result_file_np: (optional) path to where the predictions should be saved as numpy array, not saved if none given
    :param result_file_csv: (optional) path to where the predictions should be saved as csv file, not saved if none given
    :return: numpy array with the predictions
    """
    # Predict
    data_pts = Variable(torch.from_numpy(X).type(torch.FloatTensor))
    Net = NeuralNet(n_input_channels=X.shape[1], n_output=2)
    Net.load_state_dict(torch.load(classif_path))
    output = Net.predict(data_pts)
    pred = output.data.numpy()
    # Save predictions
    if result_file_np is not None:
        np.save(result_file_np, pred)
    if result_file_csv is not None:
        np.savetxt(result_file_csv, pred, delimiter=",")
    return pred
