# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:09:05 2018

@author: zoswald
"""

from torch import nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, n_input_channels=500, n_output=2):
        super().__init__()
        self.fc1 = nn.Linear(n_input_channels, n_input_channels*5)
        self.fc2 = nn.Linear(n_input_channels*5, n_input_channels*2)
        self.fc3 = nn.Linear(n_input_channels*2, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 20)
        self.fc6 = nn.Linear(20,n_output)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)