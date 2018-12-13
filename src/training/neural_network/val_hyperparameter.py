
# coding: utf-8


import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.utils.data as utils
from neural_net import NeuralNet
from NN_utils import read_labels, read_data

# Load data
#x_name = 'data/train_data_GS.csv' 
y_name = 'data/response.csv'

#X = read_data(x_name)
y = read_labels(y_name)

X = np.load('data/train_500pcs.npy')

print(X.shape)
print(y.shape)


# Split into train and validation
# such that nb of class in train and validation are equal
percent = 0.15
idx_true = np.random.permutation([i for i in range(len(y)) if y[i]==1]) # permute so that choice is random when taking the first n elements
idx_false = np.random.permutation([i for i in range(len(y)) if y[i]==0])
idx_val = np.append(idx_true[:int(len(idx_true)*percent)],idx_false[:int(len(idx_false)*percent)])
idx_train = np.append(idx_true[int(len(idx_true)*percent):],idx_false[int(len(idx_false)*percent):])

X_train = X[idx_train,:]
y_train = y[idx_train]
X_val = X[idx_val,:]
y_val = y[idx_val]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)


X_train, y_train = torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor)
X_val, y_val = torch.from_numpy(X_val).type(torch.FloatTensor), torch.from_numpy(y_val).type(torch.LongTensor)

traindataset = utils.TensorDataset(X_train, y_train)
trainloader = utils.DataLoader(traindataset, batch_size=50, shuffle=True)

val_batch_size = 50
valdataset = utils.TensorDataset(X_val, y_val)
valloader = utils.DataLoader(valdataset, batch_size=val_batch_size, shuffle=True)


# Choice of hyperparameters
n_combinations = 20
lr = 10**(np.random.uniform(-2,-1,n_combinations)) # Random numbers between 10^-10 and 10^0 (log-scale)
reg = 10**(np.random.uniform(-10,-7,n_combinations)) # Random numbers between 10^-10 and 10^0 (log-scale)
momentum = 0.95
criterion = torch.nn.CrossEntropyLoss()
epochs = 1


# Try different learning rates and regularizations
for n in range(n_combinations):
    net = NeuralNet(n_input_channels=X_train.shape[1]) # Reinitialize neural net
    optimizer = torch.optim.SGD(net.parameters(), lr = lr[n], weight_decay = reg[n], momentum = momentum)
    # Training and validation
    for e in range(epochs):
        for data, labels in iter(trainloader):
            #transofrm inputs and outputs into Variable 
            inputs, targets = Variable(data), Variable(labels)

            #set gradient to zero
            optimizer.zero_grad()

            # forward pass
            out = net.forward(inputs)

            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()


    # Validation accuracy
    accuracy = 0
    for ii, (data, labels) in enumerate(valloader):
        out = net.predict(Variable(data))
        _, prediction = torch.max(out, 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = (labels.numpy()).data
        accuracy += sum(pred_y == target_y)/val_batch_size

    print("val_acc: {:.4f}..".format(accuracy/(len(valloader))),
          "lr: {:.6e}..".format(lr[n]),
          "reg: {:.6e}..".format(reg[n]),
          "momentum: {:.4f}..".format(momentum)
          )




