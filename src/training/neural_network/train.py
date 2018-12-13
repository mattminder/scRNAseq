
# coding: utf-8

import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.utils.data as utils
from neural_net import NeuralNet
from NN_utils import read_labels, read_data
np.random.seed(17)

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
#percent = 0.15 # If we want to validate on a part of the train data
percent = 0
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
trainloader = utils.DataLoader(traindataset, batch_size=100, shuffle=True)
if len(y_val) > 0:
    val_batch_size = 50
    valdataset = utils.TensorDataset(X_val, y_val)
    valloader = utils.DataLoader(valdataset, batch_size=val_batch_size, shuffle=True)

# Choice of hyperparameters
net = NeuralNet(n_input_channels=X_train.shape[1])
lr = 6.6*10**(-2)
reg = 2.2*10**(-10)
momentum = 0.95
criterion = torch.nn.CrossEntropyLoss()
epochs = 10

# Training and validation
steps = 0
running_loss = 0
print_every = 10
for e in range(epochs):
    lr*0.01**(e/epochs) # Learning rate decay, starting from best lr from val_hyperparameters
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, weight_decay = reg, momentum = momentum)
    start = time.time()
    for data, labels in iter(trainloader):
        steps += 1
        #transofrm inputs and outputs into Variable 
        inputs, targets = Variable(data), Variable(labels)

        #set gradient to zero
        optimizer.zero_grad()

        # forward pass
        out = net.forward(inputs)

        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        if steps % print_every == 0 and len(y_val) > 0:
            stop = time.time()
            # Validation accuracy
            accuracy = 0
            for ii, (data, labels) in enumerate(valloader):
                out = net.predict(Variable(data))
                _, prediction = torch.max(out, 1)
                #pred_y = prediction.data.cpu().numpy().squeeze() #if run on cluster
                pred_y = prediction.data.numpy().squeeze()
                target_y = (labels.numpy()).data
                accuracy += sum(pred_y == target_y)/val_batch_size

            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Loss: {:.4f}..".format(running_loss/print_every),
                  "Test accuracy: {:.4f}..".format(accuracy/(ii+1)),
                  "{:.4f} s/batch..".format((stop - start)/print_every),
                  "Learning rate: {:.3e}".format(lr)
                 )
            running_loss = 0
            start = time.time()

# Save end state            
torch.save(net.state_dict(), 'neuralNet.pt')



