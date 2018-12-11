from sklearn import ensemble as ens
import numpy as np
import helpers.data_load as dl
import pickle as pk

import os
print(os.getcwd())

# Loading files
print('Loading Trainset')
DATA_FOLDER = '../../../results/pca/'
RES_FOLDER = '../../../results/pca/'

# Nb PCs to use
n_pcs=500

train_x = np.load(DATA_FOLDER+'train_500pcs.npy')[:, :n_pcs]
train_y, cell_names_y = dl.load_response(DATA_FOLDER + '../../data/response.csv.gz')
train_y[train_y == 0] = -1  # Encode as +-1

# Train Logistic Lasso
print('Fitting Model')
classif = ens.AdaBoostClassifier(random_state=161,
                                 n_estimators=1000
                                 ).fit(train_x,
                                       train_y)

# Save Classif
savefile = open(RES_FOLDER + 'adaboost_' + str(n_pcs) + 'pcs_classif.txt', 'wb')
pk.dump(classif, savefile)
savefile.close()

# Load Herring 2017 Data
print('Loading Herring 2017')
herring_x = np.load(DATA_FOLDER + 'herring_500pcs.npy')[:, :n_pcs]

# Load Joost 2016 Data
print('Loading Joost 2016')
joost_x = np.load(DATA_FOLDER + 'joost_500pcs.npy')[:, :n_pcs]

# Prediction
print('Predictions')
herring_pred = classif.predict_proba(herring_x)
joost_pred = classif.predict_proba(joost_x)
np.save(arr=herring_pred, file=RES_FOLDER+'adaboost_' + str(n_pcs) + 'pcs_preds_herring.npy')
np.save(arr=joost_pred, file=RES_FOLDER+'adaboost_' + str(n_pcs) + 'pcs_preds_joost.npy')
