from sklearn import linear_model as lm
import numpy as np
import helpers.data_load as dl
import pickle as pk

import os
print(os.getcwd())

# Loading file
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/base_features/'

train_x = np.load(RES_FOLDER + 'train_x_rescaled_nolovar.npy')
train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')
train_y[train_y == 0] = -1  # Encode as +-1


# Train Logistic Ridge
print('Fitting Model')
classif = lm.LogisticRegressionCV(penalty='l2',     # Lasso regularization
                                  Cs=10,            # Size of grid for parameter search
                                  verbose=0,
                                  cv=10,            # 10-Fold CV
                                  solver='saga',    # Stochastic Average Descent, able to handle l1-penalty
                                  fit_intercept=True,
                                  random_state=896, # Random seed
                                  n_jobs=3,         # Use 2 CPU cores for computation
                                  tol=0.005         # Set tolerance for convergence of SGD
                                  ).fit(train_x,
                                        train_y)

# Save Classif
savefile = open(RES_FOLDER + 'log_ridge_no_lovar_classif.txt', 'wb')
pk.dump(classif, savefile)
savefile.close()

# Load Herring & Joost
herring_x = np.load(RES_FOLDER+'herring2017_rescaled_nolovar.npy')
joost_x = np.load(RES_FOLDER+'joost2016_rescaled_nolovar.npy')

# Prediction
print('Predictions')
herring_pred = classif.predict_proba(herring_x)
joost_pred = classif.predict_proba(joost_x)
np.save(arr=herring_pred, file=RES_FOLDER+'log_ridge_no_lovar_preds_herring.npy')
np.save(arr=joost_pred, file=RES_FOLDER+'log_ridge_no_lovar_preds_joost.npy')

