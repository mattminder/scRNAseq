from sklearn import linear_model as lm
from sklearn import preprocessing as prep
from sklearn import feature_selection as fsel
import numpy as np
import helpers.data_load as dl
import pickle as pk

import os
print(os.getcwd())

# Loading file
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/base_features/'

train_x, gene_names_x, cell_names_x = dl.load_data(DATA_FOLDER + 'train_data.csv.gz')

# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
train_x = train_x[:, 1:]
gene_names_x = gene_names_x[1:]
cell_names_x = cell_names_x[1:]

train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')
train_y[train_y == 0] = -1  # Encode as +-1


# Preprocessing
varThresh = fsel.VarianceThreshold(threshold=0.1).fit(train_x)
scaled_x = prep.scale(varThresh.transform(train_x))

# Train Logistic Lasso
print('Fitting Model')
classif = lm.LogisticRegressionCV(penalty='l1',     # Lasso regularization
                                  Cs=10,            # Size of grid for parameter search
                                  verbose=0,
                                  cv=10,            # 10-Fold CV
                                  solver='saga',    # Stochastic Average Descent, able to handle l1-penalty
                                  fit_intercept=True,
                                  random_state=896, # Random seed
                                  n_jobs=3,         # Use 2 CPU cores for computation
                                  tol=0.005         # Set tolerance for convergence of SGD
                                  ).fit(scaled_x,
                                        train_y)

# Save Classif
savefile = open(RES_FOLDER + 'log_lasso_no_lovar_classif.txt', 'wb')
pk.dump(classif, savefile)
savefile.close()

# Load Herring 2017 Data
print('Loading Herring 2017')
herring_x, gene_names_herring, cell_names_herring = dl.load_data(DATA_FOLDER + 'herring2017_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
herring_x = herring_x[:, 1:]
herring_scaled = prep.scale(varThresh.transform(herring_x))

# Load Joost 2016 Data
print('Loading Joost 2016')
joost_x, gene_names_joost, cell_names_joost = dl.load_data(DATA_FOLDER + 'joost2016_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
joost_x = joost_x[:, 1:]
joost_scaled = prep.scale(varThresh.transform(joost_x))

# Prediction
print('Predictions')
herring_pred = classif.predict_proba(herring_scaled)
joost_pred = classif.predict_proba(joost_scaled)
np.save(arr=herring_pred, file=RES_FOLDER+'log_lasso_no_lovar_preds_herring.npy')
np.save(arr=joost_pred, file=RES_FOLDER+'log_lasso_no_lovar_preds_joost.npy')

