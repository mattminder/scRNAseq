import xgboost as xgb
import numpy as np
import helpers.data_load as dl
from sklearn.model_selection import train_test_split

import os
print(os.getcwd())

# Loading files
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/network_GBF/'

train_x = np.load(DATA_FOLDER+'train_data_GBF.npy')
train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')
#train_y[train_y == 0] = -1  # Encode as +-1

# Split into test and validation set
train_x_split, val_x, train_y_split, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=123)

dtrain = xgb.DMatrix(train_x_split, label=train_y_split)
dval = xgb.DMatrix(val_x, label=val_y)

# Train Model
print('Fitting Model')

param = {'max_depth': 3,
         'eta': .3,
         'silent': 1,
         'objective': 'binary:logistic',
         'nthread': 2,
         'booster': 'gbtree',
         'eval_metric': 'auc'}

evallist = [(dval, 'eval'), (dtrain, 'train')]

num_round = 100
classif = xgb.train(param, dtrain, num_round, evallist)

#classif = xgb.XGBClassifier(silent=True,        # verbose output
#                            n_jobs=2,           # nb processor cores
#                            max_depth=3,        # tree depth
#                            random_state=161,   # Random state
#                            objective='binary:logistic'
#                            ).fit(train_x, train_y)





# Save Classif
classif.save_model(RES_FOLDER + 'xgboost_GBF_classif.model')
classif.dump_model(RES_FOLDER + 'xgboost_GBF_classif.txt')

# Load Herring 2017 Data
print('Loading Herring 2017')
herring_x = np.load(DATA_FOLDER + 'herring2017_data_GBF.npy')

# Load Joost 2016 Data
print('Loading Joost 2016')
joost_x = np.load(DATA_FOLDER + 'joost2016_data_GBF.npy')

# Prediction
print('Predictions')
herring_pred = classif.predict(data=xgb.DMatrix(herring_x),
                               validate_features=False)
joost_pred = classif.predict(data=xgb.DMatrix(joost_x),
                             validate_features=False)

np.save(arr=herring_pred, file=RES_FOLDER+'xgboost_GBF_preds_herring.npy')
np.save(arr=joost_pred, file=RES_FOLDER+'xgboost_GBF_preds_joost.npy')
np.savetxt(RES_FOLDER+'xgboost_GBF_preds_herring.csv', herring_pred, delimiter=',')
np.savetxt(RES_FOLDER+'xgboost_GBF_preds_joost.csv', joost_pred, delimiter=',')

