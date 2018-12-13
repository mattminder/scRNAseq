from sklearn import linear_model as lm
import numpy as np
import helpers.data_load as dl
import pickle as pk

import os
print(os.getcwd())

# Loading files
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/'
OUT_FOLDER = RES_FOLDER + 'nested/'


def load_helper(folder, filelist):
    """
    Opens all numpy arrays in filelist, extracts second column containing true predictions,
    then stacks them to form an input for training a nested model.

    :param folder: base folder with all files to be later bound
    :param filelist: list of numpy prediction files to be included
    :return:
    """
    tmp = []
    for i in range(len(filelist)):
        tmp.append(np.load(folder+filelist[i])[:, 1])
    return np.stack(tmp, axis=1)


train_x = load_helper(RES_FOLDER+'train_predictions/',
                      ['log_lasso_500pcs_train_preds.npy',
                       'log_lasso_base_features_train_preds.npy',
                       'log_lasso_GBF_train_preds.npy',
                       'log_lasso_GS_train_preds.npy',
                       'randomForest_500pcs_train_preds.npy',
                       'randomForest_base_features_train_preds.npy',
                       'randomForest_GBF_train_preds.npy',
                       'randomForest_GS_train_preds.npy'])


train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')
train_y[train_y == 0] = -1  # Encode as +-1


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
                                  ).fit(train_x,

                                        train_y)
# Save Classif
savefile = open(OUT_FOLDER + 'log_lasso_nested_classif.txt', 'wb')
pk.dump(classif, savefile)
savefile.close()

# Load Herring and Joost
herring_x = load_helper(RES_FOLDER+'herring2017_predictions/',
                      ['log_lasso_500pcs_herring2017_preds.npy',
                       'log_lasso_base_features_herring2017_preds.npy',
                       'log_lasso_GBF_herring2017_preds.npy',
                       'log_lasso_GS_herring2017_preds.npy',
                       'randomForest_500pcs_herring2017_preds.npy',
                       'randomForest_base_features_herring2017_preds.npy',
                       'randomForest_GBF_herring2017_preds.npy',
                       'randomForest_GS_herring2017_preds.npy'])
joost_x = load_helper(RES_FOLDER+'joost2016_predictions/',
                      ['log_lasso_500pcs_joost2016_preds.npy',
                       'log_lasso_base_features_joost2016_preds.npy',
                       'log_lasso_GBF_joost2016_preds.npy',
                       'log_lasso_GS_joost2016_preds.npy',
                       'randomForest_500pcs_joost2016_preds.npy',
                       'randomForest_base_features_joost2016_preds.npy',
                       'randomForest_GBF_joost2016_preds.npy',
                       'randomForest_GS_joost2016_preds.npy'])



# Prediction
print('Predictions')
herring_pred = classif.predict_proba(herring_x)
joost_pred = classif.predict_proba(joost_x)
np.save(arr=herring_pred, file=OUT_FOLDER+'log_lasso_nested_preds_herring.npy')
np.save(arr=joost_pred, file=OUT_FOLDER+'log_lasso_nested_preds_joost.npy')
