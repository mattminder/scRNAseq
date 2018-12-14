"""Contains function to train all models."""
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle as pkl
import sklearn.ensemble as ens
from sklearn import linear_model as lm
import numpy as np


def all_models_train(train_x, train_y, classif_folder):
    """
    Trains all of our models based on train_x and train_y, saves the resulting classifiers as pickle file
    in classif_folder.
    :param train_x: x set
    :param train_y: response
    :param classif_folder: folder in which all classifiers are saved
    :return: Nothing
    """
    train_y_zeroone = np.copy(train_y)
    train_y_zeroone[train_y_zeroone == -1] = 0
    train_y[train_y == 0] = -1

    # XGBOOST
    # Split into test and validation set
    train_x_split, val_x, train_y_split, val_y = train_test_split(train_x, train_y_zeroone,
                                                                  test_size=0.2, random_state=123)

    dtrain = xgb.DMatrix(train_x_split, label=train_y_split)
    dval = xgb.DMatrix(val_x, label=val_y)

    # Train
    print('xgboost')

    param = {'max_depth': 3,
             'eta': .2,
             'silent': 1,
             'objective': 'binary:logistic',
             'nthread': -1,
             'booster': 'gbtree',
             'eval_metric': 'auc'}

    evallist = [(dval, 'eval'), (dtrain, 'train')]

    num_round = 100
    xgboost_classif = xgb.train(param, dtrain, num_round, evallist)

    # Save
    savefile = open(classif_folder + 'xgboost_bf_classif.pkl', 'wb')
    pkl.dump(xgboost_classif, savefile)
    savefile.close()



    # RANDOM FOREST
    print('random forest')

    # Train
    rf_classif = ens.RandomForestClassifier(random_state=161,
                                            n_jobs=-1,
                                            verbose=1,
                                            n_estimators=5000
                                            ).fit(train_x,
                                                  train_y)

    # Save
    savefile = open(classif_folder + 'randomForest_bf_classif.pkl', 'wb')
    pkl.dump(rf_classif, savefile)
    savefile.close()



    # LOGISTIC LASSO
    print('logistic lasso')
    lasso_classif = lm.LogisticRegressionCV(penalty='l1',     # Lasso regularization
                                            Cs=40,            # Size of grid for parameter search
                                            verbose=0,
                                            cv=10,            # 10-Fold CV
                                            solver='saga',    # Stochastic Average Descent, able to handle l1-penalty
                                            fit_intercept=True,
                                            random_state=896, # Random seed
                                            n_jobs=-1,        # Use all CPU cores for computation
                                            tol=0.005         # Set tolerance for convergence of SGD
                                            ).fit(train_x,
                                                  train_y)

    # Save Classif
    savefile = open(classif_folder + 'log_lasso_bf_classif.pkl', 'wb')
    pkl.dump(lasso_classif, savefile)
    savefile.close()




