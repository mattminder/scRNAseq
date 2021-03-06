from helpers.classifier_application import *
import numpy as np

OUT_FOLDER = '../../../results/joost2016_predictions/'
CLASSIF_FOLDER = '../../../results/'
DATA_FOLDER = '../../../data/'

# Function to generate prediction and save to out_path
def prediction_handler(x_path, classif_path, out_path):
    preds = predict_from_files(x_path, classif_path)
    np.save(file=out_path, arr=preds)


# No low-var features
NO_LOW_VAR_FOLDER = '../../../results/base_features/'

prediction_handler(NO_LOW_VAR_FOLDER+'joost2016_rescaled_nolovar.npy',
                   CLASSIF_FOLDER+'base_features/randomForest_classif.txt',
                   OUT_FOLDER+'randomForest_base_features_joost2016_preds.npy')

prediction_handler(NO_LOW_VAR_FOLDER+'joost2016_rescaled_nolovar.npy',
                   CLASSIF_FOLDER+'base_features/log_lasso_no_lovar_classif.txt',
                   OUT_FOLDER+'log_lasso_base_features_joost2016_preds.npy')

prediction_handler(NO_LOW_VAR_FOLDER+'joost2016_rescaled_nolovar.npy',
                   CLASSIF_FOLDER+'base_features/log_ridge_no_lovar_classif.txt',
                   OUT_FOLDER+'log_ridge_base_features_joost2016_preds.npy')


# 500 PCs
PCA_FOLDER = '../../../results/pca/'

prediction_handler(PCA_FOLDER+'joost_500pcs.npy',
                   CLASSIF_FOLDER+'pca/log_lasso_500pcs_classif.txt',
                   OUT_FOLDER+'log_lasso_500pcs_joost2016_preds.npy')

prediction_handler(PCA_FOLDER+'joost_500pcs.npy',
                   CLASSIF_FOLDER+'pca/log_ridge_500pcs_classif.txt',
                   OUT_FOLDER+'log_ridge_500pcs_joost2016_preds.npy')

prediction_handler(PCA_FOLDER+'joost_500pcs.npy',
                   CLASSIF_FOLDER+'pca/randomForest_500pcs_classif.txt',
                   OUT_FOLDER+'randomForest_500pcs_joost2016_preds.npy')

# GS
prediction_handler(DATA_FOLDER+'joost2016_data_GS.npy',
                   CLASSIF_FOLDER+'network_GS/log_lasso_GS_classif.txt',
                   OUT_FOLDER+'log_lasso_GS_joost2016_preds.npy')

prediction_handler(DATA_FOLDER+'joost2016_data_GS.npy',
                   CLASSIF_FOLDER+'network_GS/log_ridge_GS_classif.txt',
                   OUT_FOLDER+'log_ridge_GS_joost2016_preds.npy')

prediction_handler(DATA_FOLDER+'joost2016_data_GS.npy',
                   CLASSIF_FOLDER+'network_GS/randomForest_GS_classif.txt',
                   OUT_FOLDER+'randomForest_GS_joost2016_preds.npy')

# GBF
prediction_handler(DATA_FOLDER+'joost2016_data_GBF.npy',
                   CLASSIF_FOLDER+'network_GBF/log_lasso_GBF_classif.txt',
                   OUT_FOLDER+'log_lasso_GBF_joost2016_preds.npy')

prediction_handler(DATA_FOLDER+'joost2016_data_GBF.npy',
                   CLASSIF_FOLDER+'network_GBF/log_ridge_GBF_classif.txt',
                   OUT_FOLDER+'log_ridge_GBF_joost2016_preds.npy')

prediction_handler(DATA_FOLDER+'joost2016_data_GBF.npy',
                   CLASSIF_FOLDER+'network_GBF/randomForest_GBF_classif.txt',
                   OUT_FOLDER+'randomForest_GBF_joost2016_preds.npy')


