from helpers.application import *
from helpers.data_load import *
from helpers.classifier_train import *

PRED_FOLDER = '../../res/pred/'
CLASSIF_FOLDER = '../../res/classif/nested/'
DATA_FOLDER = '../../data/'

print("Load Data")
ind_train_preds = load_true_pred(PRED_FOLDER + 'ind_train/')
ind_train_y, cell_names_y = load_response(DATA_FOLDER + 'ind_response.csv.gz')

print("Train Model")
all_models_train(ind_train_preds, ind_train_y, CLASSIF_FOLDER)
