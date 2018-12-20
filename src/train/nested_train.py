from helpers.application import *
from helpers.data_load import *
from helpers.classifier_train import *

PRED_FOLDER = '../../res/pred/'
CLASSIF_FOLDER = '../../res/classif/nested/'
DATA_FOLDER = '../../data/'

base_classif_to_use = ['base_features', 'filter_and_pca_high-pass', 'filter_and_pca_low-pass',
                       'filter_highpass', 'filter_lowpass', 'gbf_highpass', 'gbf_lowpass',
                       'gs-ref_highpass', 'gs-ref_lowpass', 'nested', 'pca']

print("Load Data")
ind_train_preds = load_true_pred([PRED_FOLDER + 'ind_train/' + s + '/' for s in base_classif_to_use])
test_preds = load_true_pred([PRED_FOLDER + 'test/' + s + '/' for s in base_classif_to_use])

ind_train_y, cell_names_y = load_response(DATA_FOLDER + 'ind_response.csv.gz')

print("Train Model")
all_models_train(ind_train_preds, ind_train_y, CLASSIF_FOLDER)
