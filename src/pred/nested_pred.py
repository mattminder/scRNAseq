from helpers.application import *

PRED_FOLDER = '../../res/pred/'
CLASSIF_FOLDER = '../../res/classif/nested/'
DATA_FOLDER = '../../data'

base_classif_to_use = ['base_features', 'filter_and_pca_high-pass', 'filter_and_pca_low-pass',
                       'filter_highpass', 'filter_lowpass', 'gbf_highpass', 'gbf_lowpass',
                       'gs-ref_highpass', 'gs-ref_lowpass', 'nested', 'pca']

print("Load Data")
ind_train_preds = load_true_pred([PRED_FOLDER + 'ind_train/' + s + '/' for s in base_classif_to_use])
test_preds = load_true_pred([PRED_FOLDER + 'test/' + s + '/' for s in base_classif_to_use])

print("Predict")
predict_all_methods(ind_train_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'ind_train/nested/',
                    save_type='both', ret=False)
predict_all_methods(test_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'test/nested/',
                    save_type='both', ret=False)

