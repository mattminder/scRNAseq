from helpers.application import *

PRED_FOLDER = '../../res/pred/'
CLASSIF_FOLDER = '../../res/classif/nested/'
DATA_FOLDER = '../../data'

print("Load Data")
ind_train_preds = load_true_pred(PRED_FOLDER + 'ind_train/')
test_preds = load_true_pred(PRED_FOLDER + 'test/')

print("Predict")
predict_all_methods(ind_train_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'ind_train/nested/',
                    save_type='both', ret=False)
predict_all_methods(test_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'test/nested/',
                    save_type='both', ret=False)

