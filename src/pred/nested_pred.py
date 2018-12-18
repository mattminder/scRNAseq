from helpers.application import *

PRED_FOLDER = '../../res/pred/'
CLASSIF_FOLDER = '../../res/classif/nested/'
DATA_FOLDER = '../../data'

print("Load Data")
ind_train_preds = load_true_pred(PRED_FOLDER + 'ind_train/')

print("Train Model")
predict_all_methods(ind_train_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'nested/ind_train/',
                    save_type='both', ret=False)
predict_all_methods(ind_train_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'nested/test/',
                    save_type='both', ret=False)

