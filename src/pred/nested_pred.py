from helpers.application import *


def do_nested_pred():
    """
    Do predictions of nested classifier, save in corresponding folder.
    :return: Nothing
    """

    PRED_FOLDER = '../../res/pred/'
    CLASSIF_FOLDER = '../../res/classif/nested/'

    base_classif_to_use = ['base_features', 'filter_and_pca_high-pass', 'filter_and_pca_low-pass',
                           'filter_highpass', 'filter_lowpass', 'gbf_highpass', 'gbf_lowpass',
                           'gs-ref_highpass', 'gs-ref_lowpass', 'pca']

    ind_train_preds = load_true_pred([PRED_FOLDER + 'ind_train/' + s + '/' for s in base_classif_to_use])
    test_preds = load_true_pred([PRED_FOLDER + 'test/' + s + '/' for s in base_classif_to_use])
    joost_preds = load_true_pred([PRED_FOLDER + 'joost/' + s + '/' for s in base_classif_to_use])

    predict_all_methods(ind_train_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'ind_train/nested/',
                        save_type='both', ret=False)
    predict_all_methods(test_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'test/nested/',
                        save_type='both', ret=False)
    predict_all_methods(joost_preds, CLASSIF_FOLDER, save_folder=PRED_FOLDER+'joost/nested/',
                        save_type='both', ret=False)

