from helpers.application import load_true_pred
import re
from helpers.data_load import load_response
from sklearn.metrics import roc_auc_score
from helpers.application import binarize
from sklearn.metrics import accuracy_score
import csv


DATA_FOLDER = '../../data/'
OUT_FOLDER = '../../res/'


# --- TEST ----
# Load Predictions on Test
test_preds, filelist = load_true_pred('../../res/pred/test/', ret_filelist=True, stack=False)
pred_names = [re.sub('.+/test/', '', s) for s in filelist]
pred_names = [re.sub('_pred.npy', '', s) for s in pred_names]
pred_method = [re.sub('.+/', '', s) for s in pred_names]
pred_transf = [re.sub('/.+', '', s) for s in pred_names]


# Load true response
test_y, cell_names_y = load_response(DATA_FOLDER + 'test_response.csv.gz')

# Compute AUC
auc_score = [roc_auc_score(test_y, pred) for pred in test_preds]

# Labels
test_pred_labels = [binarize(p) for p in test_preds]
accuracy = [accuracy_score(test_y, p) for p in test_pred_labels]

# Save as csv
with open(OUT_FOLDER+'test_performance_summary.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(pred_transf, pred_method, auc_score, accuracy))


# --- IND TRAIN ----
# Load Predictions on Test
ind_train_preds, filelist = load_true_pred('../../res/pred/ind_train/', ret_filelist=True, stack=False)
pred_names = [re.sub('.+/ind_train/', '', s) for s in filelist]
pred_names = [re.sub('_pred.npy', '', s) for s in pred_names]
pred_names = [re.sub('/', ', ', s) for s in pred_names]


# Load true response
ind_train_y, cell_names_y = load_response(DATA_FOLDER + 'ind_response.csv.gz')

# Compute AUC
auc_score = [roc_auc_score(ind_train_y, pred) for pred in ind_train_preds]

# Labels
ind_train_pred_labels = [binarize(p) for p in ind_train_preds]
accuracy = [accuracy_score(ind_train_y, p) for p in ind_train_pred_labels]

# Save as csv
with open(OUT_FOLDER+'ind_train_performance_summary.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(pred_transf, pred_method, auc_score, accuracy))