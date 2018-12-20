# Loading Scripts
from helpers.data_load import load_response, load_data

# Training Scripts
from train.base_features_train import do_base_features_train
from train.filter_and_pca_highpass_train import do_filter_and_pca_highpass_train
from train.filter_and_pca_lowpass_train import do_filter_and_pca_lowpass_train
from train.filter_highpass_train import do_filter_highpass_train
from train.filter_lowpass_train import do_filter_lowpass_train
from train.gbf_highpass_train import do_gbf_highpass_train
from train.gbf_lowpass_train import do_gbf_lowpass_train
from train.gs_ref_highpass_train import do_gs_ref_highpass_train
from train.gs_ref_lowpass_train import do_gs_ref_lowpass_train
from train.pca_train import do_pca_train
from train.nested_train import do_nested_train

# Prediciton Scripts
from pred.base_pred import do_base_pred
from pred.nested_pred import do_nested_pred
from pred.performance_evaluation import do_performance_evaluation


# --- LOADING ---
DATA_FOLDER = '../../data/'

print('LOADING TRAIN')
# Load Trainset
train_x, gene_names_x, cell_names_x = load_data(DATA_FOLDER + 'train_data.csv.gz')
train_x = train_x[:, 1:]
train_y, cell_names_y = load_response(DATA_FOLDER + 'response.csv.gz')


"""
print('TRAIN BASE')
print('Base Features')
do_base_features_train(train_x, train_y)
print('Filter and PCA High')
do_filter_and_pca_highpass_train(train_x, train_y)
print('Filter and PCA Low')
do_filter_and_pca_lowpass_train(train_x, train_y)
print('Filter High')
do_filter_highpass_train(train_x, train_y)
print('Filter Low')
"""
do_filter_lowpass_train(train_x, train_y)

"""
print('GBF High')
do_gbf_highpass_train(train_x, train_y)
print('GBF Low')
do_gbf_lowpass_train(train_x, train_y)
print('GS High')
do_gs_ref_highpass_train(train_x, train_y)
print('GS Low')
do_gs_ref_lowpass_train(train_x, train_y)
print('PCA Train')
do_pca_train(train_x, train_y)
"""

print('LOAD TEST AND IND TRAIN')
# Load Generalization Set
ind_train_x, ind_gene_names_x, ind_cell_names_x = load_data(DATA_FOLDER + 'ind_train_data.csv.gz')
ind_train_x = ind_train_x[:, 1:]
ind_train_y, ind_cell_names_y = load_response(DATA_FOLDER + 'ind_response.csv.gz')

# Load Test Set
test_x, test_gene_names_x, test_cell_names_x = load_data(DATA_FOLDER + 'test_data.csv.gz')
test_x = test_x[:, 1:]
test_y, test_cell_names_y = load_response(DATA_FOLDER + 'test_response.csv.gz')

# Load Joost Set
joost_x, joost_gene_names_x, joost_cell_names_x = load_data(DATA_FOLDER + 'joost2016_data.csv.gz')
joost_x = joost_x[:, 1:]


print('PREDICT BASE')
do_base_pred(train_x, ind_train_x, test_x, joost_x)
print('Nested Train')
do_nested_train()

print('TRAIN NESTED')
do_nested_train()
print('PREDICT NESTED')
do_nested_pred()

print('PERFORMANCE EVALUATION')
do_performance_evaluation()
