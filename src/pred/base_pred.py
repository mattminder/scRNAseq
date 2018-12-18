"""
Creates predictions based on base_features classifiers.
"""

from helpers.data_load import *
from helpers.application import *

# --- LOADING ---
print('Loading')
DATA_FOLDER = '../../data/'
CLASSIF_FOLDER = '../../res/classif/'
OUT_FOLDER = '../../res/pred/'
subfolders = ['base_features/', 'pca/', 'filter_lowpass/', 'filter_highpass/', 'gbf_highpass/', 'gbf_lowpass/',
              'gs_lowpass/', 'gs_highpass/' ]
classif_subfolders = [CLASSIF_FOLDER+s for s in subfolders]


# Load Trainset
train_x, gene_names_x, cell_names_x = load_data(DATA_FOLDER + 'train_data.csv.gz')
train_x = train_x[:, 1:]
train_y, cell_names_y = load_response(DATA_FOLDER + 'response.csv.gz')

## Load Generalization Set
#ind_train_x, ind_gene_names_x, ind_cell_names_x = load_data(DATA_FOLDER + 'ind_train_data.csv.gz')
#ind_train_x = ind_train_x[:, 1:]
#ind_train_y, ind_cell_names_y = load_response(DATA_FOLDER + 'ind_response.csv.gz')

## Load Test Set
#test_x, test_gene_names_x, test_cell_names_x = load_data(DATA_FOLDER + 'test_data.csv.gz')
#test_x = test_x[:, 1:]
#test_y, test_cell_names_y = load_response(DATA_FOLDER + 'test_response.csv.gz')


# --- TRANSFORMATIONS ---
print('Transformations')
transf_list_paths = ['base_features/nolowvar_fit.pkl',
                     'pca/pca_fit.txt',
                     'filter_lowpass/netw_pca_filter_low-pass.txt',
                     'filter_highpass/netw_pca_filter_high-pass.txt',
                     'gbf_highpass/netw_pca_gbf_high-pass.txt',
                     'gbf_lowpass/netw_pca_gbf_low-pass.txt',
                     'gs_highpass/netw_pca_gs_high-pass.txt',
                     'gs_lowpass/netw_pca_gs_low-pass.txt']

transf_list_paths = [CLASSIF_FOLDER+s for s in transf_list_paths]
transf_type_list = ['skl', 'skl', 'network', 'network', 'network', 'network', 'network', 'network']
transf_names = ['nolowvar', 'pca', 'filter_low', 'filter_high', 'gbf_high', 'gbf_low', 'gs_low', 'gs_high']

train_transf_list = transform_multiple(train_x, transf_list_paths, transf_type_list)
#test_transf_list = transform_multiple(test_x, transf_list_paths, transf_type_list)
#ind_train_transf_list = transform_multiple(ind_train_x, transf_list_paths, transf_type_list)


# --- Predictions ---
print('Predictions')
for i in range(len(train_transf_list)):
    transf = train_transf_list[i]
    s = subfolders[i]
    predict_all_methods(transf, CLASSIF_FOLDER+s, OUT_FOLDER+'train/'+s, save_type='both', ret=False)

#for i in range(len(test_transf_list)):
#    transf = test_transf_list[i]
#    s = subfolders[i]
#    predict_all_methods(transf, CLASSIF_FOLDER+s, OUT_FOLDER+'test/'+s, save_type='both', ret=False)

#for i in range(len(ind_train_transf_list)):
#    transf = ind_train_transf_list[i]
#    s = subfolders[i]
#    predict_all_methods(transf, CLASSIF_FOLDER+s, OUT_FOLDER+'ind_train/'+s, save_type='both', ret=False)


