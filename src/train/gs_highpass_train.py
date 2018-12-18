from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


# Loading Training Data
print('Loading Trainset')
DATA_FOLDER = '../../data/'
CLASSIF_FOLDER = '../../res/classif/gs_highpass/'
NETWORK_FOLDER = '../../src/network/'
train_x, gene_names_x, cell_names_x = load_data(DATA_FOLDER + 'train_data.csv.gz')
train_x = train_x[:, 1:]
gene_names_x = gene_names_x[1:]
cell_names_x = cell_names_x[1:]

train_y, cell_names_y = load_response(DATA_FOLDER + 'response.csv.gz')


# Transforming Data
print('Transforming Data')
transf = fit_networkPCA(train_x, CLASSIF_FOLDER, ret=True, network_folder=NETWORK_FOLDER,
                        method='gs', attenuation='high-pass',
                        fourier_basis_path=(NETWORK_FOLDER+'eigvalues_combinatorial.npy',
                                            NETWORK_FOLDER+'eigvectors_combinatorial.npy'))
x = transf.fit_transform(train_x, '../network/genes_in_data.csv')


# Training Classifier
print('Training Classifiers')
all_models_train(x, train_y, CLASSIF_FOLDER)
