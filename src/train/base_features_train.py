from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


# Loading Training Data
print('Loading Trainset')
DATA_FOLDER = '../../data/'
CLASSIF_FOLDER = '../../res/classif/base_features/'

train_x, gene_names_x, cell_names_x = load_data(DATA_FOLDER + 'train_data.csv.gz')
train_x = train_x[:, 1:]
gene_names_x = gene_names_x[1:]
cell_names_x = cell_names_x[1:]

train_y, cell_names_y = load_response(DATA_FOLDER + 'response.csv.gz')


# Transforming Data
print('Transforming Data')
nolowvar = fit_nolowvar(train_x, CLASSIF_FOLDER, ret=True)
x = nolowvar.transform(train_x)


# Training Classifier
print('Training Classifiers')
all_models_train(x, train_y, CLASSIF_FOLDER)

