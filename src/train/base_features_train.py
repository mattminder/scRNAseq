from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


def do_base_features_train(train_x, train_y):
    """
    Trains classifier on base features, save classifier in res/classif/base_features
    :param train_x: Train set
    :param train_y: Train response
    :return: Nothing
    """
    # Loading Training Data
    CLASSIF_FOLDER = '../../res/classif/base_features/'

    # Transforming Data
    print('Transforming Data')
    nolowvar = fit_nolowvar(train_x, CLASSIF_FOLDER, ret=True)
    x = nolowvar.transform(train_x)

    # Training Classifier
    print('Training Classifiers')
    all_models_train(x, train_y, CLASSIF_FOLDER)

