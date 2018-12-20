from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


# Loading Training Data
def do_pca_train(train_x, train_y):
    """
    Trains classifier based on pca transformed data
    :param train_x: Train set
    :param train_y: Train response
    :return:
    """
    CLASSIF_FOLDER = '../../res/classif/pca/'

    # Transforming Data
    print('Transforming Data')
    pca = fit_pca(train_x, CLASSIF_FOLDER, ret=True)
    x = pca.transform(train_x)

    # Training Classifier
    print('Training Classifiers')
    all_models_train(x, train_y, CLASSIF_FOLDER)
