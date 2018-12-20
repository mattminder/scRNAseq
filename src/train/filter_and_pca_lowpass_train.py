from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


# Loading Training Data
def do_filter_and_pca_lowpass_train(train_x, train_y):
    """
    Trains classifier on low-pass filtered, and subsequently pca transformed data
    :param train_x: Train set
    :param train_y: Train response
    :return: Nothnig
    """
    print('Loading Trainset')
    CLASSIF_FOLDER = '../../res/classif/filter_and_pca_low-pass/'
    NETWORK_FOLDER = '../../src/network/'


    # Transforming Data
    print('Transforming Data')
    transf = fit_networkPCA(train_x, CLASSIF_FOLDER, ret=True, network_folder=NETWORK_FOLDER,
                            method='filter', attenuation='low-pass',
                            fourier_basis_path=(NETWORK_FOLDER+'eigvalues_combinatorial.npy',
                                                NETWORK_FOLDER+'eigvectors_combinatorial.npy'))
    filtered = transf.fit_transform(train_x, '../network/genes_in_data.csv')

    pca = fit_pca(filtered, CLASSIF_FOLDER, ret=True)
    x = pca.transform(filtered)


    # Training Classifier
    print('Training Classifiers')
    all_models_train(x, train_y, CLASSIF_FOLDER)
