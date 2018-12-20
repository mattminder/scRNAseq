from helpers.classifier_train import *
from helpers.data_load import *
from helpers.transformation_train import *


# Loading Training Data
def do_gs_ref_highpass_train(train_x, train_y):
    """
    Trains classifier on high-pass GS transformed data
    :param train_x: Train set
    :param train_y: Train response
    :return: Nothing
    """
    CLASSIF_FOLDER = '../../res/classif/gs-ref_highpass/'
    NETWORK_FOLDER = '../../src/network/'

    # Transforming Data
    print('Transforming Data')
    transf = fit_networkPCA(train_x, CLASSIF_FOLDER, ret=True, network_folder=NETWORK_FOLDER,
                            method='gs-ref', attenuation='high-pass',
                            fourier_basis_path=(NETWORK_FOLDER+'eigvalues_combinatorial.npy',
                                                NETWORK_FOLDER+'eigvectors_combinatorial.npy'))
    x = transf.fit_transform(train_x, '../network/genes_in_data.csv')

    # Training Classifier
    print('Training Classifiers')
    all_models_train(x, train_y, CLASSIF_FOLDER)
