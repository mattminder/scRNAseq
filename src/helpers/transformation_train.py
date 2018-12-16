from sklearn import feature_selection as fsel
from sklearn import decomposition as dcmp
from .network.gene_network import GeneNetworkPCA

import pickle as pkl


def fit_nolowvar(x, transf_folder, ret=False):
    """
    Removes all features that have a variance below a threshold of 0.1. Saves result,
    optionally returns it
    :param x: Input data
    :param transf_folder: Folder where to save resulting transformation
    :param ret: Bool whether to return result
    :return: Nothing or transformator object if ret=True
    """
    varThresh = fsel.VarianceThreshold(threshold=0.1).fit(x)

    savefile = open(transf_folder + 'nolowvar_fit.pkl', 'wb')
    pkl.dump(varThresh, savefile)
    savefile.close()
    if ret:
        return varThresh


def fit_pca(train_x, classif_folder, n_components=500, ret=False):
    """
    Fits PCA to input data. Saves resulting PC transformation, optionally returns it
    :param train_x:
    :param classif_folder:
    :param n_components: Amount of PCs to compute
    :param ret: Bool whether to return result
    :return: Nothing or transformator object if ret=True
    """
    pca_obj = dcmp.PCA(whiten=True, random_state=161, n_components=n_components).fit(train_x)

    savefile = open(classif_folder + 'pca_fit.txt', 'wb')
    pkl.dump(pca_obj, savefile)
    savefile.close()
    if ret:
        return pca_obj
    
def fit_networkPCA(train_x, transf_folder, n_components=500, ret=False):
    """
    Fits PCA to input data. Saves resulting PC transformation, optionally returns it
    :param train_x:
    :param classif_folder:
    :param n_components: Amount of PCs to compute
    :param ret: Bool whether to return result
    :return: Nothing or transformator object if ret=True
    """
    ### CHANGE FOURIER BASIS PATH TO WHEREVER THE FILES ARE SAVED ONCE IT WAS COMPUTED
    ### method: choose 'gs' or 'gbf'
    ### attenuation: choose 'low-pas' or 'high-pass'
    netw_pca_obj = GeneNetworkPCA('../network/adjacency_sparse.npz','../network/node_index.csv', 
                              n_components=500, fourier_basis_path=None, 
                              method='gs', attenuation='low-pass')
    
    savefile = open(transf_folder + 'netw_pca_fit.txt', 'wb')
    pkl.dump(netw_pca_obj, savefile)
    savefile.close()
    if ret:
        return netw_pca_obj


