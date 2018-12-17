from sklearn import feature_selection as fsel
from sklearn import decomposition as dcmp
from network.gene_network import GeneNetworkPCA

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


def fit_pca(train_x, transf_folder, n_components=500, ret=False):
    """
    Fits PCA to input data. Saves resulting PC transformation, optionally returns it
    :param train_x: Input data
    :param transf_folder: Folder where to save resulting transformation
    :param n_components: Amount of PCs to compute
    :param ret: Bool whether to return result
    :return: Nothing or transformator object if ret=True
    """
    pca_obj = dcmp.PCA(whiten=True, random_state=161, n_components=n_components).fit(train_x)

    savefile = open(transf_folder + 'pca_fit.txt', 'wb')
    pkl.dump(pca_obj, savefile)
    savefile.close()
    if ret:
        return pca_obj


def fit_networkPCA(train_x, transf_folder, network_folder, n_components=500, ret=False,
                   method='gs', attenuation='low-pass', fourier_basis_path=None,
                   gene_name_path='../network/genes_in_data.csv'):
    """
    Fits PCA to input data. Saves resulting PC transformation, optionally returns it
    :param train_x: Input data
    :param transf_folder: Folder in which classifier will be saved
    :param network_folder: Folder containing file node_index.csv and adjacency_sparse.npy for network construction
    :param n_components: Amount of PCs to compute
    :param ret: Bool whether to return result
    :param method: 'gs' or 'gbf', 'gs' for graph sampling, 'gbf' for graph-based filtering
    :param attenuation: Only relevant if method='gbf', either 'low-pass' or 'high-pass'.
    :param fourier_basis_path: If None, fourier basis is computed. Otherwise, path to eigenvalues and eigenvectors saved
    in numpy format. (Tuple, path to eigenvalues is first element)
    :param gene_name_path:
    :return: Nothing or transformator object if ret=True
    """
    netw_pca_obj = GeneNetworkPCA(network_folder + 'adjacency_sparse.npz', network_folder + 'node_index.csv',
                                  n_components=n_components, fourier_basis_path=fourier_basis_path,
                                  method=method, attenuation=attenuation)
    netw_pca_obj = netw_pca_obj.fit(train_x, gene_name_path)
    if method == 'gs':
        filename = 'netw_pca_gs_fit.txt'
    elif method == 'gbf':
        filename = 'netw_pca_gbf_'+attenuation+'fit.txt'
    savefile = open(transf_folder + filename, 'wb')
    pkl.dump(netw_pca_obj, savefile)
    savefile.close()
    if ret:
        return netw_pca_obj


