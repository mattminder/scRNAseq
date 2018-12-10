import numpy as np
import helpers.data_load as dl
import pickle as pkl
from sklearn import decomposition as dcmp

# Loading file
print('Loading Trainset')
DATA_FOLDER = '../../../data/'
RES_FOLDER = '../../../results/pca/'

train_x, gene_names_x, cell_names_x = dl.load_data(DATA_FOLDER + 'train_data.csv.gz')

# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
train_x = train_x[:, 1:]
gene_names_x = gene_names_x[1:]
cell_names_x = cell_names_x[1:]

train_y, cell_names_y = dl.load_response(DATA_FOLDER + 'response.csv.gz')
train_y[train_y == 0] = -1  # Encode as +-1


# Fit PCA
pca_obj = dcmp.PCA(whiten=True, random_state=161, n_components=500).fit(train_x)

# Save PC
savefile = open(RES_FOLDER + 'train_500pc_fit.txt', 'wb')
pkl.dump(pca_obj, savefile)
savefile.close()


# Transform data
train_pca = pca_obj.transform(train_x)
np.save(arr=train_pca, file=RES_FOLDER+'train_500pcs.npy')


# Load Herring 2017 Data
print('Loading Herring 2017')
herring_x, gene_names_herring, cell_names_herring = dl.load_data(DATA_FOLDER + 'herring2017_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
herring_x = herring_x[:, 1:]

# Load Joost 2016 Data
print('Loading Joost 2016')
joost_x, gene_names_joost, cell_names_joost = dl.load_data(DATA_FOLDER + 'joost2016_data.csv.gz')
# TODO: FIX DATA_LOAD FUNCTION SUCH THAT THIS ISN'T NECESSARY
joost_x = joost_x[:, 1:]

# Transformation
print('Predictions')
herring_pca = pca_obj.transform(herring_x)
joost_pca = pca_obj.transform(joost_x)
np.save(arr=joost_pca, file=RES_FOLDER+'joost_500pcs.npy')
np.save(arr=herring_pca, file=RES_FOLDER+'herring_500pcs.npy')

